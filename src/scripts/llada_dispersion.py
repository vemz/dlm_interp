from __future__ import annotations
import argparse
import csv
import re
import sys
import numpy as np
import torch

MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
MASK_ID = 126336         
BOOT = 2000

def spaced_topk(scores, positions, k, min_gap):
    order = torch.argsort(scores, descending=True).tolist()
    pos = positions.tolist()
    chosen, taken = [], set()
    for idx in order:
        if len(chosen) >= k:
            break
        if all(abs(pos[idx] - pos[c]) >= min_gap for c in chosen):
            chosen.append(idx)
            taken.add(idx)
    while len(chosen) < k and len(taken) < len(order):
        best, best_key = None, None
        for idx in order:
            if idx in taken:
                continue
            d = min((abs(pos[idx] - pos[c]) for c in chosen), default=10 ** 9)
            key = (d, float(scores[idx]))
            if best_key is None or key > best_key:
                best, best_key = idx, key
        chosen.append(best)
        taken.add(best)
    return torch.tensor(chosen[:k], dtype=torch.long)


def gap_for(n_masked, k, alpha=0.5):
    return max(1, int(alpha * n_masked / max(k, 1)))


def select(rule, conf, positions, k, alpha, generator):
    base, _, tag = rule.partition("@")
    if tag:
        g = gap_for(positions.numel(), k, alpha) if tag == "gap" else int(tag)
        return spaced_topk(conf, positions, k, g)
    if base == "confidence":
        return conf.topk(min(k, conf.numel())).indices
    if base == "random":
        return torch.randperm(positions.numel(), generator=generator)[:k]
    if base == "left_to_right":
        return torch.arange(min(k, positions.numel()))
    raise ValueError(rule)

def transfer_schedule(block_len, steps):
    base = block_len // steps
    out = [base] * steps
    for i in range(block_len - base * steps):
        out[i] += 1
    return out

BIND_GAPS = (2, 4, 8, 16, 32, 64)


def step_geometry(sel_positions):
    p = sorted(sel_positions)
    if len(p) < 2:
        return None
    pair = [abs(a - b) for i, a in enumerate(p) for b in p[i + 1:]]
    adj = sum(1 for i, a in enumerate(p)
              if (i and a - p[i - 1] == 1) or (i + 1 < len(p) and p[i + 1] - a == 1))
    out = {"mean_dist": float(np.mean(pair)), "min_gap": float(min(pair)),
           "adjacent_frac": adj / len(p)}
    for g in BIND_GAPS:
        out[f"binds_{g}"] = float(min(pair) < g)
    return out

@torch.no_grad()
def generate(model, prompt_ids, rule, gen_len, block_len, steps_per_block,
             mask_id, temperature, alpha, generator, device, trace=None):
    p = prompt_ids.numel()
    x = torch.full((1, p + gen_len), mask_id, dtype=torch.long, device=device)
    x[0, :p] = prompt_ids.to(device)
    prompt_copy = x[0, :p].clone()

    u_table = torch.rand(gen_len, generator=generator).clamp(1e-6, 1.0 - 1e-6)

    n_blocks = gen_len // block_len
    assert n_blocks * block_len == gen_len, "gen_len must be a multiple of block_len"

    for b in range(n_blocks):
        lo, hi = p + b * block_len, p + (b + 1) * block_len
        for k_step in transfer_schedule(block_len, steps_per_block):
            block = x[0, lo:hi]
            masked_local = (block == mask_id).nonzero(as_tuple=False).squeeze(-1)
            if masked_local.numel() == 0:
                break
            logits = model(x).logits[0, lo:hi][masked_local].float()
            logits[:, mask_id] = float("-inf")

            probs = logits.softmax(-1)
            conf = probs.max(-1).values.cpu()

            take = min(k_step, masked_local.numel())
            pick = select(rule, conf, masked_local.cpu(), take, alpha,
                          generator).to(device)
            sel = masked_local[pick]                    # index within the block

            if temperature > 0:
                q = (logits[pick] / temperature).softmax(-1)
                u = u_table[(lo + sel - p).cpu()].to(q.device).unsqueeze(-1)
                tokens = (q.cumsum(-1) < u).sum(-1).clamp_(0, q.shape[-1] - 1)
            else:
                tokens = logits[pick].argmax(-1)
            assert not (tokens == mask_id).any(), "sampled the mask token"

            if trace is not None:
                g = step_geometry((lo + sel - p).tolist())
                if g is not None:
                    trace.append(g)

            x[0, lo + sel] = tokens

    assert not (x[0, p:] == mask_id).any(), f"{rule}: masked positions survived"
    assert torch.equal(x[0, :p], prompt_copy), f"{rule}: prompt was overwritten"
    return x[0], x[0, p:]

@torch.no_grad()
def nelbo_gen(model, ids, p, mask_id, n_mc, generator, device):
    L = ids.numel()
    total = 0.0
    for j in range(n_mc):
        t = (j + 0.5) / n_mc
        keep = torch.rand(L - p, generator=generator) >= t
        if bool(keep.all()):
            continue
        x = ids.clone().unsqueeze(0)
        gen = x[0, p:]
        x[0, p:] = torch.where(keep.to(device), gen,
                               torch.full_like(gen, mask_id))
        pos = (~keep).nonzero(as_tuple=False).squeeze(-1).to(device) + p
        logp = model(x).logits[0][pos].float().log_softmax(-1)
        nll = -logp.gather(-1, ids[pos].unsqueeze(-1)).squeeze(-1).sum()
        total += float(nll) / t
    return total / n_mc / (L - p)

def distinct_n(ids, n):
    if len(ids) < n:
        return float("nan")
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return len(set(grams)) / len(grams)

def rep_rate(ids, w=32):
    return sum(1 for i in range(len(ids)) if ids[i] in ids[max(0, i - w):i]) / len(ids)

def diversity(ids):
    ids = ids.tolist()
    return {"distinct_1": distinct_n(ids, 1), "distinct_2": distinct_n(ids, 2),
            "distinct_3": distinct_n(ids, 3), "rep_32": rep_rate(ids)}

def real_reference(tok, n_windows, seq_len, corpus):
    from datasets import load_dataset
    ds = load_dataset(corpus, "wikitext-103-raw-v1", split="test") \
        if "wikitext" in corpus else load_dataset(corpus, split="test")
    ids = []
    for row in ds:
        t = row.get("text") or ""
        if len(t) > 200:
            ids.extend(tok(t)["input_ids"])
        if len(ids) > n_windows * seq_len + seq_len:
            break
    out = [diversity(torch.tensor(ids[i * seq_len:(i + 1) * seq_len]))
           for i in range(n_windows)]
    return {k: float(np.mean([o[k] for o in out])) for k in out[0]}

ANS = re.compile(r"(-?[\d,]*\d)")

def extract_answer(text):
    if "####" in text:
        hits = ANS.findall(text.split("####")[1].replace(",", ""))
        return hits[0] if hits else None
    hits = ANS.findall(text.replace(",", ""))
    return hits[-1] if hits else None


def gsm8k_eval(model, tok, rules, n_problems, args, device):
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    n = min(n_problems, len(ds))
    if n < n_problems:
        print(f"  GSM8K test has {len(ds)} problems; running {n}")
    ds = ds.select(range(n))
    acc = {r: [] for r in rules}
    fmt = {r: [] for r in rules}
    for i, row in enumerate(ds):
        msg = [{"role": "user", "content": row["question"] +
                "\n\nReason step by step and end with '#### <answer>'."}]
        text = tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
        prompt_ids = torch.tensor(tok(text)["input_ids"], dtype=torch.long)
        gold = extract_answer(row["answer"])
        for rule in rules:
            g = torch.Generator("cpu").manual_seed(0)
            _, gen = generate(model, prompt_ids, rule, args.gen_len, args.block_len,
                              args.steps_per_block, args.mask_id, 0.0, args.alpha,
                              g, device)
            out = tok.decode(gen.tolist(), skip_special_tokens=True)
            fmt[rule].append(float("####" in out))
            pred = extract_answer(out)
            acc[rule].append(float(pred is not None and gold is not None
                                   and pred == gold))
    return acc, fmt

def load_llada(model_id, device):
    import transformers
    from transformers import AutoModel, AutoTokenizer

    ver = transformers.__version__
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = None
    for key in ("dtype", "torch_dtype"):        # renamed in 5.x, deprecated in 4.5x
        try:
            model = AutoModel.from_pretrained(
                model_id, trust_remote_code=True, **{key: dtype})
            break
        except TypeError:
            continue
        except AttributeError as exc:
            if "all_tied_weights_keys" in str(exc):
                raise SystemExit(
                    f"\ntransformers {ver} is too new for LLaDA's remote modeling "
                    "code.\nThe weights downloaded fine and are cached; only the "
                    "loader is\nincompatible. Fix:\n\n"
                    "    pip install 'transformers<5'\n\n"
                    "then re-run — nothing re-downloads.\n")
            raise
    if model is None:
        raise SystemExit("could not load the model with either dtype keyword")
    return model.to(device).eval(), tok

def selfcheck(model, tok, args, device):
    """Five things that must hold before any number below means anything."""
    print("=" * 74)
    print("SELFCHECK")
    print("=" * 74)
    ok = True

    prompt = torch.tensor(tok(tok.apply_chat_template(
        [{"role": "user", "content": "Tell me a short story."}],
        add_generation_prompt=True, tokenize=False))["input_ids"], dtype=torch.long)

    x = torch.full((1, prompt.numel() + 32), args.mask_id, dtype=torch.long, device=device)
    x[0, :prompt.numel()] = prompt.to(device)
    with torch.no_grad():
        logits = model(x).logits
    print(f"  forward shape {tuple(logits.shape)}  vocab {logits.shape[-1]}")
    if logits.shape[:2] != x.shape:
        print("  FAIL: logits do not align with input")
        ok = False
    if args.mask_id >= logits.shape[-1]:
        print(f"  FAIL: mask id {args.mask_id} outside vocab — pass --mask-id")
        ok = False
    else:
        m = logits[0, prompt.numel():].float().softmax(-1)[:, args.mask_id].mean()
        print(f"  mean p(mask token) at masked positions: {m:.2e} (want ~0)")
        if m > 0.01:
            print("  FAIL: the model wants to emit the mask token — wrong mask id?")
            ok = False

    a = generate(model, prompt, "confidence", 32, 32, 32, args.mask_id, 0.0,
                 args.alpha, torch.Generator("cpu").manual_seed(0), device)[1]
    b = generate(model, prompt, "confidence@gap", 32, 32, 32, args.mask_id, 0.0,
                 args.alpha, torch.Generator("cpu").manual_seed(0), device)[1]
    same = torch.equal(a, b)
    print(f"  k=1 constrained == unconstrained: {same}")
    if not same:
        print("  FAIL: at one commit per step the constraint cannot bind")
        ok = False

    # Sampling must actually sample, or every comparison is between identical texts.
    c = generate(model, prompt, "confidence", 32, 32, 8, args.mask_id, 1.0,
                 args.alpha, torch.Generator("cpu").manual_seed(0), device)[1]
    d = generate(model, prompt, "confidence", 32, 32, 8, args.mask_id, 1.0,
                 args.alpha, torch.Generator("cpu").manual_seed(1), device)[1]
    print(f"  two seeds differ at temperature 1: {not torch.equal(c, d)}")
    if torch.equal(c, d):
        print("  FAIL: generations are deterministic; the comparison is vacuous")
        ok = False

    uniq = len(set(c.tolist()))
    print(f"  distinct tokens in a 32-token sample: {uniq}")
    if uniq <= 2:
        print("  FAIL: the sampler is emitting a near-constant token — look at the")
        print("        token draw, not at the selection rule")
        ok = False

    print(f"\n  sample: {tok.decode(c.tolist(), skip_special_tokens=True)[:120]!r}")
    print(f"\n  -> {'PASS' if ok else 'FAIL — fix before running anything else'}")
    return ok

def run_config(model, tok, name, block_len, args, device, real=None):
    rules = ("confidence", "confidence@gap", "random", "left_to_right")
    prompts = PROMPTS
    k0 = transfer_schedule(block_len, args.steps_per_block)[0]
    print("\n" + "=" * 74)
    print(f"CONFIG {name} — block_length = {block_len}, gen_len = {args.gen_len}, "
          f"{args.steps_per_block} steps/block, k = {k0}")
    print("=" * 74)
    print(f"  distance constraint at the first step: gap = "
          f"{gap_for(block_len, k0, args.alpha)} tokens "
          f"(alpha={args.alpha}, max satisfiable ~{block_len // k0})")

    nb, dv = {r: [] for r in rules}, {r: [] for r in rules}
    first = {r: [] for r in rules}
    for seed in range(args.seeds):
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompts[seed % len(prompts)]}],
            add_generation_prompt=True, tokenize=False)
        prompt_ids = torch.tensor(tok(text)["input_ids"], dtype=torch.long)
        for rule in rules:
            g = torch.Generator("cpu").manual_seed(seed)
            full, gen = generate(model, prompt_ids, rule, args.gen_len, block_len,
                                 args.steps_per_block, args.mask_id, args.temperature,
                                 args.alpha, g, device)
            g2 = torch.Generator("cpu").manual_seed(10_000 + seed)
            nb[rule].append(nelbo_gen(model, full, prompt_ids.numel(), args.mask_id,
                                      args.n_mc, g2, device))
            dv[rule].append(diversity(gen))
            if seed < 2:
                first[rule].append(gen)

    for rule, xs in first.items():
        assert not torch.equal(xs[0], xs[1]), (
            f"{rule}: two seeds identical — not sampling, comparison vacuous")

    idx = np.random.default_rng(12345).integers(0, args.seeds, size=(BOOT, args.seeds))
    base = np.array(nb["confidence"])
    print(f"\n{'rule':>18} {'NELBO':>9} {'vs conf':>10} {'95% CI':>22}   "
          f"{'d-2':>7} {'rep32':>7}")
    print("-" * 82)
    rows = []
    for rule in rules:
        v = np.array(nb[rule])
        d = v - base
        draws = d[idx].mean(axis=1)
        lo, hi = np.percentile(draws, [2.5, 97.5])
        d2 = np.mean([x["distinct_2"] for x in dv[rule]])
        rp = np.mean([x["rep_32"] for x in dv[rule]])
        print(f"{rule:>18} {v.mean():>9.4f} {d.mean():>+10.4f} "
              f"{f'({lo:+.4f}, {hi:+.4f})':>22}   {d2:>7.4f} {rp:>7.4f}")
        rows.append([name, rule, f"{v.mean():.4f}", f"{d.mean():.4f}",
                     f"{lo:.4f}", f"{hi:.4f}", f"{d2:.4f}", f"{rp:.4f}"])

    if real is not None:
        print("-" * 82)
        print(f"{'REAL TEXT':>18} {'—':>9} {'—':>10} {'—':>22}   "
              f"{real['distinct_2']:>7.4f} {real['rep_32']:>7.4f}")
        rows.append([name, "real_text", "", "", "", "",
                     f"{real['distinct_2']:.4f}", f"{real['rep_32']:.4f}"])

        print(f"\n  distance from real text (lower is better, either direction):")
        print(f"    {'rule':>16} {'|d-2 − real|':>14} {'|rep32 − real|':>16}")
        side = []
        for r in rules:
            a = np.mean([x["distinct_2"] for x in dv[r]])
            b = np.mean([x["rep_32"] for x in dv[r]])
            print(f"    {r:>16} {abs(a - real['distinct_2']):>14.4f} "
                  f"{abs(b - real['rep_32']):>16.4f}")
            side.append(a < real["distinct_2"])
        print(f"    {'(all rules ' + ('BELOW' if all(side) else 'ABOVE' if not any(side) else 'MIXED vs') + ' real text on distinct-2)':>50}")

    t = np.array(nb["confidence@gap"]) - base
    draws = t[idx].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    print(f"\n  PRE-REGISTERED: confidence@gap - confidence = {t.mean():+.4f} "
          f"({lo:+.4f}, {hi:+.4f})   [NELBO, so NEGATIVE wins]")

    worse = []
    for col in ("distinct_2", "rep_32"):
        a = np.array([x[col] for x in dv["confidence@gap"]])
        b = np.array([x[col] for x in dv["confidence"]])
        dd = (a - b)[idx].mean(axis=1)
        l2, h2 = np.percentile(dd, [2.5, 97.5])
        line = (f"  {col:>14} confidence@gap - confidence = {(a - b).mean():+.4f} "
                f"({l2:+.4f}, {h2:+.4f})")
        if real is not None:
            ddist = (np.abs(a[idx].mean(axis=1) - real[col])
                     - np.abs(b[idx].mean(axis=1) - real[col]))
            l3, h3 = np.percentile(ddist, [2.5, 97.5])
            if h3 < 0:
                line += "   toward real"
            elif l3 > 0:
                line += "   AWAY from real"
                worse.append(col)
        print(line)
    if worse:
        print(f"\n  !! the constraint moves {', '.join(worse)} AWAY from real text.")
        print("     A NELBO gain alongside that is not a quality gain. Sweep the")
        print("     gap (--gap-sweep) before quoting it: too large a gap leaves the")
        print("     criterion no freedom and the rule collapses toward a fixed")
        print("     schedule, which NELBO rewards and readers would not.")

    width = hi - lo
    if hi < 0:
        print("  -> constraint WINS on this configuration.")
    elif lo > 0:
        print("  -> constraint LOSES. If this is config A, §3.7 does not "
              "generalise and the paper's scope shrinks.")
    elif abs(t.mean()) > 0.25 * width:
        print(f"  -> UNDERPOWERED, not null. The interval is {width:.2f} wide "
              f"around an effect of {t.mean():+.2f};")
        print(f"     at {args.seeds} seeds that cannot resolve it. Re-run with "
              f"--seeds {max(50, args.seeds * 3)} before calling it either way,")
        print("     and read the diversity deltas above: if they move together "
              "with the NELBO point estimate, the effect is real and the test")
        print("     is what is weak.")
    else:
        print("  -> null. If this is config B with A positive, that is the "
              "'blocks already do it' result — report it, do not bury it.")
    return rows

PROMPTS = [
    "Tell me a short story about a lighthouse keeper.",
    "Explain why the sky is blue, in one paragraph.",
    "Write a short recipe for lentil soup.",
    "Describe a busy train station at night.",
    "Summarise the causes of the 1929 crash in a paragraph.",
    "Write instructions for changing a bicycle tyre.",
    "What is the difference between weather and climate?",
    "Write a polite email declining a meeting invitation.",
    "Describe how a refrigerator works.",
    "Write a short dialogue between a shopkeeper and a lost child.",
    "Explain recursion to someone who has never programmed.",
    "Give three arguments for and against remote work.",
]

SWEEP_GAPS = (2, 4, 8, 16, 32, 64)

def steps_for(block_len, args):
    if block_len == args.gen_len:
        return (args.gen_len // args.block_len) * args.steps_per_block
    return args.steps_per_block


def geometry(model, tok, name, block_len, args, device):
    rules = ("confidence", "random", "left_to_right", "confidence@gap")
    print("\n" + "=" * 88)
    print(f"GEOMETRY — config {name}, block {block_len}, {args.seeds} seeds")
    print("=" * 88)
    head = (f"{'rule':>16} {'mean dist':>10} {'min gap':>8} {'adj frac':>9}  "
            + " ".join(f"{'b' + str(g):>6}" for g in BIND_GAPS))
    print(head)
    print("-" * len(head))
    rows = []
    for rule in rules:
        tr = []
        for seed in range(args.seeds):
            text = tok.apply_chat_template(
                [{"role": "user", "content": PROMPTS[seed % len(PROMPTS)]}],
                add_generation_prompt=True, tokenize=False)
            pid = torch.tensor(tok(text)["input_ids"], dtype=torch.long)
            g = torch.Generator("cpu").manual_seed(seed)
            generate(model, pid, rule, args.gen_len, block_len,
                     args.steps_per_block, args.mask_id, args.temperature,
                     args.alpha, g, device, trace=tr)
        agg = {k: float(np.mean([t[k] for t in tr])) for k in tr[0]}
        print(f"{rule:>16} {agg['mean_dist']:>10.2f} {agg['min_gap']:>8.2f} "
              f"{agg['adjacent_frac']:>9.3f}  "
              + " ".join(f"{agg['binds_' + str(g)]:>6.3f}" for g in BIND_GAPS))
        rows.append([f"geometry_{name}", rule, f"{agg['mean_dist']:.2f}",
                     f"{agg['min_gap']:.2f}", f"{agg['adjacent_frac']:.3f}", "",
                     "", ""])
    print("\n  adj frac is DUS Appendix B.7's metric — directly comparable to their")
    print("  Table 11 (they report 49-58% for self-confidence at B = 16, 32).")
    print("  b<g> is the fraction of steps where a gap-g constraint would bind. A")
    print("  sweep row at a gap whose b is 0.000 measured nothing at all.")
    return rows

def gap_sweep(model, tok, block_len, args, device, real):
    spb = steps_for(block_len, args)
    k0 = transfer_schedule(block_len, spb)[0]
    print("\n" + "=" * 78)
    print(f"GAP SWEEP — block {block_len}, {spb} steps, k = {k0}, "
          f"{args.seeds} seeds, prompts fixed")
    print("=" * 78)
    usable = [g for g in SWEEP_GAPS if g <= block_len // k0]
    if len(usable) < len(SWEEP_GAPS):
        print(f"  gaps above {block_len // k0} are unsatisfiable with k = {k0} in a"
              f" {block_len}-token span; sweeping {usable}")
    head = (f"{'rule':>8} {'gap':>5} {'NELBO':>9} {'|d2-real|':>11} "
            f"{'95% CI':>20} {'|rep-real|':>11} {'95% CI':>20}")
    print(head)
    print("-" * len(head))
    out, rec = [], []
    for g_abs in (None,) + tuple(usable):
        rule = "confidence" if g_abs is None else f"confidence@{g_abs}"
        a = args.alpha
        nb, d2, rp = [], [], []
        for seed in range(args.seeds):
            text = tok.apply_chat_template(
                [{"role": "user", "content": PROMPTS[seed % len(PROMPTS)]}],
                add_generation_prompt=True, tokenize=False)
            pid = torch.tensor(tok(text)["input_ids"], dtype=torch.long)
            g = torch.Generator("cpu").manual_seed(seed)
            full, gen = generate(model, pid, rule, args.gen_len, block_len,
                                 spb, args.mask_id,
                                 args.temperature, a, g, device)
            g2 = torch.Generator("cpu").manual_seed(10_000 + seed)
            nb.append(nelbo_gen(model, full, pid.numel(), args.mask_id,
                                args.n_mc, g2, device))
            m = diversity(gen)
            d2.append(m["distinct_2"])
            rp.append(m["rep_32"])
        gap = "—" if g_abs is None else g_abs
        lbl = "conf" if g_abs is None else f"gap {g_abs}"

        idx = np.random.default_rng(4242).integers(0, args.seeds,
                                                   size=(BOOT, args.seeds))
        dd = pp = float("nan")
        dlo = dhi = plo = phi = float("nan")
        if real:
            dboot = np.abs(np.array(d2)[idx].mean(axis=1) - real["distinct_2"])
            pboot = np.abs(np.array(rp)[idx].mean(axis=1) - real["rep_32"])
            dd, pp = abs(np.mean(d2) - real["distinct_2"]), \
                abs(np.mean(rp) - real["rep_32"])
            dlo, dhi = np.percentile(dboot, [2.5, 97.5])
            plo, phi = np.percentile(pboot, [2.5, 97.5])
        print(f"{lbl:>8} {str(gap):>5} {np.mean(nb):>9.4f} {dd:>11.4f} "
              f"{f'({dlo:.4f}, {dhi:.4f})':>20} {pp:>11.4f} "
              f"{f'({plo:.4f}, {phi:.4f})':>20}")

        out.append([f"gap_sweep_{block_len}", rule, f"{np.mean(nb):.4f}",
                    "", "", "",
                    f"{np.mean(d2):.4f}", f"{np.mean(rp):.4f}", "", "",
                    f"{dd:.4f}", f"{dlo:.4f}", f"{dhi:.4f}"])
        rec.append((lbl, np.mean(nb), dd, dlo, dhi))
    if real:
        print("-" * len(head))
        print(f"{'REAL':>8} {'—':>5} {'—':>9} {0.0:>11.4f} "
              f"{'—':>20} {0.0:>11.4f} {'—':>20}")
        best = min(rec, key=lambda r: r[2])
        cheapest = min(rec, key=lambda r: r[1])
        print(f"\n  closest to real text : {best[0]}   "
              f"(|d2-real| {best[2]:.4f})")
        print(f"  best NELBO           : {cheapest[0]}   "
              f"(NELBO {cheapest[1]:.4f})")
        if best[0] != cheapest[0]:
            print("\n  THE TWO DISAGREE. NELBO keeps rewarding spacing past the point")
            print("  where the text stops looking like text: beyond the dependence")
            print("  range the constraint is no longer removing dependence, it is")
            print("  removing the criterion. Quote the diversity optimum, and treat")
            print("  a likelihood-only sweep as unable to see this.")
            overlap = [r[0] for r in rec if r[3] <= best[4] and r[0] != best[0]]
            if overlap:
                print(f"  Intervals overlapping the optimum: {', '.join(overlap)} —")
                print("  the location of the minimum is not resolved at this n.")
    return out

def gsm8k_tier(model, tok, args, device):
    k0 = transfer_schedule(args.block_len, args.steps_per_block)[0]
    nfe = (args.gen_len // args.block_len) * args.steps_per_block
    g0 = gap_for(args.block_len, k0, args.alpha)

    rules = ("confidence", "random") if k0 == 1 else \
            ("confidence", "confidence@gap", "confidence@8")

    fixed = [int(r.split("@")[1]) for r in rules
             if "@" in r and not r.endswith("@gap")]
    maxsat = args.block_len // max(k0, 1)

    print("\n" + "=" * 78)
    print(f"GSM8K — {args.gsm8k} problems, block {args.block_len}, k = {k0}, "
          f"{nfe} NFE, greedy")
    print(f"  rules: {', '.join(rules)}")
    if k0 == 1:
        print("  k = 1 -> ceiling arm; the gap constraint is a no-op and is dropped.")
    else:
        if any(r.endswith("@gap") for r in rules):
            print(f"  confidence@gap: first-step gap = {g0} tokens (alpha="
                  f"{args.alpha}), decaying with n_masked to 1")
            if g0 < 2:
                print("  !! gap < 2 at the first step constrains nothing, and it "
                      "only shrinks from there. Raise --alpha.")
        if fixed:
            print(f"  absolute gaps: {', '.join(map(str, fixed))} tokens, held for "
                  f"the whole block; --alpha is NOT read by these arms")
            over = [g for g in fixed if g > maxsat]
            if over:
                print(f"  !! gap(s) {over} exceed what k = {k0} can satisfy in "
                      f"{args.block_len} tokens (max ~{maxsat}); the maximin "
                      "fallback runs on most steps")
    if args.config:
        print(f"  NOTE: --config {'/'.join(args.config)} is IGNORED here. This tier "
              f"runs at --block-len {args.block_len}; pass --block-len "
              f"{args.gen_len} for the global configuration.")
    print("=" * 78)

    acc, fmt = gsm8k_eval(model, tok, rules, args.gsm8k, args, device)

    n = len(acc["confidence"])
    idx = np.random.default_rng(7).integers(0, n, size=(BOOT, n))
    base = np.array(acc["confidence"])
    print(f"\n{'rule':>18} {'acc':>7} {'vs conf':>9} {'95% CI':>20} "
          f"{'b/c':>9} {'fmt':>7}")
    print("-" * 74)
    rows = []
    for rule in rules:
        v = np.array(acc[rule])
        d = v - base
        draws = d[idx].mean(axis=1)
        lo, hi = np.percentile(draws, [2.5, 97.5])

        b, c = int((d > 0).sum()), int((d < 0).sum())
        f = float(np.mean(fmt[rule]))
        print(f"{rule:>18} {v.mean():>7.3f} {d.mean():>+9.3f} "
              f"{f'({lo:+.3f}, {hi:+.3f})':>20} {f'{b}/{c}':>9} {f:>7.3f}")
        rows.append(["gsm8k", rule, f"{v.mean():.4f}", f"{d.mean():.4f}",
                     f"{lo:.4f}", f"{hi:.4f}", "", "", f"{f:.4f}", f"{b}/{c}"])

    print("\n  b/c = problems this rule got right and confidence got wrong / the")
    print("  reverse. The paired 95% CI is roughly +-1.96*sqrt(b+c)/n, so it is")
    print("  b + c and not n that says what this run can resolve.")
    print("  fmt = fraction of generations containing '####'. A rule that wins on")
    print("  accuracy AND on fmt won on formatting: the extra correct answers are")
    print("  ones the extractor could newly find, not ones the model newly got")
    print("  right. Report both columns or the headline number is not what it says.")
    if k0 == 1:
        print("\n  This is the ceiling. Read it against the k > 1 run as:")
        print("    recovered = (gap_acc - conf_acc) / (ceiling_acc - conf_acc)")
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--mask-id", type=int, default=MASK_ID)
    ap.add_argument("--config", action="append", choices=["A", "B"], default=[])
    ap.add_argument("--gen-len", type=int, default=256)
    ap.add_argument("--block-len", type=int, default=32)
    ap.add_argument("--steps-per-block", type=int, default=8)   # k = 32/8 = 4
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--n-mc", type=int, default=16)
    ap.add_argument("--gsm8k", type=int, default=0)
    ap.add_argument("--corpus", default="wikitext",
                    help="HF dataset for the real-text diversity reference row")
    ap.add_argument("--no-reference", action="store_true")
    ap.add_argument("--gap-sweep", action="store_true")
    ap.add_argument("--geometry", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--out", default="llada_dispersion.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {args.model} on {device} ...", flush=True)
    model, tok = load_llada(args.model, device)

    if args.selfcheck or not (args.config or args.gsm8k or args.gap_sweep or args.geometry):
        if not selfcheck(model, tok, args, device):
            sys.exit(1)
        if not (args.config or args.gsm8k or args.gap_sweep or args.geometry):
            return

    real = None
    if (args.config or args.gap_sweep) and not args.no_reference:
        try:
            real = real_reference(tok, 200, args.gen_len, args.corpus)
            print(f"  real-text reference from {args.corpus}: "
                  f"d-2 {real['distinct_2']:.4f}, rep32 {real['rep_32']:.4f}")
        except Exception as exc:                     # noqa: BLE001
            print(f"  (no real-text reference: {exc}; diversity columns are "
                  "uninterpretable without it — see --corpus)")

    rows = []
    for name in args.config:
        bl = args.gen_len if name == "A" else args.block_len
        spb = steps_for(bl, args)
        rows += run_config(model, tok, name, bl,
                           argparse.Namespace(**{**vars(args), "steps_per_block": spb}),
                           device, real)

    if args.geometry:
        for nm, bl in (("A", args.gen_len), ("B", args.block_len)):
            spb = steps_for(bl, args)
            rows += geometry(model, tok, nm, bl,
                             argparse.Namespace(**{**vars(args),
                                                   "steps_per_block": spb}), device)

    if args.gap_sweep:
        rows += gap_sweep(model, tok, args.gen_len, args, device, real)

    if args.gsm8k:
        rows += gsm8k_tier(model, tok, args, device)

    header = ["config", "rule", "value", "delta_vs_confidence",
              "boot_lo", "boot_hi", "distinct_2", "rep_32",
              "fmt_rate", "discordance_b_c",
              "dist_from_real", "dist_lo", "dist_hi"]
    with open(args.out, "w", newline="") as h:
        w = csv.writer(h)
        w.writerow(header)
        w.writerows([r + [""] * (len(header) - len(r)) for r in rows])
    print(f"\nwrote {args.out}")

if __name__ == "__main__":
    main()
