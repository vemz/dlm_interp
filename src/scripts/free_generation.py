from __future__ import annotations
import csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from src.dlm_interp.load import load_model, nano_forward_fn
from src.dlm_interp.paths import RESULTS
from src.dlm_interp.quality import content_mask, nelbo, sequence_logprob, structural_token_ids
from src.dlm_interp.samplers import score_positions

N_SEEDS = 50
K = 4
GAP = 16
N_MC = 16
BOOT = 2000
RULES = ("confidence", f"confidence@{GAP}", "left_to_right",
         "random", "random_clustered")
REFERENCE = "confidence"
PUBLISHED = {
    "confidence": -3.534,
    "left_to_right": -3.560,
    "random": -3.693,
}
K_SWEEP = (1, 2, 4, 8, 16)
SWEEP_SEEDS = 20

def spaced_topk(scores, positions, k, min_gap):
    order = torch.argsort(scores, descending=True).tolist()
    pos = positions.tolist()
    chosen = []
    for idx in order:
        if all(abs(pos[idx] - pos[other]) >= min_gap for other in chosen):
            chosen.append(idx)
            if len(chosen) == k:
                return torch.tensor(chosen)

    taken = set(chosen)
    while len(chosen) < k and len(taken) < len(order):
        best, best_key = None, None
        for idx in order:
            if idx in taken:
                continue
            distance = min((abs(pos[idx] - pos[other]) for other in chosen),
                           default=10**9)
            key = (distance, float(scores[idx]))
            if best_key is None or key > best_key:
                best, best_key = idx, key
        chosen.append(best)
        taken.add(best)
    return torch.tensor(chosen[:k])

def clustered(positions, k, generator):
    """Select a random masked position and its nearest masked neighbours."""
    seed = int(torch.randint(positions.numel(), (1,), generator=generator))
    distance = (positions - positions[seed]).abs()
    return torch.argsort(distance)[:k]


def pick(rule, conf, masked, k, generator):
    base, _, gap = rule.partition("@")
    if gap:
        return spaced_topk(conf, masked, k, int(gap))
    if base == "confidence":
        return conf.topk(k).indices
    if base == "random":
        return torch.randperm(masked.numel(), generator=generator)[:k]
    if base == "random_clustered":
        return clustered(masked, k, generator)
    if base == "left_to_right":
        return torch.arange(min(k, masked.numel()))
    raise ValueError(rule)

@torch.no_grad()
def generate(fwd, rule, mask_id, seq_len, k, generator):
    x = torch.full((seq_len,), mask_id, dtype=torch.long)
    for _ in range(seq_len // k + 1):
        masked = (x == mask_id).nonzero(as_tuple=False).squeeze(-1)
        if masked.numel() == 0:
            break
        logits_all, _ = fwd(x.unsqueeze(0), 0.0)
        logits = logits_all[masked].float()
        logits[:, mask_id] = float("-inf")
        conf, _, _, _, _ = score_positions(logits)
        take = min(k, masked.numel())
        chosen = pick(rule, conf, masked, take, generator)
        probs = logits[chosen].softmax(-1)
        tokens = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
        x[masked[chosen]] = tokens
    assert not (x == mask_id).any(), f"{rule} left masked positions"
    return x


def k_sweep(fwd, model, mask_id, seq_len, structural):
    # Measure where parallel decoding changes the ranking of rules.
    rules = ("confidence", f"confidence@{GAP}", "random", "left_to_right")
    print("\n" + "=" * 78)
    print(f"K SWEEP — {SWEEP_SEEDS} seeds per cell, sequence_logprob nats/token")
    print("=" * 78)
    head = f"{'k':>4} " + "".join(f"{r[:16]:>17}" for r in rules) + f"{'spaced - conf':>15}"
    print(head)
    print("-" * len(head))
    for k in K_SWEEP:
        cell = {}
        for rule in rules:
            vals = []
            for seed in range(SWEEP_SEEDS):
                g = torch.Generator("cpu").manual_seed(seed)
                x = generate(fwd, rule, mask_id, seq_len, k, g)
                m = content_mask(x, structural)
                vals.append(sequence_logprob(model, x, m)[0] / seq_len)
            cell[rule] = float(np.mean(vals))
        delta = cell[f"confidence@{GAP}"] - cell["confidence"]
        print(f"{k:>4} " + "".join(f"{cell[r]:>17.4f}" for r in rules)
              + f"{delta:>+15.4f}")
        if k == 1:
            assert abs(delta) < 1e-9, (
                "at k = 1 the gap constraint cannot bind — a non-zero difference "
                "means spaced_topk is doing something it should not")

def main():
    from transformers import AutoTokenizer

    model, cfg = load_model()
    mask_id, seq_len = int(cfg["mask_id"]), int(cfg["seq_len"])
    fwd = nano_forward_fn(model)
    tok = AutoTokenizer.from_pretrained("roneneldan/TinyStories-33M")
    structural = structural_token_ids(tok)

    # Generate matched-NFE samples under each selection rule.
    q, qc, nb = ({r: [] for r in RULES} for _ in range(3))
    first = {}
    for seed in range(N_SEEDS):
        for rule in RULES:
            g = torch.Generator("cpu").manual_seed(seed)
            x = generate(fwd, rule, mask_id, seq_len, K, g)
            m = content_mask(x, structural)
            a, b = sequence_logprob(model, x, m)
            q[rule].append(a / seq_len)
            qc[rule].append(b / max(int(m.sum()), 1))
            g2 = torch.Generator("cpu").manual_seed(10_000 + seed)
            nb[rule].append(nelbo(model, x, mask_id, N_MC, g2))
            if seed < 2:
                first.setdefault(rule, []).append(x)
    for rule, xs in first.items():
        assert not torch.equal(xs[0], xs[1]), (
            f"{rule}: two seeds produced identical text — tokens are not being "
            "sampled, and every comparison below is vacuous")

    idx = np.random.default_rng(12345).integers(0, N_SEEDS, size=(BOOT, N_SEEDS))

    def table(name, data, published=False):
        print("\n" + "=" * 78)
        print(name)
        print("=" * 78)
        base = np.array(data[REFERENCE])
        head = (f"{'rule':>18} {'nats/token':>11} {'vs confidence':>14} "
                f"{'95% CI':>22} {'P(>0)':>7}" + ("   published" if published else ""))
        print(head)
        print("-" * len(head))
        rows = []
        for rule in RULES:
            v = np.array(data[rule])
            d = v - base
            draws = d[idx].mean(axis=1)
            lo, hi = np.percentile(draws, [2.5, 97.5])
            line = (f"{rule:>18} {v.mean():>11.4f} {d.mean():>+14.4f} "
                    f"{f'({lo:+.4f}, {hi:+.4f})':>22} {float((draws > 0).mean()):>7.3f}")
            if published:
                p = PUBLISHED.get(rule.partition('@')[0] if '@' not in rule else None)
                p = PUBLISHED.get(rule)
                line += f"   {p:+.3f}" if p is not None else "   —"
            print(line)
            rows.append([name, rule, f"{v.mean():.4f}", f"{d.mean():.4f}",
                         f"{lo:.4f}", f"{hi:.4f}", f"{float((draws > 0).mean()):.4f}"])
        return rows

    rows = table(f"sequence_logprob (all tokens) — the published column is k = 1, "
                 f"this is k = {K}", q, published=True)
    rows += table("sequence_logprob (content tokens only)", qc)
    rows += table("NELBO — the model's own objective, lower is better so a "
                  "NEGATIVE delta wins", nb)

    target = np.array(q[f"confidence@{GAP}"]) - np.array(q[REFERENCE])
    draws = target[idx].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    print("\npre-registered test")
    print(f"  confidence@{GAP} minus confidence, sequence_logprob: "
          f"{target.mean():+.4f} ({lo:+.4f}, {hi:+.4f})")
    print(f"  minimum interesting effect: +0.0200")
    if target.mean() >= 0.02 and lo > 0:
        print("PASS")
    elif lo > 0:
        print("BELOW_MINIMUM")
    else:
        print("FAIL")

    if "--k-sweep" in sys.argv:
        k_sweep(fwd, model, mask_id, seq_len, structural)

    out = RESULTS / "free_generation.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["metric", "rule", "mean", "delta_vs_confidence",
                    "boot_lo", "boot_hi", "p_positive"])
        w.writerows(rows)
    print(f"saved {out}")

if __name__ == "__main__":
    main()
