"""Readiness along the sampler's own trajectory, and what a different order buys.

Part 1 measures readiness on i.i.d. Bernoulli masks over ground-truth text. The
sampler never produces such a state: its masks are confidence-structured and its
canvas fills with generated tokens. Two things follow that this script fixes.

1. A probe direction fitted on Part 1 states is not obviously the direction that
   exists during decoding, and RQ2 patches during decoding. The pre-registration
   asks for a probe "fitted on the training generations"; there was no such probe.

2. Part 2 only starts probing once the horizon buffer is full -- step 16 of 64 --
   so it says nothing about the early phase, which is exactly where the
   output-side study locates the damage. This probes from step 0.

**Teacher-forced trajectories.** Decode with confidence top-k on held-out text,
but write the *true* token at every committed position. The sampler decides
where, the ground truth decides what. That keeps `gt_margin` and `wait_gain`
defined all the way down -- they need a true token, which free generation does
not have -- while the mask pattern is the sampler's own.

It also fixes the text, which makes the ordering question exactly computable:
the same text under different rules gives different sums of log p, with no
sampling noise anywhere.

Two modes.

    python src/scripts/collect_trajectory.py collect
        -> runs/labels_trajectory.pt
        Both targets on the *same rows*, unlike Part 1 where they live in two
        files with different rows: the dissociation can then be differenced
        per row rather than per window.

    python src/scripts/collect_trajectory.py orders
        -> results/ordering_rules.csv
        The same text decoded under seven position-selection rules. Add
        --penalty for the one-step commit penalty per rule, which is the only
        metric here that is fair to the oracles -- see the note the script
        prints above its own table.

Both honour DLM_CKPT and DLM_TAG.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from src.dlm_interp.load import load_model, nano_forward_fn
from src.dlm_interp.paths import DATA, RESULTS, RUNS
from src.dlm_interp.samplers import HiddenCapture, score_positions

VAL_BIN = "tinystories/val.bin"          # under paths.DATA
OUT = RUNS / "labels_trajectory.pt"
ORDERS_CSV = RESULTS / "ordering_rules.csv"

N_TEXTS = 200               # collect mode
N_TEXTS_ORDERS = 100        # orders mode; every rule replays every text
K = 4
PROBE_EVERY = 4             # record a row only every PROBE_EVERY steps
MAX_ROWS_PER_STEP = 16      # subsample positions; keeps the file near Part 1's size
BLOCK_LAYERS = (1, 3, 5)    # ln_f is appended automatically, as in collect_waitgain
SEED = 0

RULES = ("confidence", "confidence_spaced", "gt_margin", "oracle_logp",
         "wait_gain", "wait_gain_resid", "random", "random_clustered",
         "left_to_right")

# --sweep replaces RULES with a ladder of FIXED minimum gaps. The point is not to
# tune a hyperparameter: the gap at which the penalty stops falling is a
# measurement of how far the model's commit dependence reaches, and the shape of
# the curve is what says whether the Part 3 result rests on one lucky value.
MIN_GAPS = (2, 4, 8, 16, 32, 64)
SWEEP_RULES = (("confidence",) + tuple(f"spaced{g}" for g in MIN_GAPS)
               + ("random", "gt_margin"))

SCALARS = ("gt_margin", "wait_gain", "wait_gain_resid", "confidence", "entropy",
           "margin", "logp_true", "t", "step", "n_masked", "position", "text")


# --------------------------------------------------------------------- data

def sample_windows(path, n_windows, seq_len, rng):
    """Verbatim from collect_labels.py, so the windows are the SAME windows.

    Same file, same numpy Generator, same seed, same seq_len: rng.integers
    draws sequentially, so the first n starts of a 200-draw call are the first
    n of a 100-draw call. With N_TEXTS = 200 and SEED = 0 this script therefore
    runs on exactly the 200 windows Part 1 used, and every contrast with Part 1
    is paired by window rather than merely distributional.

    DLM_WINDOWS=<file.pt> overrides with a flat tensor of token ids, for a
    different corpus.
    """
    data = np.memmap(path, dtype=np.uint16, mode="r")
    starts = rng.integers(0, len(data) - seq_len, size=n_windows)
    return torch.from_numpy(
        np.stack([data[s: s + seq_len].astype(np.int64) for s in starts]))


def load_windows(n, seq_len, _generator):
    override = os.environ.get("DLM_WINDOWS")
    if override:
        ids = torch.load(override, map_location="cpu").flatten()
        n_avail = len(ids) // seq_len
        assert n_avail >= n, f"corpus holds {n_avail} windows, need {n}"
        starts = np.random.default_rng(SEED).integers(0, n_avail, size=n)
        print(f"windows from DLM_WINDOWS={override}")
        return torch.stack([ids[s * seq_len:(s + 1) * seq_len] for s in starts])

    path = DATA / VAL_BIN
    assert path.exists(), f"{path} not found -- set DLM_WINDOWS instead"
    print(f"windows from {path} (same draw as collect_labels.py at SEED={SEED})")
    return sample_windows(path, n, seq_len, np.random.default_rng(SEED))


# ----------------------------------------------------------------- measures

def masked_stats(logits_m, true_m, mask_id):
    """confidence, entropy, margin, gt_margin, logp_true at masked positions."""
    logits_m = logits_m.float().clone()
    logits_m[:, mask_id] = float("-inf")
    confidence, entropy, margin, _, _ = score_positions(logits_m)
    logp = logits_m.log_softmax(-1)
    p = logp.exp()
    top2 = p.topk(2, dim=-1)
    is_gt_top1 = top2.indices[:, 0] == true_m
    best_other = torch.where(is_gt_top1, top2.values[:, 1], top2.values[:, 0])
    p_gt = p.gather(1, true_m[:, None]).squeeze(1)
    return {
        "confidence": confidence,
        "entropy": entropy,
        "margin": margin,
        "gt_margin": p_gt - best_other,
        "logp_true": logp.gather(1, true_m[:, None]).squeeze(1),
    }


@torch.no_grad()
def wait_gain_all(forward_fn, x, true, masked, mask_id, before_gt, generator):
    """Wait gain for EVERY masked position, in two complementary lookaheads.

    Part 1's definition: reveal a random half of the *other* masked positions
    with their true tokens, and see how much the position's gt_margin improves.
    Revealing H gives wait gain for the complement, and revealing the complement
    gives it for H, so two extra forward passes cover all of them and the
    "other" requirement holds by construction in both halves.

    It has to be a function of the current canvas alone. Differencing along the
    trajectory instead would make the ranking rule circular: you cannot order by
    a gain that depends on what you are about to commit.
    """
    n = masked.numel()
    perm = masked[torch.randperm(n, generator=generator)]
    halves = (perm[: n // 2], perm[n // 2:])
    out = torch.zeros(n, dtype=torch.float32)
    slot = {int(pos): i for i, pos in enumerate(masked.tolist())}

    for reveal in halves:
        if reveal.numel() == 0 or reveal.numel() == n:
            continue
        probe = x.clone()
        probe[reveal] = true[reveal]
        still = masked[~torch.isin(masked, reveal)]
        logits, _ = forward_fn(probe.unsqueeze(0), 0.0)
        after = masked_stats(logits[still], true[still], mask_id)["gt_margin"]
        idx = torch.tensor([slot[int(p)] for p in still.tolist()])
        out[idx] = after - before_gt[idx]
    return out


def residualise(wait, gt_margin):
    """Wait gain with gt_margin partialled out, within the step.

    Wait gain is gt_margin_after MINUS gt_margin_before, so it carries
    -gt_margin as a term by construction: ranking by it is partly ranking by
    the Gt-Margin oracle, and on an untrained model it degenerates into exactly
    that. A raw wait-gain rule that beat confidence would therefore not be
    evidence about readiness -- it could be Where-to-Unmask's result arriving
    through the back door.

    This is the same move Part 2 already makes on its own target, where the
    chain term is reported with the marginal regressed out.
    """
    g = gt_margin - gt_margin.mean()
    var = float((g * g).sum())
    w = wait - wait.mean()
    if var < 1e-12:                      # a step where gt_margin is constant
        return w
    return w - (float((g * w).sum()) / var) * g


def spaced_topk(scores, positions, k, min_gap):
    """Top-k by score, subject to every pair being at least min_gap apart.

    Greedy: take the best, then the best among what is still far enough. If the
    constraint cannot be met -- late in the decode there may not be k positions
    that far apart -- it relaxes by filling from the remaining best.
    """
    order = torch.argsort(scores, descending=True).tolist()
    pos = positions.tolist()
    chosen = []
    for idx in order:
        if all(abs(pos[idx] - pos[c]) >= min_gap for c in chosen):
            chosen.append(idx)
            if len(chosen) == k:
                return torch.tensor(chosen)

    # Late in the decode there may not be k positions that far apart. The naive
    # fallback -- fill from the remaining best -- reverts to confidence order,
    # i.e. to clustering, and precisely for the large gaps that can never be
    # met. That turns "constraint too wide" into "no constraint at all" and puts
    # a spurious rising arm on the sweep.
    #
    # MAXIMIN instead: add the candidate that maximises the minimum distance to
    # what is already chosen, ties broken by score. It degrades to "as spread as
    # this canvas allows" rather than to the unconstrained rule.
    taken = set(chosen)
    while len(chosen) < k and len(taken) < len(order):
        best, best_key = None, None
        for idx in order:
            if idx in taken:
                continue
            d = min((abs(pos[idx] - pos[c]) for c in chosen), default=10**9)
            key = (d, float(scores[idx]))
            if best_key is None or key > best_key:
                best, best_key = idx, key
        chosen.append(best)
        taken.add(best)
    return torch.tensor(chosen[:k])


def clustered(positions, k, generator):
    """A random seed position and its k-1 nearest masked neighbours."""
    n = positions.numel()
    seed = int(torch.randint(n, (1,), generator=generator))
    d = (positions - positions[seed]).abs()
    return torch.argsort(d)[:k]


def select(rule, stats, wait, n_masked, k, generator, positions=None, min_gap=1):
    """Indices into the masked list, k of them, under one position rule."""
    if rule == "confidence":
        return stats["confidence"].topk(k).indices
    if rule == "confidence_spaced":
        return spaced_topk(stats["confidence"], positions, k, min_gap)
    if rule.startswith("spaced"):
        return spaced_topk(stats["confidence"], positions, k, int(rule[6:]))
    if rule == "random_clustered":
        return clustered(positions, k, generator)
    if rule == "gt_margin":
        return stats["gt_margin"].topk(k).indices
    if rule == "oracle_logp":
        # The greedy per-step maximiser of the log-p metric -- not a guaranteed
        # ceiling on the total, since an early commit changes every later canvas.
        return stats["logp_true"].topk(k).indices
    if rule == "wait_gain":
        return (-wait).topk(k).indices          # least to gain from waiting first
    if rule == "wait_gain_resid":
        return (-residualise(wait, stats["gt_margin"])).topk(k).indices
    if rule == "random":
        return torch.randperm(n_masked, generator=generator)[:k]
    if rule == "left_to_right":
        return torch.arange(min(k, n_masked))
    raise ValueError(rule)


# ------------------------------------------------------------ collect mode

@torch.no_grad()
def collect_one(forward_fn, true, mask_id, layers, k, generator, text_id):
    seq_len = true.numel()
    x = torch.full((seq_len,), mask_id, dtype=torch.long)
    rows, hidden_rows = [], []

    for step in range(seq_len // k + 1):
        masked = (x == mask_id).nonzero(as_tuple=False).squeeze(-1)
        if masked.numel() <= k:
            break
        logits, hidden = forward_fn(x.unsqueeze(0), 0.0)
        stats = masked_stats(logits[masked], true[masked], mask_id)

        if step % PROBE_EVERY == 0:
            wait = wait_gain_all(forward_fn, x, true, masked, mask_id,
                                 stats["gt_margin"], generator)
            n = masked.numel()
            take = torch.randperm(n, generator=generator)[:MAX_ROWS_PER_STEP]
            pos = masked[take]
            row = {key: stats[key][take].cpu() for key in
                   ("gt_margin", "confidence", "entropy", "margin", "logp_true")}
            row["wait_gain"] = wait[take]
            row["wait_gain_resid"] = residualise(wait, stats["gt_margin"])[take]
            row["t"] = torch.full((len(take),), n / seq_len)
            row["step"] = torch.full((len(take),), float(step))
            row["n_masked"] = torch.full((len(take),), float(n))
            row["position"] = pos.float()
            row["text"] = torch.full((len(take),), float(text_id))
            rows.append(row)
            hidden_rows.append({l: hidden[l][pos].half().cpu() for l in layers})

        chosen = stats["confidence"].topk(k).indices
        x[masked[chosen]] = true[masked[chosen]]

    return rows, hidden_rows


def run_collect(model, cfg, forward_fn, layers):
    mask_id, seq_len = int(cfg["mask_id"]), int(cfg["seq_len"])
    generator = torch.Generator("cpu").manual_seed(SEED)
    windows = load_windows(N_TEXTS, seq_len, generator)
    assert (windows == mask_id).sum() == 0, "corpus contains the mask token"

    scalars, hidden = {k: [] for k in SCALARS}, {l: [] for l in layers}
    for i in range(N_TEXTS):
        rows, hrows = collect_one(forward_fn, windows[i], mask_id, layers,
                                  K, generator, i)
        for row in rows:
            for key in SCALARS:
                scalars[key].append(row[key])
        for hrow in hrows:
            for l in layers:
                hidden[l].append(hrow[l])
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{N_TEXTS} texts, "
                  f"{sum(len(v) for v in scalars['gt_margin'])} rows", flush=True)

    record = {key: torch.cat(value) for key, value in scalars.items()}
    record["hidden"] = {l: torch.cat(v) for l, v in hidden.items()}
    record["window"] = record["text"].long()      # the grouping unit, named as in Part 1

    # The §2.1 guard, adapted. Trajectories differ here because the texts differ,
    # not because tokens are sampled -- so check it rather than assume it.
    a = record["text"] == 0
    b = record["text"] == 1
    n = min(int(a.sum()), int(b.sum()))
    assert n > 0 and not torch.allclose(record["gt_margin"][a][:n],
                                        record["gt_margin"][b][:n]), (
        "two texts produced identical trajectories -- grouped splits would "
        "separate nothing and any probe score is memorisation")

    wg, conf = record["wait_gain"], record["confidence"]
    print(f"\n{len(wg)} rows over {len(torch.unique(record['window']))} texts, k={K}")
    print(f"wait_gain : mean {wg.mean():+.4f}  sd {wg.std():.4f}  "
          f"share>0 {float((wg > 0).float().mean()):.3f}")
    print(f"gt_margin : mean {record['gt_margin'].mean():+.4f}")
    print(f"corr(confidence, wait_gain) = "
          f"{float(torch.corrcoef(torch.stack([conf, wg]))[0, 1]):+.4f}")
    print("   Part 1, i.i.d. masks, ground-truth canvas: -0.18. A large gap here")
    print("   means the mask pattern matters and Part 1 generalises less than it looks.")
    gt = record["gt_margin"]
    print(f"corr(gt_margin,  wait_gain) = "
          f"{float(torch.corrcoef(torch.stack([gt, wg]))[0, 1]):+.4f}")
    print("   Necessarily negative: wait gain is gt_margin_after MINUS")
    print("   gt_margin_before, so it carries -gt_margin as a term. The size of")
    print("   this is how much a raw wait-gain rule is really a Gt-Margin rule.")
    print(f"corr(gt_margin,  wait_gain_resid) = "
          f"{float(torch.corrcoef(torch.stack([gt, record['wait_gain_resid']]))[0, 1]):+.4f}")
    print("   Zero within each step by construction. Pooled across steps it can")
    print("   drift a little; anything past about 0.05 means gt_margin is nearly")
    print("   constant inside some steps and the partialling has no purchase there.")
    ce = -record["logp_true"]
    print(f"\n{'':>8} {'n':>7} {'CE':>7} {'acc':>7} {'conf':>7}   (collect_labels.py"
          " prints the same four)")
    edges = [(0.8, 1.01), (0.5, 0.8), (0.2, 0.5), (0.0, 0.2)]
    for lo, hi in edges:
        m = (record["t"] >= lo) & (record["t"] < hi)
        if not bool(m.any()):
            continue
        print(f"t {lo:.1f}-{hi:.1f} {int(m.sum()):7d} {float(ce[m].mean()):7.3f} "
              f"{float((record['gt_margin'][m] > 0).float().mean()):7.3f} "
              f"{float(conf[m].mean()):7.4f}")
    print(f"{'all':>8} {len(ce):7d} {float(ce.mean()):7.3f} "
          f"{float((record['gt_margin'] > 0).float().mean()):7.3f} "
          f"{float(conf.mean()):7.4f}")
    print("   Part 1 on the same 200 windows: CE 2.757, acc 0.455, conf 0.459.")
    print("   Its t buckets are i.i.d. masks at a fixed rate; these are the rates")
    print("   the sampler actually passes through, so they are comparable in")
    print("   masking level but not in mask structure -- which is the point.")

    for l in layers:
        print(f"  hidden[{l}]: {tuple(record['hidden'][l].shape)} {record['hidden'][l].dtype}")

    by_step = {}
    for s, w in zip(record["step"].tolist(), wg.tolist()):
        by_step.setdefault(int(s), []).append(w)
    early = [w for s, v in by_step.items() if s < 16 for w in v]
    late = [w for s, v in by_step.items() if s >= 16 for w in v]
    if early and late:
        print(f"\nsteps < 16 (invisible to Part 2): mean wait gain "
              f"{np.mean(early):+.4f}, n={len(early)}")
        print(f"steps >= 16                      : mean wait gain "
              f"{np.mean(late):+.4f}, n={len(late)}")

    torch.save(record, OUT)
    print(f"\nsaved to {OUT}")


# ------------------------------------------------------------- orders mode

@torch.no_grad()
def chain_sum(forward_fn, x, positions, tokens, order):
    """Sum of log p(x_j | c, x_{S<j}) for one reveal order. As in Part 2."""
    state = x.clone()
    total = 0.0
    for idx in order:
        logits, _ = forward_fn(state.unsqueeze(0), 0.0)
        total += float(logits[positions[idx]].float().log_softmax(-1)[tokens[idx]])
        state[positions[idx]] = tokens[idx]
    return total


@torch.no_grad()
def orders_one(forward_fn, true, mask_id, k, rule, generator, with_penalty=False):
    """Replay one text under one position rule.

    Returns the sum of log p over the whole decode, and -- when asked -- the
    mean one-step penalty of the sets it chose to commit together.
    """
    seq_len = true.numel()
    x = torch.full((seq_len,), mask_id, dtype=torch.long)
    total, steps, penalties, spread = 0.0, 0, [], []

    while True:
        masked = (x == mask_id).nonzero(as_tuple=False).squeeze(-1)
        if masked.numel() == 0:
            break
        logits, _ = forward_fn(x.unsqueeze(0), 0.0)
        stats = masked_stats(logits[masked], true[masked], mask_id)
        wait = None
        if rule in ("wait_gain", "wait_gain_resid"):
            wait = wait_gain_all(forward_fn, x, true, masked, mask_id,
                                 stats["gt_margin"], generator)
        take = min(k, masked.numel())
        min_gap = max(1, masked.numel() // (2 * k))
        chosen = select(rule, stats, wait, masked.numel(), take, generator,
                        positions=masked, min_gap=min_gap)
        positions = masked[chosen]
        tokens = true[positions]
        marginal = float(stats["logp_true"][chosen].sum())
        total += marginal

        if with_penalty and steps % PROBE_EVERY == 0 and take > 1:
            orders = [list(range(take)),
                      torch.randperm(take, generator=generator).tolist()]
            chain = np.mean([chain_sum(forward_fn, x, positions, tokens, o)
                             for o in orders])
            penalties.append(float(chain) - marginal)
            # How spread out is the set this rule chose? Adjacent positions are
            # strongly dependent whatever the rule, so any penalty comparison
            # has to be read against this.
            pos = np.sort(positions.numpy())
            gaps = np.abs(pos[:, None] - pos[None, :])
            off = gaps[np.triu_indices(len(pos), k=1)]
            spread.append((float(off.mean()), float(off.min()),
                           float((off == 1).sum()), float(masked.numel())))

        x[positions] = tokens
        steps += 1

    return (total, float(np.mean(penalties)) if penalties else float("nan"),
            penalties, spread)


def run_orders(model, cfg, forward_fn):
    mask_id, seq_len = int(cfg["mask_id"]), int(cfg["seq_len"])
    generator = torch.Generator("cpu").manual_seed(SEED)
    windows = load_windows(N_TEXTS_ORDERS, seq_len, generator)
    assert (windows == mask_id).sum() == 0, "corpus contains the mask token"

    with_penalty = "--penalty" in sys.argv
    rules = SWEEP_RULES if "--sweep" in sys.argv else RULES
    if "--sweep" in sys.argv:
        print(f"sweep mode: fixed minimum gaps {MIN_GAPS}\n")
    totals = {rule: [] for rule in rules}
    pens = {rule: [] for rule in rules}
    steps_rows = []
    for i in range(N_TEXTS_ORDERS):
        for rule in rules:
            g = torch.Generator("cpu").manual_seed(SEED * 1000 + i)
            total, pen, per_step, sp = orders_one(
                forward_fn, windows[i], mask_id, K, rule, g, with_penalty)
            totals[rule].append(total / seq_len)      # nats per token
            pens[rule].append(pen)
            for value, (mg, mn, adj, nm) in zip(per_step, sp):
                steps_rows.append((rule, i, value, mg, mn, adj, nm))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{N_TEXTS_ORDERS} texts", flush=True)

    base = np.array(totals["confidence"])
    print("\nSum of log p(true token | canvas) over the whole decode, nats/token.")
    print("The text is identical under every rule, so this is the ordering")
    print("decision and nothing else. Differences are paired by text.\n")
    header = f"{'rule':>14} {'nats/token':>12} {'vs confidence':>15} {'95% CI':>22} {'P(>0)':>8}"
    print(header)
    print("-" * len(header))

    rng = np.random.default_rng(12345)
    idx = rng.integers(0, len(base), size=(2000, len(base)))
    rows = []
    for rule in rules:
        v = np.array(totals[rule])
        d = v - base
        draws = d[idx].mean(axis=1)
        lo, hi = np.percentile(draws, [2.5, 97.5])
        print(f"{rule:>14} {v.mean():>12.4f} {d.mean():>+15.4f} "
              f"{f'({lo:+.4f}, {hi:+.4f})':>22} {float((draws > 0).mean()):>8.3f}")
        rows.append([rule, f"{v.mean():.4f}", f"{d.mean():.4f}",
                     f"{lo:.4f}", f"{hi:.4f}", f"{float((draws > 0).mean()):.4f}"])

    print("\nREAD THIS BEFORE READING THE TABLE. The metric above is maximised,")
    print("greedily, by `oracle_logp` itself, so it cannot arbitrate between two")
    print("oracles: any oracle correlated with log p(true) wins by construction,")
    print("and wait gain is such an oracle -- it is gt_margin_after MINUS")
    print("gt_margin_before, so it carries the current correctness as a term.")
    print("This table is a fair comparison only among rules that do NOT see the")
    print("ground truth: confidence, random, left_to_right, and any learned probe.")

    if with_penalty:
        print("\nThe fair comparison for the oracles is below: the one-step penalty")
        print("of the sets each rule chose to commit together, chain minus marginal,")
        print("averaged over two reveal orders. No rule optimises this directly.")
        print("The readiness hypothesis predicts wait-gain ordering commits sets")
        print("that are less conditionally dependent, so a penalty closer to zero.\n")
        base_p = np.array(pens["confidence"])
        head = f"{'rule':>16} {'mean penalty':>14} {'vs confidence':>15} {'95% CI':>22}"
        print(head)
        print("-" * len(head))
        for rule in rules:
            v = np.array(pens[rule])
            d = v - base_p
            draws = d[idx].mean(axis=1)
            lo, hi = np.percentile(draws, [2.5, 97.5])
            print(f"{rule:>16} {v.mean():>14.4f} {d.mean():>+15.4f} "
                  f"{f'({lo:+.4f}, {hi:+.4f})':>22}")
            rows[rules.index(rule)] += [f"{v.mean():.4f}", f"{d.mean():.4f}"]

        # --- the two tables recomposed --------------------------------------
        # The first table is the MARGINAL sum: what a parallel decoder actually
        # pays, since it scores each committed position by its own marginal and
        # ignores how they co-vary. The penalty is exactly what that ignores.
        # Their sum is the sequential log-probability of the same text under the
        # same order -- what a decoder committing one position at a time would
        # get. Read separately the two tables mislead in opposite directions: a
        # rule that commits mutually dependent sets looks terrible on the first
        # and is partly absolved by the second.
        steps_per_text = int(np.ceil(seq_len / K))
        print("\n" + "=" * 74)
        print("Marginal sum + penalty = the sequential log-probability of the")
        print("same text under the same order. Per token, so the penalty (which")
        print(f"is per committed set of {K}) is divided by {seq_len // steps_per_text}.")
        print("Approximate: the penalty is sampled every "
              f"{PROBE_EVERY} steps. Set PROBE_EVERY = 1 for the exact value.\n")
        head2 = (f"{'rule':>16} {'marginal':>10} {'penalty/tok':>12} "
                 f"{'sequential':>11} {'vs confidence':>14}")
        print(head2)
        print("-" * len(head2))
        scale = seq_len / steps_per_text
        chain_of = {r: np.mean(totals[r]) + np.mean(pens[r]) / scale for r in rules}
        for rule in sorted(rules, key=lambda r: -chain_of[r]):
            print(f"{rule:>16} {np.mean(totals[rule]):>10.4f} "
                  f"{np.mean(pens[rule]) / scale:>12.4f} {chain_of[rule]:>11.4f} "
                  f"{chain_of[rule] - chain_of['confidence']:>+14.4f}")
        for rule in rules:
            rows[rules.index(rule)].append(f"{chain_of[rule]:.4f}")
        print("\nA rule that is far apart on the two tables is paying the")
        print("lockstep tax: it commits positions that inform each other, so the")
        print("marginal sum it actually earns is much worse than the order is.")

        # --- is the penalty advantage just spatial dispersion? ---------------
        if steps_rows:
            print("\n" + "=" * 74)
            print("DISPERSION CONTROL")
            print("=" * 74)
            print("Adjacent positions are strongly dependent whatever chose them, so")
            print("a rule that scatters its commits pays a lower penalty for a reason")
            print("that has nothing to do with readiness. Below: the dispersion each")
            print("rule produces, then the penalty with dispersion regressed out.\n")

            rule_of = np.array([r[0] for r in steps_rows])
            text_of = np.array([r[1] for r in steps_rows])
            pen_of = np.array([r[2] for r in steps_rows], dtype=float)
            feat = np.array([[1.0, np.log1p(r[3]), np.log1p(r[4]), r[5], r[6] / seq_len]
                             for r in steps_rows], dtype=float)

            beta, *_ = np.linalg.lstsq(feat, pen_of, rcond=None)
            resid = pen_of - feat @ beta
            ss_tot = float(((pen_of - pen_of.mean()) ** 2).sum())
            share = 1 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")
            print(f"dispersion explains {share:.1%} of the step-level penalty, pooled "
                  f"over all rules ({len(pen_of)} steps)\n")

            # One resampling of texts, reused by every rule, so the
            # comparisons stay paired.
            texts = np.unique(text_of)
            per = {}
            for rule in rules:
                m = rule_of == rule
                per[rule] = np.array(
                    [resid[m & (text_of == t)].mean()
                     if (m & (text_of == t)).any() else np.nan for t in texts])
            pick = np.random.default_rng(12345).integers(
                0, len(texts), size=(2000, len(texts)))
            base_draws = np.nanmean(per["confidence"][pick], axis=1)

            head3 = (f"{'rule':>16} {'mean gap':>9} {'min gap':>9} {'adjacent':>9} "
                     f"{'penalty':>9} {'residual':>10} {'vs conf, 95% CI':>22}")
            print(head3)
            print("-" * len(head3))
            for rule in rules:
                m = rule_of == rule
                if not m.any():
                    continue
                d = np.nanmean(per[rule][pick], axis=1) - base_draws
                lo, hi = np.percentile(d, [2.5, 97.5])
                f = feat[m]
                print(f"{rule:>16} {np.expm1(f[:, 1]).mean():>9.1f} "
                      f"{np.expm1(f[:, 2]).mean():>9.2f} {f[:, 3].mean():>9.2f} "
                      f"{pen_of[m].mean():>9.4f} {resid[m].mean():>+10.4f} "
                      f"{f'({lo:+.4f},{hi:+.4f})':>22}")

            print("\nThe residual column is the question. A rule whose penalty")
            print("advantage survives dispersion is choosing sets for a reason beyond")
            print("'not adjacent'. One whose residual collapses to confidence's was")
            print("only ever scattering, and dispersion -- not readiness -- is the")
            print("finding. Random is the reference for pure scattering.")

            out = RESULTS / "ordering_steps.csv"
            with open(out, "w", newline="") as handle:
                w = csv.writer(handle)
                w.writerow(["rule", "text", "penalty", "mean_gap", "min_gap",
                            "n_adjacent", "n_masked"])
                w.writerows(steps_rows)
            print(f"\nwrote {out}")

    else:
        print("\nRe-run with --penalty for the comparison that is fair to the oracles.")

    ORDERS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(ORDERS_CSV, "w", newline="") as handle:
        writer = csv.writer(handle)
        header = ["rule", "nats_per_token", "delta_vs_confidence",
                  "boot_lo", "boot_hi", "p_positive"]
        if with_penalty:
            header += ["mean_penalty", "penalty_vs_confidence", "sequential_logp"]
        writer.writerow(header)
        writer.writerows(rows)
    print(f"\nwrote {ORDERS_CSV}")


# ------------------------------------------------------------------- entry

def main():
    mode = next((a for a in sys.argv[1:] if a in ("collect", "orders")), "collect")
    model, cfg = load_model()
    layers = list(BLOCK_LAYERS) + [len(model.blocks)]
    modules = {i: model.blocks[i] for i in BLOCK_LAYERS}
    modules[len(model.blocks)] = model.ln_f      # the final representation, as in Part 1

    print(f"mode: {mode} | layers: {layers} (last is ln_f)")
    with HiddenCapture(modules) as capture:
        forward_fn = capture.wrap(nano_forward_fn(model))
        if mode == "collect":
            if OUT.exists():
                print(f"{OUT} exists; delete it to recollect")
                return
            run_collect(model, cfg, forward_fn, layers)
        else:
            run_orders(model, cfg, forward_fn)


if __name__ == "__main__":
    main()
