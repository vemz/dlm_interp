"""Collect teacher-forced trajectories and compare commit-order rules."""

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

# --grid crosses criterion with constraint. The ceiling has to be a SPACED
# oracle: comparing spaced confidence against an unspaced oracle flatters the
# free fix, because the unspaced oracle is still paying a lockstep tax of its
# own. GRID_GAP is the plateau the sweep found.
GRID_GAP = 16
GRID_RULES = ("confidence", f"confidence@{GRID_GAP}",
              "gt_margin", f"gt_margin@{GRID_GAP}",
              "oracle_logp", f"oracle_logp@{GRID_GAP}",
              "wait_gain", f"wait_gain@{GRID_GAP}",
              "random")

SCALARS = ("gt_margin", "wait_gain", "wait_gain_resid", "confidence", "entropy",
           "margin", "logp_true", "t", "step", "n_masked", "position", "text")


# --------------------------------------------------------------------- data

def sample_windows(path, n_windows, seq_len, rng):
    """Load reproducible validation windows."""
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
        return torch.stack([ids[s * seq_len:(s + 1) * seq_len] for s in starts])

    path = DATA / VAL_BIN
    assert path.exists(), f"{path} not found -- set DLM_WINDOWS instead"
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
    """Estimate wait gain for every masked position with two lookaheads."""
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
    """Remove the within-step linear contribution of ground-truth margin."""
    g = gt_margin - gt_margin.mean()
    var = float((g * g).sum())
    w = wait - wait.mean()
    if var < 1e-12:                      # a step where gt_margin is constant
        return w
    return w - (float((g * w).sum()) / var) * g


def spaced_topk(scores, positions, k, min_gap):
    """Select high-scoring positions with a minimum spatial gap."""
    order = torch.argsort(scores, descending=True).tolist()
    pos = positions.tolist()
    chosen = []
    for idx in order:
        if all(abs(pos[idx] - pos[c]) >= min_gap for c in chosen):
            chosen.append(idx)
            if len(chosen) == k:
                return torch.tensor(chosen)

    # Use maximin when the requested gap cannot be satisfied.
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


def score_for(rule, stats, wait, n_masked, generator):
    """Return the score used by a position rule."""
    if rule == "confidence":
        return stats["confidence"]
    if rule == "gt_margin":
        return stats["gt_margin"]
    if rule == "oracle_logp":
        return stats["logp_true"]
    if rule == "wait_gain":
        return -wait                       # least to gain from waiting first
    if rule == "wait_gain_resid":
        return -residualise(wait, stats["gt_margin"])
    if rule == "random":
        return torch.rand(n_masked, generator=generator)
    if rule == "left_to_right":
        return -torch.arange(n_masked, dtype=torch.float32)
    raise ValueError(rule)


NEEDS_WAIT = ("wait_gain", "wait_gain_resid")


def select(rule, stats, wait, n_masked, k, generator, positions=None, min_gap=1):
    """Select k masked positions under a scoring and gap rule."""
    if rule == "confidence_spaced":                   # the adaptive variant
        return spaced_topk(stats["confidence"], positions, k, min_gap)
    if rule == "random_clustered":
        return clustered(positions, k, generator)
    if rule.startswith("spaced"):                     # legacy: confidence@gap
        return spaced_topk(stats["confidence"], positions, k, int(rule[6:]))

    base, _, gap = rule.partition("@")
    scores = score_for(base, stats, wait, n_masked, generator)
    if not gap:
        return scores.topk(k).indices
    return spaced_topk(scores, positions, k, int(gap))


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
    print(f"rows={len(wg)}, texts={len(torch.unique(record['window']))}, k={K}")
    print(f"wait_gain={wg.mean():+.4f}, corr_conf={float(torch.corrcoef(torch.stack([conf, wg]))[0, 1]):+.4f}")
    gt = record["gt_margin"]
    print(f"corr_gt_wait={float(torch.corrcoef(torch.stack([gt, wg]))[0, 1]):+.4f}, "
          f"corr_gt_resid={float(torch.corrcoef(torch.stack([gt, record['wait_gain_resid']]))[0, 1]):+.4f}")
    ce = -record["logp_true"]
    print("masking summary")
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
    

    by_step = {}
    for s, w in zip(record["step"].tolist(), wg.tolist()):
        by_step.setdefault(int(s), []).append(w)
    early = [w for s, v in by_step.items() if s < 16 for w in v]
    late = [w for s, v in by_step.items() if s >= 16 for w in v]
    if early and late:
                print(f"early_wait={np.mean(early):+.4f}, late_wait={np.mean(late):+.4f}")

    torch.save(record, OUT)
    print(f"saved {OUT}")


# ------------------------------------------------------------- orders mode

@torch.no_grad()
def chain_sum(forward_fn, x, positions, tokens, order):
    """Score one sequential reveal order."""
    state = x.clone()
    total = 0.0
    for idx in order:
        logits, _ = forward_fn(state.unsqueeze(0), 0.0)
        total += float(logits[positions[idx]].float().log_softmax(-1)[tokens[idx]])
        state[positions[idx]] = tokens[idx]
    return total


@torch.no_grad()
def orders_one(forward_fn, true, mask_id, k, rule, generator, with_penalty=False):
    """Replay one text under one position rule."""
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
        if rule.partition("@")[0] in NEEDS_WAIT:
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
            # Record bundle geometry with the penalty.
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
    if "--grid" in sys.argv:
        rules = GRID_RULES
    elif "--sweep" in sys.argv:
        rules = SWEEP_RULES
    else:
        rules = RULES
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
        
    base = np.array(totals["confidence"])
    print("ordering comparison")
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

    if with_penalty:
        print("penalty comparison")
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

        # Recompose the marginal score and sequential penalty.
        steps_per_text = int(np.ceil(seq_len / K))
        print("recomposed sequential score")
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
        # Control the penalty comparison for bundle geometry.
        if steps_rows:
            print("dispersion control")

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

            out = RESULTS / "ordering_steps.csv"
            with open(out, "w", newline="") as handle:
                w = csv.writer(handle)
                w.writerow(["rule", "text", "penalty", "mean_gap", "min_gap",
                            "n_adjacent", "n_masked"])
                w.writerows(steps_rows)
            print(f"saved {out}")

    else:
        print("use --penalty for set-dependence metrics")

    ORDERS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(ORDERS_CSV, "w", newline="") as handle:
        writer = csv.writer(handle)
        header = ["rule", "nats_per_token", "delta_vs_confidence",
                  "boot_lo", "boot_hi", "p_positive"]
        if with_penalty:
            header += ["mean_penalty", "penalty_vs_confidence", "sequential_logp"]
        writer.writerow(header)
        writer.writerows(rows)
    print(f"saved {ORDERS_CSV}")


# ------------------------------------------------------------------- entry

def main():
    mode = next((a for a in sys.argv[1:] if a in ("collect", "orders")), "collect")
    model, cfg = load_model()
    layers = list(BLOCK_LAYERS) + [len(model.blocks)]
    modules = {i: model.blocks[i] for i in BLOCK_LAYERS}
    modules[len(model.blocks)] = model.ln_f      # the final representation, as in Part 1

    # Run either teacher-forced collection or rule comparison.
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
