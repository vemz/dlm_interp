from __future__ import annotations
import csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from src.dlm_interp.paths import RESULTS, RUNS, TAG

LABELS = RUNS / "labels_bundle_states.pt"
SEEDS = (0, 1, 2, 3, 4)
ALPHAS = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5)
TEST_FRACTION = VAL_FRACTION = 0.25
GATE_RATES = (0.05, 0.10, 0.20, 0.30)
N_BOOT = 2000
BOOT_SEED = 12345
FREE = ("mean_dist", "min_gap", "conf_mean", "conf_min", "conf_max",
        "conf_spread", "entropy_mean", "margin_mean", "n_masked")

def three_way_split(group, seed):
    ids = np.unique(group)
    perm = np.random.default_rng(seed).permutation(len(ids))
    n_test = int(round(len(ids) * TEST_FRACTION))
    n_val = int(round(len(ids) * VAL_FRACTION))
    test_ids = set(ids[perm[:n_test]].tolist())
    val_ids = set(ids[perm[n_test:n_test + n_val]].tolist())
    test = np.array([g in test_ids for g in group])
    val = np.array([g in val_ids for g in group])
    return ~(test | val), val, test

def r2(pred, truth):
    ss_res = float(((truth - pred) ** 2).sum())
    ss_tot = float(((truth - truth.mean()) ** 2).sum())
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

def fit_ridge(x, y, train, val):
    scaler = StandardScaler().fit(x[train])
    xt, xv = scaler.transform(x[train]), scaler.transform(x[val])
    scores = [r2(Ridge(alpha=a).fit(xt, y[train]).predict(xv), y[val])
              for a in ALPHAS]
    alpha = ALPHAS[int(np.argmax(scores))]
    keep = train | val
    scaler = StandardScaler().fit(x[keep])
    model = Ridge(alpha=alpha).fit(scaler.transform(x[keep]), y[keep])
    return lambda z: model.predict(scaler.transform(z))

def fit_gbm(x, y, train, val):
    keep = train | val
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06,
                                      early_stopping=True,
                                      validation_fraction=0.2,
                                      random_state=0).fit(x[keep], y[keep])
    return m.predict

def capture(score, cost, rate):
    k = max(1, int(len(score) * rate))
    return (cost[np.argsort(-score)[:k]].sum() / max(cost.sum(), 1e-12)) / rate

def clustered(score, cost, gen, rate, rng):
    gens = np.unique(gen)
    by_gen = [np.where(gen == g)[0] for g in gens]
    draws = []
    for _ in range(N_BOOT):
        idx = np.concatenate([by_gen[i]
                              for i in rng.integers(0, len(gens), len(gens))])
        draws.append(capture(score[idx], cost[idx], rate))
    return float(np.mean(draws)), *np.percentile(draws, [2.5, 97.5])

def encode(states, kind):
    if kind == "mean":
        return states.mean(1)
    if kind == "moments":
        return np.concatenate([states.mean(1), states.std(1),
                               states.min(1), states.max(1)], axis=1)
    if kind == "concat":
        return states.reshape(states.shape[0], -1)
    raise ValueError(kind)

def main():
    if not LABELS.exists():
        raise SystemExit(f"{LABELS} not found — run collect_bundle_states.py first")
    rec = torch.load(LABELS, map_location="cpu", weights_only=False)

    layers = sorted(rec["states"])
    states = np.concatenate([rec["states"][l].float().numpy() for l in layers],
                            axis=2)                      # (n, k, layers*d)
    cost = np.maximum(-rec["penalty"].numpy(), 0)
    gen = rec["generation"].numpy().astype(int)
    free = np.column_stack([rec[k].numpy() for k in FREE])
    dist = rec["mean_dist"].numpy()

    n, k, d = states.shape
    # Compare compact encodings of the committed bundle.

    encoders = {"mean": None, "moments": None, "concat": None}
    for name in encoders:
        encoders[name] = encode(states, name)

    variants = [("mean_dist (free, unfitted)", None, None)]
    for name, enc in encoders.items():
        variants.append((f"{name} + free, ridge",
                         np.hstack([enc, free]), fit_ridge))
    variants.append(("free only, ridge", free, fit_ridge))

    scores = {label: [] for label, _, _ in variants}
    costs, gens_test, r2s = [], [], {label: [] for label, _, _ in variants}

    print()
    for seed in SEEDS:
        train, val, test = three_way_split(gen, seed)
        costs.append(cost[test])
        gens_test.append(gen[test])
        for label, x, fitter in variants:
            if fitter is None:
                s = -dist[test]
            else:
                s = fit_ridge(x, cost, train, val)(x[test])
                r2s[label].append(r2(s, cost[test]))
            scores[label].append(s)

    rng = np.random.default_rng(BOOT_SEED)
    print("encoder comparison")
    head = f"  {'variant':>28} {'R2':>7}" + "".join(
        f"{f'@{int(r * 100)}%':>20}" for r in GATE_RATES)
    print(head)
    print("  " + "-" * (len(head) - 2))

    table, out_rows = {}, []
    for label, _, _ in variants:
        line = f"  {label:>28}"
        line += f"{np.mean(r2s[label]):>7.3f}" if r2s[label] else f"{'—':>7}"
        table[label] = {}
        for rate in GATE_RATES:
            pts = [capture(scores[label][i], costs[i], rate)
                   for i in range(len(SEEDS))]
            m, lo, hi = clustered(np.concatenate(scores[label]),
                                  np.concatenate(costs),
                                  np.concatenate(gens_test), rate, rng)
            table[label][rate] = (float(np.mean(pts)), lo, hi)
            line += f"{f'{np.mean(pts):.2f}x ({lo:.2f},{hi:.2f})':>20}"
            out_rows.append([TAG or "_s0", label, rate, f"{np.mean(pts):.4f}",
                             f"{lo:.4f}", f"{hi:.4f}"])
        print(line)

    orc = float(np.mean([capture(costs[i], costs[i], 0.20)
                         for i in range(len(SEEDS))]))
    print(f"\n  oracle @20%: {orc:.2f}x")

    print("\nencoder summary")
    base_free = table["free only, ridge"][0.20][0]
    base_dist = table["mean_dist (free, unfitted)"][0.20][0]
    print(f"  free scalars alone: {base_free:.2f}x;  "
          f"mean_dist unfitted: {base_dist:.2f}x;  oracle: {orc:.2f}x\n")
    print(f"  {'encoder':>12} {'@20%':>8} {'over free':>11} {'over mean':>11}")
    print("  " + "-" * 45)
    m_mean = table["mean + free, ridge"][0.20][0]
    for name in encoders:
        v = table[f"{name} + free, ridge"][0.20][0]
        print(f"  {name:>12} {v:>8.2f}x {v - base_free:>+10.2f}x "
              f"{v - m_mean:>+10.2f}x")
        out_rows.append([TAG or "_s0", f"{name}_over_mean", 0.20,
                         f"{v - m_mean:.4f}", "", ""])

    best_name = max(encoders, key=lambda nm: table[f"{nm} + free, ridge"][0.20][0])
    best = table[f"{best_name} + free, ridge"][0.20][0]
    gain = best - m_mean

    print(f"best={best_name}, gain_over_mean={gain:+.2f}x, oracle={orc:.2f}x")

    # best achievable line, separate from the controlled comparison
    print("gbm check")
    x = np.hstack([encoders[best_name], free])
    pts = []
    for seed in SEEDS:
        train, val, test = three_way_split(gen, seed)
        s = fit_gbm(x, cost, train, val)(x[test])
        pts.append(capture(s, cost[test], 0.20))
    print(f"    {best_name} + free, GBM @20%: {np.mean(pts):.2f}x "
          f"(ridge {best:.2f}x, oracle {orc:.2f}x)")
    out_rows.append([TAG or "_s0", f"{best_name}+free_gbm", 0.20,
                     f"{np.mean(pts):.4f}", "", ""])
    # Save the compact comparison table.
    path = RESULTS / "bundle_encoder_probe.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["model", "variant", "gate_rate", "capture", "lo", "hi"])
        w.writerows(out_rows)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
