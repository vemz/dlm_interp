"""RQ1: how far ahead, and at what depth, is the commit decision decodable?

Target is the marginal-free chain term — the sequential log-likelihood of the
set about to be committed, with the trivial marginal component regressed out.
That is the part which carries dependence and which costs k extra forwards to
measure directly.

Two axes at once: how many steps before the decision the residual stream is
read, and which layer it is read from. Plus a mismatched-state control at every
horizon, because a flat curve is exactly what a leak looks like.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.dlm_interp.paths import RESULTS, RUNS

LABELS = RUNS / "labels_penalty_horizon.pt"
SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)
ALPHAS = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5)
PCA_DIMS = 20
TEST_FRACTION = VAL_FRACTION = 0.25
COV = ("step", "n_masked")
OUTPUT = ("conf_mean", "conf_min", "conf_max", "conf_spread", "entropy_mean", "margin_mean")


def three_way_split(groups, seed):
    ids = torch.unique(groups)
    perm = torch.randperm(len(ids), generator=torch.Generator().manual_seed(seed))
    n_test = int(round(len(ids) * TEST_FRACTION))
    n_val = int(round(len(ids) * VAL_FRACTION))

    def mask_for(chunk):
        chosen = set(ids[chunk].tolist())
        return torch.tensor([float(g) in chosen for g in groups])

    test = mask_for(perm[:n_test])
    val = mask_for(perm[n_test : n_test + n_val])
    return ~(test | val), val, test


def r2(pred, truth):
    ss_res = float(((truth - pred) ** 2).sum())
    ss_tot = float(((truth - truth.mean()) ** 2).sum())
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def fit_predict(x, y, train, val):
    scaler = StandardScaler().fit(x[train])
    xt, xv = scaler.transform(x[train]), scaler.transform(x[val])
    scores = [r2(Ridge(alpha=a).fit(xt, y[train]).predict(xv), y[val]) for a in ALPHAS]
    alpha = ALPHAS[int(np.argmax(scores))]
    fit = train | val
    scaler = StandardScaler().fit(x[fit])
    return Ridge(alpha=alpha).fit(scaler.transform(x[fit]), y[fit]).predict(scaler.transform(x))


def score(x, y, train, val, test):
    return r2(fit_predict(x, y, train, val)[test], y[test])


def reduce_one(h, train, dims=PCA_DIMS):
    h = h.float().numpy()
    n = min(dims, h.shape[1], int(train.sum()))
    return PCA(n_components=n, random_state=0).fit(h[train]).transform(h)


def summarise(values):
    return float(np.mean(values)), float(np.std(values))


def main():
    record = torch.load(LABELS, weights_only=False)
    groups = record["generation"]
    horizons = record["horizons"]
    layers = sorted(record["hidden"][horizons[0]])

    cov = torch.stack([record[k] for k in COV], dim=1).numpy()
    out_feat = torch.stack([record[k] for k in OUTPUT], dim=1).numpy()
    base = np.concatenate([cov, out_feat], axis=1)
    marginal = record["marginal"].numpy()[:, None]
    chain = record["chain"].numpy()

    names = [f"layer {l}" for l in layers] + ["all layers", "mismatched"]
    results = {h: {name: [] for name in names} for h in horizons}
    baseline = []

    rng = np.random.default_rng(0)
    for seed in SEEDS:
        train, val, test = three_way_split(groups, seed)
        tr = train.numpy()
        y = chain - fit_predict(marginal, chain, train, val)
        baseline.append(score(base, y, train, val, test))

        for h in horizons:
            reduced = {l: reduce_one(record["hidden"][h][l], tr) for l in layers}
            for l in layers:
                results[h][f"layer {l}"].append(
                    score(np.concatenate([base, reduced[l]], axis=1), y, train, val, test))
            stacked = np.concatenate(list(reduced.values()), axis=1)
            results[h]["all layers"].append(
                score(np.concatenate([base, stacked], axis=1), y, train, val, test))

            idx = np.arange(len(y))
            for gen in torch.unique(groups):
                m = (groups == gen).numpy()
                idx[m] = rng.permutation(idx[m])
            results[h]["mismatched"].append(
                score(np.concatenate([base, stacked[idx]], axis=1), y, train, val, test))
        print(f"  seed {seed} done")

    b_mean, b_sd = summarise(baseline)
    print(f"\n{len(chain)} steps, {len(torch.unique(groups))} generations, "
          f"{len(SEEDS)} split seeds")
    print(f"target: marginal-free chain, sd {chain.std():.3f}")
    print(f"baseline (cov + output): R2 {b_mean:.4f} ± {b_sd:.4f}\n")

    print("delta over the cov+output baseline, paired across split seeds:")
    print(f"{'h':>4} {'all layers':>18} {'best layer':>18}")
    for h in horizons:
        paired = [a - b for a, b in zip(results[h]["all layers"], baseline)]
        best = max((f"layer {l}" for l in layers),
                   key=lambda n: np.mean(results[h][n]))
        paired_best = [a - b for a, b in zip(results[h][best], baseline)]
        m, s = summarise(paired)
        mb, sb = summarise(paired_best)
        star = "*" if m - 2 * s / np.sqrt(len(SEEDS)) > 0 else " "
        print(f"{h:>4} {m:>+11.4f}±{s:.3f}{star} {mb:>+11.4f}±{sb:.3f}  ({best})")
    print("  * = mean minus two standard errors still above zero\n")

    header = f"{'h':>4}" + "".join(f"{name:>14}" for name in names)
    print(header)
    print("-" * len(header))
    for h in horizons:
        row = f"{h:>4}"
        for name in names:
            mean, sd = summarise(results[h][name])
            row += f"{mean:>9.4f}±{sd:.3f}"
        print(row)

    with open(RESULTS / "rq1_horizon_depth.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["horizon", "features", "r2_mean", "r2_sd", "n_seeds"])
        writer.writerow(["-", "cov + output", f"{b_mean:.4f}", f"{b_sd:.4f}", len(SEEDS)])
        for h in horizons:
            for name in names:
                mean, sd = summarise(results[h][name])
                writer.writerow([h, name, f"{mean:.4f}", f"{sd:.4f}", len(SEEDS)])

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    palette = ["#f0a07c", "#d4693a", "#8f3b12"]
    for colour, l in zip(palette, layers):
        means = [summarise(results[h][f"layer {l}"])[0] for h in horizons]
        errs = [summarise(results[h][f"layer {l}"])[1] for h in horizons]
        ax.errorbar(horizons, means, yerr=errs, fmt="o-", color=colour,
                    capsize=3, label=f"layer {l}", markersize=4)
    means = [summarise(results[h]["all layers"])[0] for h in horizons]
    ax.plot(horizons, means, "^-", color="#3b1a08", label="all layers")
    mismatched = [summarise(results[h]["mismatched"])[0] for h in horizons]
    ax.plot(horizons, mismatched, "x--", color="0.6", label="mismatched states")
    ax.axhline(b_mean, color="0.35", ls=":", label="cov + output only")

    ax.set_xlabel("steps before the decision")
    ax.set_ylabel("out-of-sample $R^2$")
    ax.set_title("Commit dependence is decodable from the residual stream one to\n"
                 "two steps ahead, and not from the decoder's own signals", fontsize=10)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_rq1_horizon_depth.png", dpi=160)
    print(f"\nwrote {RESULTS / 'rq1_horizon_depth.csv'}")
    print(f"wrote {RESULTS / 'fig_rq1_horizon_depth.png'}")


if __name__ == "__main__":
    main()