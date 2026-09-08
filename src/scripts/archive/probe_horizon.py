from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.dlm_interp.paths import RESULTS, RUNS

LABELS = RUNS / "labels_penalty_horizon.pt"
TARGET = "penalty"
SEEDS = (0, 1, 2)
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

def fit_and_score(x, y, train, val, test):
    scaler = StandardScaler().fit(x[train])
    xt, xv = scaler.transform(x[train]), scaler.transform(x[val])
    scores = [r2(Ridge(alpha=a).fit(xt, y[train]).predict(xv), y[val]) for a in ALPHAS]
    alpha = ALPHAS[int(np.argmax(scores))]
    fit = train | val
    scaler = StandardScaler().fit(x[fit])
    pred = Ridge(alpha=alpha).fit(scaler.transform(x[fit]), y[fit]).predict(
        scaler.transform(x[test])
    )
    rho = 0.0 if np.std(pred) == 0 else float(spearmanr(pred, y[test]).statistic)
    return r2(pred, y[test]), rho

def reduce_layers(layers, train):
    out = []
    for _, h in sorted(layers.items()):
        h = h.float().numpy()
        dims = min(PCA_DIMS, h.shape[1], int(train.sum()))
        out.append(PCA(n_components=dims, random_state=0).fit(h[train]).transform(h))
    return np.concatenate(out, axis=1)

def main():
    record = torch.load(LABELS, weights_only=False)
    y = record[TARGET].float().numpy()
    horizons = record["horizons"]
    cov = torch.stack([record[k] for k in COV], dim=1).numpy()
    output = torch.stack([record[k] for k in OUTPUT], dim=1).numpy()
    base = np.concatenate([cov, output], axis=1)

    baseline, curves = [], {h: [] for h in horizons}
    for seed in SEEDS:
        train, val, test = three_way_split(record["generation"], seed)
        baseline.append(fit_and_score(base, y, train, val, test))
        for h in horizons:
            hidden = reduce_layers(record["hidden"][h], train.numpy())
            x = np.concatenate([base, hidden], axis=1)
            curves[h].append(fit_and_score(x, y, train, val, test))

    def summarise(pairs, index):
        values = [p[index] for p in pairs]
        return float(np.mean(values)), float(np.std(values))

    b_r2, b_sd = summarise(baseline, 0)
    b_rho, b_rho_sd = summarise(baseline, 1)
    print(f"{len(y)} steps, {len(torch.unique(record['generation']))} generations, "
          f"{len(SEEDS)} split seeds\n")
    print(f"{'features':>28} {'R2':>16} {'rho':>16}")
    print("-" * 62)
    print(f"{'cov + output':>28} {b_r2:9.4f} ±{b_sd:5.3f} {b_rho:9.4f} ±{b_rho_sd:5.3f}")
    for h in horizons:
        r2_mean, r2_sd = summarise(curves[h], 0)
        rho_mean, rho_sd = summarise(curves[h], 1)
        tag = "  (circular)" if h == 0 else ""
        print(f"{'+ hidden at h=' + str(h):>28} {r2_mean:9.4f} ±{r2_sd:5.3f} "
              f"{rho_mean:9.4f} ±{rho_sd:5.3f}{tag}")

    means = [summarise(curves[h], 0)[0] for h in horizons]
    errs = [summarise(curves[h], 0)[1] for h in horizons]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(horizons, means, yerr=errs, fmt="o-", color="#c1440e", capsize=3,
                label="cov + output + hidden")
    ax.axhline(b_r2, color="0.45", ls="--", label="cov + output only")
    ax.fill_between([min(horizons) - 0.2, max(horizons) + 0.2],
                    b_r2 - b_sd, b_r2 + b_sd, color="0.45", alpha=0.15)
    ax.set_xlabel("steps before the decision (h)")
    ax.set_ylabel("out-of-sample $R^2$ on the commit penalty")
    ax.set_title("How far ahead is the parallel-commitment penalty decodable?",
                 fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = RESULTS / "fig_penalty_horizon.png"
    fig.savefig(path, dpi=160)
    print(f"\nwrote {path}")
    print("h=0 is circular by construction; the decay from h=1 onward is the result.")

if __name__ == "__main__":
    main()