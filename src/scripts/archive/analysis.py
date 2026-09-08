"""Headline analysis: does the residual stream carry readiness that output
confidence does not?

Runs both targets (predictability vs readiness) over several split seeds,
writes a CSV and two figures.
"""

from __future__ import annotations

import csv
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

TARGETS = {
    "predictability\n(Gt-Margin)": ("labels_gtmargin.pt", "gt_margin"),
    "readiness\n(wait gain)": ("labels_waitgain.pt", "wait_gain"),
}
FAMILIES = ("t alone", "output", "output + t", "hidden pca50", "output + t + hidden pca50")
SEEDS = (0, 1, 2, 3, 4)
ALPHAS = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5)
PCA_DIMS = 50
TEST_FRACTION = VAL_FRACTION = 0.25


# ---------------------------------------------------------------- plumbing

def three_way_split(windows, seed):
    ids = torch.unique(windows)
    perm = torch.randperm(len(ids), generator=torch.Generator().manual_seed(seed))
    n_test = int(round(len(ids) * TEST_FRACTION))
    n_val = int(round(len(ids) * VAL_FRACTION))

    def mask_for(chunk):
        chosen = set(ids[chunk].tolist())
        return torch.tensor([int(w) in chosen for w in windows])

    test = mask_for(perm[:n_test])
    val = mask_for(perm[n_test : n_test + n_val])
    return ~(test | val), val, test


def r2(pred, truth):
    ss_res = float(((truth - pred) ** 2).sum())
    ss_tot = float(((truth - truth.mean()) ** 2).sum())
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def rho(pred, truth):
    if np.std(pred) == 0:
        return 0.0
    return float(spearmanr(pred, truth).statistic)


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
    return r2(pred, y[test]), rho(pred, y[test])


def build_families(record, train):
    """PCA is fit on this seed's train rows only."""
    output = torch.stack(
        [record["confidence"], record["entropy"], record["margin"]], dim=1
    ).numpy()
    t_col = record["t"][:, None].numpy()

    reduced = []
    for _, h in sorted(record["hidden"].items()):
        h = h.float().numpy()
        dims = min(PCA_DIMS, h.shape[1], int(train.sum()))
        pca = PCA(n_components=dims, random_state=0).fit(h[train])
        reduced.append(pca.transform(h))
    hidden = np.concatenate(reduced, axis=1)

    return {
        "t alone": t_col,
        "output": output,
        "output + t": np.concatenate([output, t_col], axis=1),
        "hidden pca50": hidden,
        "output + t + hidden pca50": np.concatenate([output, t_col, hidden], axis=1),
    }


# ---------------------------------------------------------------- runs

def run_target(path, target):
    record = torch.load(path, weights_only=False)
    y = record[target].float().numpy()
    t_values = sorted(set(record["t"].tolist()))

    overall = {name: {"r2": [], "rho": []} for name in FAMILIES}
    per_t = {name: {t: [] for t in t_values} for name in FAMILIES}

    for seed in SEEDS:
        train, val, test = three_way_split(record["window"], seed)
        families = build_families(record, train.numpy())
        for name, x in families.items():
            score, correlation = fit_and_score(x, y, train, val, test)
            overall[name]["r2"].append(score)
            overall[name]["rho"].append(correlation)
            for t in t_values:
                m = record["t"] == t
                per_t[name][t].append(fit_and_score(x, y, train & m, val & m, test & m)[0])
        print(f"  seed {seed} done")

    return overall, per_t, t_values, len(y)


def summarise(values):
    return statistics.fmean(values), (
        statistics.stdev(values) if len(values) > 1 else 0.0
    )


# ---------------------------------------------------------------- figures

def figure_contrast(results, path="fig_contrast.png"):
    fig, axes = plt.subplots(1, len(results), figsize=(11, 4.2), sharey=False)
    for ax, (label, (overall, _, _, n)) in zip(np.atleast_1d(axes), results.items()):
        means = [summarise(overall[f]["r2"])[0] for f in FAMILIES]
        errs = [summarise(overall[f]["r2"])[1] for f in FAMILIES]
        colours = ["0.75", "0.55", "0.4", "#c1440e", "#7d2e00"]
        ax.bar(range(len(FAMILIES)), means, yerr=errs, capsize=3, color=colours)
        ax.set_xticks(range(len(FAMILIES)))
        ax.set_xticklabels(FAMILIES, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{label}\nn={n}", fontsize=10)
        ax.set_ylabel("out-of-sample $R^2$")
        ax.axhline(0, color="k", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Output confidence explains predictability; the residual stream adds readiness",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    print(f"wrote {path}")


def figure_per_t(results, path="fig_per_t.png"):
    label = "readiness\n(wait gain)"
    if label not in results:
        return
    _, per_t, t_values, _ = results[label]
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, style in (
        ("output + t", ("o-", "0.45")),
        ("hidden pca50", ("s-", "#c1440e")),
        ("output + t + hidden pca50", ("^-", "#7d2e00")),
    ):
        means = [summarise(per_t[name][t])[0] for t in t_values]
        errs = [summarise(per_t[name][t])[1] for t in t_values]
        ax.errorbar(t_values, means, yerr=errs, fmt=style[0], color=style[1],
                    capsize=3, label=name)
    ax.set_xlabel("masking level $t$")
    ax.set_ylabel("out-of-sample $R^2$ on wait gain")
    ax.set_title("Internal states add readiness information at every noise level", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    print(f"wrote {path}")


def write_csv(results, path="results.csv"):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["target", "features", "metric", "t", "mean", "std", "n_seeds"])
        for label, (overall, per_t, t_values, _) in results.items():
            clean = label.replace("\n", " ")
            for name in FAMILIES:
                for metric in ("r2", "rho"):
                    mean, std = summarise(overall[name][metric])
                    writer.writerow([clean, name, metric, "all", f"{mean:.4f}",
                                     f"{std:.4f}", len(SEEDS)])
                for t in t_values:
                    mean, std = summarise(per_t[name][t])
                    writer.writerow([clean, name, "r2", t, f"{mean:.4f}",
                                     f"{std:.4f}", len(SEEDS)])
    print(f"wrote {path}")


def main():
    results = {}
    for label, (path, target) in TARGETS.items():
        print(f"\n=== {label.replace(chr(10), ' ')} ({path}) ===")
        results[label] = run_target(path, target)

    print()
    for label, (overall, _, _, _) in results.items():
        print(f"{label.replace(chr(10), ' ')}")
        for name in FAMILIES:
            m_r2, s_r2 = summarise(overall[name]["r2"])
            m_rho, s_rho = summarise(overall[name]["rho"])
            print(f"  {name:>28}  R2 {m_r2:7.4f} ± {s_r2:.4f}   rho {m_rho:7.4f} ± {s_rho:.4f}")
        print()

    write_csv(results)
    figure_contrast(results)
    figure_per_t(results)


if __name__ == "__main__":
    main()