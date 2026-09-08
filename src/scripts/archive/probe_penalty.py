from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from src.dlm_interp.paths import RUNS 

LABELS = RUNS / "labels_penalty.pt"
TARGET = "penalty"
GROUP = "generation"
TEST_FRACTION = VAL_FRACTION = 0.25
SEED = 0
ALPHAS = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5)
PCA_DIMS = 20          
COV = ("step", "n_masked")
OUTPUT = ("conf_mean", "conf_min", "conf_max", "conf_spread", "entropy_mean", "margin_mean")

def build_features(record, train):
    cov = torch.stack([record[k] for k in COV], dim=1).numpy()
    output = torch.stack([record[k] for k in OUTPUT], dim=1).numpy()

    reduced = []
    for _, h in sorted(record["hidden"].items()):
        h = h.float().numpy()
        dims = min(PCA_DIMS, h.shape[1], int(train.sum()))
        reduced.append(PCA(n_components=dims, random_state=SEED).fit(h[train]).transform(h))
    hidden = np.concatenate(reduced, axis=1)

    base = np.concatenate([cov, output], axis=1)
    return {
        "cov alone": cov,
        "output alone": output,
        "cov + output": base,
        f"hidden pca{PCA_DIMS}": hidden,
        f"cov + hidden pca{PCA_DIMS}": np.concatenate([cov, hidden], axis=1),
        f"cov + output + hidden pca{PCA_DIMS}": np.concatenate([base, hidden], axis=1),
    }

def three_way_split(groups, seed=SEED):
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
    assert np.isfinite(x).all(), "features contain NaN or inf"
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
    return r2(pred, y[test]), rho, alpha

def step_buckets(step):
    cuts = torch.quantile(step, torch.tensor([1 / 3, 2 / 3]))
    return {
        "early": step <= cuts[0],
        "middle": (step > cuts[0]) & (step <= cuts[1]),
        "late": step > cuts[1],
    }

def main():
    record = torch.load(LABELS, weights_only=False)
    y = record[TARGET].float().numpy()
    train, val, test = three_way_split(record[GROUP])
    features = build_features(record, train.numpy())
    buckets = step_buckets(record["step"])

    print(f"{int(train.sum())} train / {int(val.sum())} val / {int(test.sum())} test steps, "
          f"grouped on {len(torch.unique(record[GROUP]))} generations")
    print(f"target: {TARGET}, sd {y.std():.3f}\n")

    header = (f"{'features':>30} {'R2':>8} {'rho':>8} {'alpha':>8}"
              + "".join(f"{name:>9}" for name in buckets))
    print(header)
    print("-" * len(header))

    for name, x in features.items():
        score, rho, alpha = fit_and_score(x, y, train, val, test)
        per_bucket = [
            fit_and_score(x, y, train & m, val & m, test & m)[0] for m in buckets.values()
        ]
        print(f"{name:>30} {score:8.4f} {rho:8.4f} {alpha:8.0f}"
              + "".join(f"{v:9.4f}" for v in per_bucket))

    print("\nThe line that matters: 'cov + output' vs 'cov + output + hidden'.")
    print("Spearman is the comparable statistic — the output-side study reports rank")
    print("correlation deltas on this same target.")

if __name__ == "__main__":
    main()