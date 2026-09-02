from __future__ import annotations
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

LABELS = "labels_gtmargin.pt"
TEST_FRACTION = 0.25
SEED = 0
ALPHA = 1.0

def build_features(record):
    out = {
        "confidence": record["confidence"][:, None],
        "entropy": record["entropy"][:, None],
        "margin": record["margin"][:, None],
        "output (all 3)": torch.stack(
            [record["confidence"], record["entropy"], record["margin"]], dim=1
        ),
    }
    hidden = record["hidden"]
    for layer, h in sorted(hidden.items()):
        out[f"hidden[{layer}]"] = h.float()
    out["hidden (all)"] = torch.cat([hidden[l].float() for l in sorted(hidden)], dim=1)
    out["output + hidden"] = torch.cat([out["output (all 3)"], out["hidden (all)"]], dim=1)
    return out

def split_by_window(windows, test_fraction=TEST_FRACTION, seed=SEED):
    ids = torch.unique(windows)
    perm = torch.randperm(len(ids), generator=torch.Generator().manual_seed(seed))
    n_test = int(round(len(ids) * test_fraction))
    test_ids = set(ids[perm[:n_test]].tolist())
    is_test = torch.tensor([int(w) in test_ids for w in windows])
    return ~is_test, is_test


def fit_and_score(x, y, train, test, alpha=ALPHA):
    scaler = StandardScaler().fit(x[train])
    model = Ridge(alpha=alpha).fit(scaler.transform(x[train]), y[train])
    pred = model.predict(scaler.transform(x[test]))

    truth = y[test]
    ss_res = float(((truth - pred) ** 2).sum())
    ss_tot = float(((truth - truth.mean()) ** 2).sum())
    return {
        "r2": 1 - ss_res / ss_tot,
        "spearman": float(spearmanr(pred, truth).statistic),
        "n_test": len(truth),
    }


def main():
    record = torch.load(LABELS, weights_only=False)
    y = record["gt_margin"].float().numpy()
    features = build_features(record)
    train, test = split_by_window(record["window"])
    t_values = sorted(set(record["t"].tolist()))

    print(f"{int(train.sum())} train rows, {int(test.sum())} test rows, "
          f"split on {len(torch.unique(record['window']))} windows\n")

    header = f"{'features':>18} {'R2':>8} {'rho':>8}" + "".join(
        f"{'R2 t=' + f'{t:.1f}':>10}" for t in t_values
    )
    print(header)
    print("-" * len(header))

    for name, x in features.items():
        x = x.numpy()
        overall = fit_and_score(x, y, train, test)
        per_t = []
        for t in t_values:
            mask = record["t"] == t
            per_t.append(fit_and_score(x, y, train & mask, test & mask)["r2"])

        print(
            f"{name:>18} {overall['r2']:8.4f} {overall['spearman']:8.4f}"
            + "".join(f"{v:10.4f}" for v in per_t)
        )

if __name__ == "__main__":
    main()