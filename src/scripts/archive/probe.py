from __future__ import annotations
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

LABELS = "labels_waitgain.pt"
TARGET = "wait_gain"
TEST_FRACTION = 0.25
VAL_FRACTION = 0.25        
SEED = 0
ALPHAS = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5)
PCA_DIMS = 50

def build_features(record, pca_dims=PCA_DIMS, fit_mask=None):
    conf, ent, mar = record["confidence"], record["entropy"], record["margin"]
    output = torch.stack([conf, ent, mar], dim=1)

    out = {
        "confidence": conf[:, None],
        "entropy": ent[:, None],
        "margin": mar[:, None],
        "output (all 3)": output,
    }

    hidden = {layer: h.float() for layer, h in sorted(record["hidden"].items())}
    for layer, h in hidden.items():
        out[f"hidden[{layer}]"] = h
    stacked = torch.cat(list(hidden.values()), dim=1)
    out["hidden (all)"] = stacked
    out["output + hidden"] = torch.cat([output, stacked], dim=1)

    reduced = {}
    for layer, h in hidden.items():
        pca = PCA(n_components=pca_dims, random_state=SEED)
        pca.fit(h[fit_mask].numpy())
        reduced[layer] = torch.from_numpy(pca.transform(h.numpy())).float()
        out[f"hidden[{layer}] pca{pca_dims}"] = reduced[layer]

    stacked_pca = torch.cat(list(reduced.values()), dim=1)
    out[f"hidden pca{pca_dims}"] = stacked_pca
    out[f"output + pca{pca_dims}"] = torch.cat([output, stacked_pca], dim=1)

    t_col = record["t"][:, None]
    out["t alone"] = t_col
    out["output + t"] = torch.cat([output, t_col], dim=1)
    
    return out


def three_way_split(windows, seed=SEED):
    ids = torch.unique(windows)
    perm = torch.randperm(len(ids), generator=torch.Generator().manual_seed(seed))
    n_test = int(round(len(ids) * TEST_FRACTION))
    n_val = int(round(len(ids) * VAL_FRACTION))

    def mask_for(chunk):
        chosen = set(ids[chunk].tolist())
        return torch.tensor([int(w) in chosen for w in windows])

    test = mask_for(perm[:n_test])
    val = mask_for(perm[n_test : n_test + n_val])
    train = ~(test | val)
    return train, val, test


def r2(pred, truth):
    ss_res = float(((truth - pred) ** 2).sum())
    ss_tot = float(((truth - truth.mean()) ** 2).sum())
    return 1 - ss_res / ss_tot


def fit_and_score(x, y, train, val, test, alphas=ALPHAS):
    assert np.isfinite(x).all(), "features contain NaN or inf"

    scaler = StandardScaler().fit(x[train])
    xt, xv = scaler.transform(x[train]), scaler.transform(x[val])
    scores = [r2(Ridge(alpha=a).fit(xt, y[train]).predict(xv), y[val]) for a in alphas]
    best = alphas[int(np.argmax(scores))]

    fit = train | val
    scaler = StandardScaler().fit(x[fit])
    model = Ridge(alpha=best).fit(scaler.transform(x[fit]), y[fit])
    pred = model.predict(scaler.transform(x[test]))

    return {
        "r2": r2(pred, y[test]),
        "spearman": float(spearmanr(pred, y[test]).statistic),
        "alpha": best,
    }

def main():
    record = torch.load(LABELS, weights_only=False)
    y = record[TARGET].float().numpy()
    train, val, test = three_way_split(record["window"])
    features = build_features(record, fit_mask=train)
    t_values = sorted(set(record["t"].tolist()))

    print(f"{int(train.sum())} train / {int(val.sum())} val / {int(test.sum())} test rows, "
          f"grouped on {len(torch.unique(record['window']))} windows")

    header = (f"{'features':>22} {'R2':>8} {'rho':>8} {'alpha':>8}"
              + "".join(f"{'t=' + f'{t:.1f}':>9}" for t in t_values))
    print(header)
    print("-" * len(header))

    for name, x in features.items():
        x = x.numpy()
        overall = fit_and_score(x, y, train, val, test)
        per_t = [
            fit_and_score(x, y, train & m, val & m, test & m)["r2"]
            for m in (record["t"] == t for t in t_values)
        ]
        print(
            f"{name:>22} {overall['r2']:8.4f} {overall['spearman']:8.4f} "
            f"{overall['alpha']:8.0f}" + "".join(f"{v:9.4f}" for v in per_t)
        )

if __name__ == "__main__":
    main()