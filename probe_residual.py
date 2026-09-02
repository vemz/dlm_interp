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
    return ~(test | val), val, test


def r2(pred, truth):
    ss_res = float(((truth - pred) ** 2).sum())
    ss_tot = float(((truth - truth.mean()) ** 2).sum())
    return 1 - ss_res / ss_tot


def ridge_predict(x, y, fit, target, alphas=ALPHAS, val=None):
    scaler = StandardScaler().fit(x[fit])
    xf = scaler.transform(x[fit])
    if val is not None:
        xv = scaler.transform(x[val])
        scores = [r2(Ridge(alpha=a).fit(xf, y[fit]).predict(xv), y[val]) for a in alphas]
        alpha = alphas[int(np.argmax(scores))]
    else:
        alpha = alphas[0]
    model = Ridge(alpha=alpha).fit(xf, y[fit])
    return model.predict(scaler.transform(x)), alpha


def main():
    record = torch.load(LABELS, weights_only=False)
    gt = record[TARGET].float().numpy()
    train, val, test = three_way_split(record["window"])
    t_values = sorted(set(record["t"].tolist()))

    output = torch.stack(
        [record["confidence"], record["entropy"], record["margin"], record["t"]], dim=1
    ).numpy()

    pred_out, alpha_out = ridge_predict(output, gt, train, gt, val=val)
    residual = gt - pred_out

    print(f"{int(train.sum())} train / {int(val.sum())} val / {int(test.sum())} test rows")
    print(f"stage 1: output model on gt_margin, alpha={alpha_out:.0f}, "
          f"test R2 = {r2(pred_out[test], gt[test]):.4f}")
    print(f"residual std: {residual.std():.4f} (gt_margin std {gt.std():.4f})\n")

    hidden = {layer: h.float() for layer, h in sorted(record["hidden"].items())}
    families = {}
    reduced = []
    for layer, h in hidden.items():
        pca = PCA(n_components=PCA_DIMS, random_state=SEED)
        pca.fit(h[train].numpy())
        z = pca.transform(h.numpy())
        families[f"hidden[{layer}] pca{PCA_DIMS}"] = z
        reduced.append(z)
    families[f"hidden pca{PCA_DIMS}"] = np.concatenate(reduced, axis=1)
    families["hidden (all, raw)"] = torch.cat(list(hidden.values()), dim=1).numpy()
    families["output (control)"] = output

    header = (f"{'features -> residual':>24} {'R2':>8} {'rho':>8} {'alpha':>8}"
              + "".join(f"{'t=' + f'{t:.1f}':>9}" for t in t_values))
    print(header)
    print("-" * len(header))

    for name, x in families.items():
        pred, alpha = ridge_predict(x, residual, train, residual, val=val)
        per_t = []
        for t in t_values:
            m = (record["t"] == t).numpy()
            p, _ = ridge_predict(x[m], residual[m], (train.numpy() & m)[m], residual[m],
                                 val=(val.numpy() & m)[m])
            tm = (test.numpy() & m)[m]
            per_t.append(r2(p[tm], residual[m][tm]))
        print(
            f"{name:>24} {r2(pred[test], residual[test]):8.4f} "
            f"{float(spearmanr(pred[test], residual[test]).statistic):8.4f} "
            f"{alpha:8.0f}" + "".join(f"{v:9.4f}" for v in per_t)
        )

if __name__ == "__main__":
    main()