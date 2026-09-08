from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.dlm_interp.paths import RESULTS, RUNS

TARGETS = {
    "readiness": ("labels_waitgain.pt", "wait_gain"),
    "predictability": ("labels_gtmargin.pt", "gt_margin"),
}
SEEDS = (0, 1, 2, 3, 4)
ALPHAS = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5)
LADDER = (10, 25, 50, 100, 200, None)
TEST_FRACTION = VAL_FRACTION = 0.25
OUTPUT = ("confidence", "entropy", "margin")
N_BOOT = 2000
BOOT_SEED = 12345


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


def fit_predict(x, y, train, val):
    x = np.asarray(x, dtype=np.float32)
    scaler = StandardScaler().fit(x[train])
    xt, xv = scaler.transform(x[train]), scaler.transform(x[val])
    scores = [r2(Ridge(alpha=a).fit(xt, y[train]).predict(xv), y[val]) for a in ALPHAS]
    alpha = ALPHAS[int(np.argmax(scores))]
    fit = train | val
    scaler = StandardScaler().fit(x[fit])
    return Ridge(alpha=alpha).fit(scaler.transform(x[fit]), y[fit]).predict(scaler.transform(x))


def cluster_stats(y, pred, rows_by_group):
    n = np.array([len(r) for r in rows_by_group], dtype=np.float64)
    s = np.array([y[r].sum() for r in rows_by_group], dtype=np.float64)
    q = np.array([(y[r] ** 2).sum() for r in rows_by_group], dtype=np.float64)
    sse = np.array([((y[r] - pred[r]) ** 2).sum() for r in rows_by_group], dtype=np.float64)
    return n, s, q, sse


def boot_r2(counts, n, s, q, sse):
    total_n, total_s = counts @ n, counts @ s
    total_q, total_e = counts @ q, counts @ sse
    ss_tot = total_q - total_s**2 / total_n
    return np.where(ss_tot > 0, 1.0 - total_e / ss_tot, np.nan)


def interval(values):
    values = values[np.isfinite(values)]
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi), float((values > 0).mean())


def main():
    name = next((a for a in sys.argv[1:] if a in TARGETS), "readiness")
    path, target = TARGETS[name]
    record = torch.load(RUNS / path, map_location="cpu", weights_only=False)
    y = record[target].float().numpy()
    windows = record["window"]
    wnp = windows.numpy()
    layers = sorted(record["hidden"])
    out_feat = torch.stack([record[k] for k in OUTPUT], dim=1).numpy()
    base = np.concatenate([out_feat, record["t"][:, None].numpy()], axis=1)
    max_dims = max(d for d in LADDER if d is not None)
    shallow, deep = layers[0], layers[-1]

    print(f"{name}: {len(y)} rows, {len(np.unique(wnp))} windows, layers {layers}")
    print(f"(the deepest key is ln_f, not a block, where it is present)\n")

    point = {d: {l: [] for l in layers} for d in LADDER}
    gap = {d: [] for d in LADDER}
    base_point = []
    boot_rng = np.random.default_rng(BOOT_SEED)

    for seed in SEEDS:
        train, val, test = three_way_split(windows, seed)
        tr, te = train.numpy(), test.numpy()
        test_idx = np.where(te)[0]
        test_windows = np.unique(wnp[test_idx])
        rows_by_group = [test_idx[wnp[test_idx] == w] for w in test_windows]
        counts = boot_rng.multinomial(
            len(test_windows), np.full(len(test_windows), 1.0 / len(test_windows)),
            size=N_BOOT).astype(np.float64)

        pred = fit_predict(base, y, train, val)
        n, s, q, sse = cluster_stats(y, pred, rows_by_group)
        base_point.append(r2(pred[te], y[te]))

        projected = {}
        for l in layers:
            h = record["hidden"][l].float().numpy()
            projected[l] = PCA(n_components=min(max_dims, h.shape[1]),
                               random_state=0).fit(h[tr]).transform(h).astype(np.float32)
            del h

        for dims in LADDER:
            per_layer = {}
            for l in layers:
                feat = (record["hidden"][l].float().numpy() if dims is None
                        else projected[l][:, :dims])
                pred = fit_predict(np.concatenate([base, feat], axis=1), y, train, val)
                point[dims][l].append(r2(pred[te], y[te]))
                _, _, _, sse_l = cluster_stats(y, pred, rows_by_group)
                per_layer[l] = boot_r2(counts, n, s, q, sse_l)
                del feat
            gap[dims].append(per_layer[shallow] - per_layer[deep])
        del projected
        print(f"  seed {seed} done", flush=True)

    print(f"\nbaseline (output + t): R2 {np.mean(base_point):.4f}\n")
    header = (f"{'dims':>6}" + "".join(f"{f'layer {l}':>11}" for l in layers)
              + f"{f'L{shallow} - L{deep}':>13} {'95% CI':>21} {'P(>0)':>8}")
    print(header)
    print("-" * len(header))
    rows = []
    for dims in LADDER:
        label = dims if dims is not None else "none"
        line = f"{str(label):>6}" + "".join(f"{np.mean(point[dims][l]):>11.4f}" for l in layers)
        g = float(np.mean(point[dims][shallow]) - np.mean(point[dims][deep]))
        lo, hi, frac = interval(np.concatenate(gap[dims]))
        print(line + f"{g:>+13.4f} {f'({lo:+.4f},{hi:+.4f})':>21} {frac:>8.3f}")
        rows.append([name, label] + [f"{np.mean(point[dims][l]):.4f}" for l in layers]
                    + [f"{g:.4f}", f"{lo:.4f}", f"{hi:.4f}", f"{frac:.4f}"])

    print("\nA gap that closes as components are added is the Part 2 pattern reproduced")
    print("on unpooled states: the depth null holds and pooling is not the explanation.")
    print("A gap that survives to the last rung means Part 2's null was the pooling.")

    out = RESULTS / f"depth_part1_{name}.csv"
    with open(out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["target", "pca_dims"] + [f"layer {l}" for l in layers]
                        + ["gap", "boot_lo", "boot_hi", "p_positive"])
        writer.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
