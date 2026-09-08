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

from src.dlm_interp.paths import RESULTS, RUNS, TAG

TARGETS = {
    "predictability": ("labels_gtmargin.pt", "gt_margin"),
    "readiness": ("labels_waitgain.pt", "wait_gain"),
}
SEEDS = (0, 1, 2, 3, 4)                  # as in analysis.py
ALPHAS = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5)
LADDER = (10, 25, 50, 100, 200, None)    # None = no PCA, the raw residual stream
TEST_FRACTION = VAL_FRACTION = 0.25
OUTPUT = ("confidence", "entropy", "margin")
N_BOOT = 2000
BOOT_SEED = 12345
CACHE = RUNS / "part1_ladder_{}.npz"


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
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi), float((values > 0).mean())


def common_layers():
    sets = []
    for path, _ in TARGETS.values():
        record = torch.load(RUNS / path, map_location="cpu", weights_only=False)
        sets.append(set(record["hidden"].keys()))
        del record
    shared = sorted(set.intersection(*sets))
    dropped = sorted(set.union(*sets) - set(shared))
    return shared, dropped


def run_target(name, layers):
    path, target = TARGETS[name]
    record = torch.load(RUNS / path, map_location="cpu", weights_only=False)
    y = record[target].float().numpy()
    windows = record["window"]
    wnp = windows.numpy()

    out_feat = torch.stack([record[k] for k in OUTPUT], dim=1).numpy()
    t_col = record["t"][:, None].numpy()
    base = np.concatenate([out_feat, t_col], axis=1)          # "output + t"
    max_dims = max(d for d in LADDER if d is not None)

    store = {}
    for seed in SEEDS:
        train, val, test = three_way_split(windows, seed)
        tr, te = train.numpy(), test.numpy()
        test_windows = np.unique(wnp[te])
        test_idx = np.where(te)[0]
        rows_by_group = [test_idx[wnp[test_idx] == w] for w in test_windows]

        pred = fit_predict(base, y, train, val)
        n, s, q, sse_base = cluster_stats(y, pred, rows_by_group)
        store[f"{seed}/windows"] = test_windows
        store[f"{seed}/n"] = n
        store[f"{seed}/s"] = s
        store[f"{seed}/q"] = q
        store[f"{seed}/sse_base"] = sse_base

        # PCA components are nested, so fit once at the top rung per layer and
        # slice: the first k columns are what PCA(k) would have produced.
        projected = {}
        for l in layers:
            h = record["hidden"][l].float().numpy()
            pca = PCA(n_components=min(max_dims, h.shape[1]), random_state=0).fit(h[tr])
            projected[l] = pca.transform(h).astype(np.float32)
            del h

        for dims in LADDER:
            if dims is None:
                hidden = np.concatenate(
                    [record["hidden"][l].float().numpy() for l in layers], axis=1)
            else:
                hidden = np.concatenate([projected[l][:, :dims] for l in layers], axis=1)
            pred = fit_predict(np.concatenate([base, hidden], axis=1), y, train, val)
            _, _, _, sse = cluster_stats(y, pred, rows_by_group)
            store[f"{seed}/{dims}"] = sse
            del hidden
            print(f"    seed {seed}, dims {dims if dims is not None else 'none':>4}: "
                  f"R2 base {r2(fit_predict(base, y, train, val)[te], y[te]):.4f} -> "
                  f"{1 - sse.sum() / (q.sum() - s.sum()**2 / n.sum()):.4f}", flush=True)
        del projected

    np.savez(str(CACHE).format(name), **store)
    print(f"  cached {str(CACHE).format(name)}")


def load_cache(name):
    path = Path(str(CACHE).format(name))
    return dict(np.load(path)) if path.exists() else None


def report(name, cache):
    print("\n" + "=" * 78)
    print(f"{name}: increment of the residual stream over 'output + t'")
    print("=" * 78)
    header = f"{'dims':>6} {'increment':>11} {'boot 95% CI':>21} {'P(>0)':>8}"
    print(header)
    print("-" * len(header))
    boot_rng = np.random.default_rng(BOOT_SEED)
    per_dim = {}
    for dims in LADDER:
        key = dims if dims is not None else None
        pooled, points = [], []
        rng = np.random.default_rng(BOOT_SEED)
        for seed in SEEDS:
            n, s, q = cache[f"{seed}/n"], cache[f"{seed}/s"], cache[f"{seed}/q"]
            sse_b, sse_f = cache[f"{seed}/sse_base"], cache[f"{seed}/{key}"]
            counts = rng.multinomial(len(n), np.full(len(n), 1.0 / len(n)),
                                     size=N_BOOT).astype(np.float64)
            pooled.append(boot_r2(counts, n, s, q, sse_f) - boot_r2(counts, n, s, q, sse_b))
            tot = q.sum() - s.sum() ** 2 / n.sum()
            points.append((1 - sse_f.sum() / tot) - (1 - sse_b.sum() / tot))
        pooled = np.concatenate(pooled)
        lo, hi, frac = interval(pooled)
        label = dims if dims is not None else "none"
        print(f"{str(label):>6} {np.mean(points):>+11.4f} "
              f"{f'({lo:+.4f},{hi:+.4f})':>21} {frac:>8.3f}")
        per_dim[key] = pooled
    _ = boot_rng
    return per_dim


def dissociation(a, b):
    """Readiness increment minus predictability increment, paired by window."""
    print("\n" + "=" * 78)
    print("readiness increment MINUS predictability increment")
    print("=" * 78)
    print("   Both targets are grouped on the same 200 windows and split with the")
    print("   same seed, so each draw resamples the same test windows in both.\n")
    header = f"{'dims':>6} {'difference':>12} {'boot 95% CI':>21} {'P(>0)':>8}"
    print(header)
    print("-" * len(header))
    rows = []
    for dims in LADDER:
        key = dims if dims is not None else None
        pooled, points = [], []
        rng = np.random.default_rng(BOOT_SEED)
        for seed in SEEDS:
            wa, wb = a[f"{seed}/windows"], b[f"{seed}/windows"]
            if not np.array_equal(wa, wb):
                print(f"  seed {seed}: test windows differ between targets, skipped")
                continue
            counts = rng.multinomial(len(wa), np.full(len(wa), 1.0 / len(wa)),
                                     size=N_BOOT).astype(np.float64)
            deltas = []
            for cache in (b, a):          # readiness first, then predictability
                n, s, q = cache[f"{seed}/n"], cache[f"{seed}/s"], cache[f"{seed}/q"]
                deltas.append(boot_r2(counts, n, s, q, cache[f"{seed}/{key}"])
                              - boot_r2(counts, n, s, q, cache[f"{seed}/sse_base"]))
            pooled.append(deltas[0] - deltas[1])
            points.append(float(np.mean(deltas[0] - deltas[1])))
        if not pooled:
            continue
        lo, hi, frac = interval(np.concatenate(pooled))
        label = dims if dims is not None else "none"
        print(f"{str(label):>6} {np.mean(points):>+12.4f} "
              f"{f'({lo:+.4f},{hi:+.4f})':>21} {frac:>8.3f}")
        rows.append([TAG or "_s0", label, f"{np.mean(points):.4f}",
                     f"{lo:.4f}", f"{hi:.4f}", f"{frac:.4f}"])
    print("\n   A difference that clears zero at every rung is the dissociation.")
    print("   One that closes as components are added was the budget.")

    out = RESULTS / "part1_dissociation.csv"
    with open(out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "pca_dims", "difference",
                         "boot_lo", "boot_hi", "p_positive"])
        writer.writerows(rows)
    print(f"\nwrote {out}")


def main():
    wanted = [a for a in sys.argv[1:] if a in TARGETS] or list(TARGETS)
    layers, dropped = common_layers()
    print(f"layers used: {layers}")
    if dropped:
        print(f"layers dropped so the two targets get matched feature sets: {dropped}")

    for name in wanted:
        if load_cache(name) is None:
            print(f"\n=== {name} ({TARGETS[name][0]}) ===")
            run_target(name, layers)

    caches = {name: load_cache(name) for name in TARGETS}
    for name in wanted:
        if caches[name] is not None:
            report(name, caches[name])
    if all(c is not None for c in caches.values()):
        dissociation(caches["predictability"], caches["readiness"])
    else:
        missing = [n for n, c in caches.items() if c is None]
        print(f"\nrun {' and '.join(missing)} too for the dissociation table")


if __name__ == "__main__":
    main()
