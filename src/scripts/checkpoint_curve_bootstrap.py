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

from src.dlm_interp.paths import RESULTS, ROOT

CACHE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "cache_waitgain"
STEPS = (5000, 10000, 15000, 20000, 25000, 30000)
SEEDS = (0, 1, 2)                        # as in checkpoint_curve.py
ALPHAS = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5)
DIMS = (50, None)                        # None = no PCA
TEST_FRACTION = VAL_FRACTION = 0.25
OUTPUT = ("confidence", "entropy", "margin")
N_BOOT = 2000
BOOT_SEED = 12345
MIN_BASE = 0.005                         # below this the ratio is not defined


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
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi), float((values > 0).mean())


def main():
    available = [s for s in STEPS if (CACHE / f"waitgain_step{s}.pt").exists()]
    if not available:
        print(f"no cached records in {CACHE}")
        return
    print(f"checkpoints: {available}")

    # boot[dims][step] = list over seeds of (base_draws, full_draws)
    boot = {d: {s: [] for s in available} for d in DIMS}
    point = {d: {s: {"base": [], "full": []} for s in available} for d in DIMS}
    counts_by_seed, windows_by_seed = {}, {}

    for step in available:
        record = torch.load(CACHE / f"waitgain_step{step}.pt", weights_only=False)
        y = record["wait_gain"].float().numpy()
        windows = record["window"]
        wnp = windows.numpy()
        layers = sorted(record["hidden"])
        out_feat = torch.stack([record[k] for k in OUTPUT], dim=1).numpy()
        base_x = np.concatenate([out_feat, record["t"][:, None].numpy()], axis=1)
        max_dims = max((d for d in DIMS if d is not None), default=50)

        for seed in SEEDS:
            train, val, test = three_way_split(windows, seed)
            tr, te = train.numpy(), test.numpy()
            test_windows = np.unique(wnp[te])
            test_idx = np.where(te)[0]
            rows_by_group = [test_idx[wnp[test_idx] == w] for w in test_windows]

            # The draws must be shared across checkpoints for the trend to be
            # paired, which is only legitimate if the windows really do match.
            if seed not in counts_by_seed:
                windows_by_seed[seed] = test_windows
                rng = np.random.default_rng(BOOT_SEED + seed)
                counts_by_seed[seed] = rng.multinomial(
                    len(test_windows), np.full(len(test_windows), 1.0 / len(test_windows)),
                    size=N_BOOT).astype(np.float64)
            elif not np.array_equal(windows_by_seed[seed], test_windows):
                raise SystemExit(
                    f"step {step}, seed {seed}: test windows differ from the first "
                    "checkpoint, so the trend cannot be paired")
            counts = counts_by_seed[seed]

            pred = fit_predict(base_x, y, train, val)
            n, s, q, sse_b = cluster_stats(y, pred, rows_by_group)
            base_draws = boot_r2(counts, n, s, q, sse_b)
            base_pt = r2(pred[te], y[te])

            projected = {}
            for l in layers:
                h = record["hidden"][l].float().numpy()
                pca = PCA(n_components=min(max_dims, h.shape[1]), random_state=0).fit(h[tr])
                projected[l] = pca.transform(h).astype(np.float32)
                del h

            for dims in DIMS:
                if dims is None:
                    hidden = np.concatenate(
                        [record["hidden"][l].float().numpy() for l in layers], axis=1)
                else:
                    hidden = np.concatenate([projected[l][:, :dims] for l in layers], axis=1)
                pred = fit_predict(np.concatenate([base_x, hidden], axis=1), y, train, val)
                _, _, _, sse_f = cluster_stats(y, pred, rows_by_group)
                boot[dims][step].append((base_draws, boot_r2(counts, n, s, q, sse_f)))
                point[dims][step]["base"].append(base_pt)
                point[dims][step]["full"].append(r2(pred[te], y[te]))
                del hidden
            del projected
        print(f"  step {step} done", flush=True)
        del record

    rows = []
    for dims in DIMS:
        label = dims if dims is not None else "no PCA"
        print("\n" + "=" * 84)
        print(f"advantage of the residual stream over 'output + t'   [{label} per layer]")
        print("=" * 84)
        header = (f"{'step':>7} {'base R2':>9} {'advantage':>11} {'95% CI':>21} "
                  f"{'relative':>10} {'95% CI':>19}")
        print(header)
        print("-" * len(header))
        rel_draws = {}
        for step in available:
            abs_d = np.concatenate([f - b for b, f in boot[dims][step]])
            base_d = np.concatenate([b for b, _ in boot[dims][step]])
            rel = np.where(base_d > MIN_BASE, abs_d / base_d, np.nan)
            rel_draws[step] = rel
            a_lo, a_hi, _ = interval(abs_d)
            r_lo, r_hi, _ = interval(rel)
            base_pt = float(np.mean(point[dims][step]["base"]))
            adv_pt = float(np.mean(point[dims][step]["full"])) - base_pt
            print(f"{step:>7} {base_pt:>9.4f} {adv_pt:>+11.4f} "
                  f"{f'({a_lo:+.4f},{a_hi:+.4f})':>21} "
                  f"{adv_pt / base_pt:>9.1%} {f'({r_lo:.0%},{r_hi:.0%})':>19}")
            rows.append([label, step, f"{base_pt:.4f}", f"{adv_pt:.4f}",
                         f"{a_lo:.4f}", f"{a_hi:.4f}", f"{adv_pt / base_pt:.4f}",
                         f"{r_lo:.4f}", f"{r_hi:.4f}"])

        first, last = available[0], available[-1]
        print(f"\n  the claim is a trend, so difference it inside each draw "
              f"({first} minus {last}):")
        abs_first = np.concatenate([f - b for b, f in boot[dims][first]])
        abs_last = np.concatenate([f - b for b, f in boot[dims][last]])
        d_abs = abs_first - abs_last
        lo, hi, frac = interval(d_abs)
        print(f"    absolute advantage : {np.mean(d_abs):>+8.4f}  "
              f"({lo:+.4f},{hi:+.4f})   P(>0) {frac:.3f}")
        d_rel = rel_draws[first] - rel_draws[last]
        lo, hi, frac = interval(d_rel)
        print(f"    relative advantage : {np.nanmean(d_rel):>+8.1%}  "
              f"({lo:+.0%},{hi:+.0%})   P(>0) {frac:.3f}")
        print("    the absolute claim is that the first line does NOT clear zero;")
        print("    the prediction for LLaDA/Dream is that the second one does.")

    out = RESULTS / "checkpoint_curve_bootstrap.csv"
    with open(out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dims", "step", "base_r2", "advantage", "adv_lo", "adv_hi",
                         "relative", "rel_lo", "rel_hi"])
        writer.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
