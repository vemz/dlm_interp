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

LABELS = Path(sys.argv[1]) if len(sys.argv) > 1 else RUNS / "labels_penalty_horizon.pt"
SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)
ALPHAS = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5)
PCA_DIMS = 20
TEST_FRACTION = VAL_FRACTION = 0.25
COV = ("step", "n_masked")
OUTPUT = ("conf_mean", "conf_min", "conf_max", "conf_spread", "entropy_mean", "margin_mean")
N_BOOT = 2000
BOOT_SEED = 12345
TITLE = ("Commit dependence is decodable from the residual stream up to four\n"
         "steps before the decision, and better from shallow layers")
BASE_WITH_MARGINAL = False

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

def reduce_one(h, train, dims=PCA_DIMS):
    h = h.float().numpy()
    n = min(dims, h.shape[1], int(train.sum()))
    return PCA(n_components=n, random_state=0).fit(h[train]).transform(h)

def cluster_stats(y, pred, rows_by_gen):
    """Per-generation sufficient statistics for R^2 on any resample."""
    n = np.array([len(r) for r in rows_by_gen], dtype=np.float64)
    s = np.array([y[r].sum() for r in rows_by_gen], dtype=np.float64)
    q = np.array([(y[r] ** 2).sum() for r in rows_by_gen], dtype=np.float64)
    sse = np.array([((y[r] - pred[r]) ** 2).sum() for r in rows_by_gen], dtype=np.float64)
    return n, s, q, sse

def boot_r2(counts, n, s, q, sse):
    total_n = counts @ n
    total_s = counts @ s
    total_q = counts @ q
    total_e = counts @ sse
    ss_tot = total_q - total_s**2 / total_n
    return np.where(ss_tot > 0, 1.0 - total_e / ss_tot, np.nan)

def interval(values):
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi), float((values > 0).mean())

def main():
    record = torch.load(LABELS, weights_only=False)
    groups = record["generation"]
    horizons = record["horizons"]
    layers = sorted(record["hidden"][horizons[0]])
    gnp = groups.numpy()

    cov = torch.stack([record[k] for k in COV], dim=1).numpy()
    out_feat = torch.stack([record[k] for k in OUTPUT], dim=1).numpy()
    marginal = record["marginal"].numpy()[:, None]
    base = np.concatenate([cov, out_feat], axis=1)
    if BASE_WITH_MARGINAL:
        base = np.concatenate([base, marginal], axis=1)
    chain = record["chain"].numpy()

    names = [f"layer {l}" for l in layers] + ["all layers", "mismatched"]
    point = {h: {name: [] for name in names} for h in horizons}
    draws = {h: {name: [] for name in names} for h in horizons}       
    raw = {h: {name: [] for name in names} for h in horizons}     
    depth = {h: [] for h in horizons}                               
    base_point, base_raw = [], []

    boot_rng = np.random.default_rng(BOOT_SEED)
    shuffle_rng = np.random.default_rng(0)
    shallow, deep = f"layer {layers[0]}", f"layer {layers[-1]}"

    for seed in SEEDS:
        train, val, test = three_way_split(groups, seed)
        tr, te = train.numpy(), test.numpy()
        y = chain - fit_predict(marginal, chain, train, val)

        test_idx = np.where(te)[0]
        test_gens = np.unique(gnp[test_idx])
        rows_by_gen = [test_idx[gnp[test_idx] == g] for g in test_gens]
        counts = boot_rng.multinomial(
            len(test_gens), np.full(len(test_gens), 1.0 / len(test_gens)), size=N_BOOT
        ).astype(np.float64)

        pred_base = fit_predict(base, y, train, val)
        n, s, q, sse = cluster_stats(y, pred_base, rows_by_gen)
        base_boot = boot_r2(counts, n, s, q, sse)
        base_point.append(r2(pred_base[te], y[te]))
        base_raw.append(base_boot)

        for h in horizons:
            reduced = {l: reduce_one(record["hidden"][h][l], tr) for l in layers}
            stacked = np.concatenate(list(reduced.values()), axis=1)
            feature_sets = {f"layer {l}": reduced[l] for l in layers}
            feature_sets["all layers"] = stacked

            idx = np.arange(len(y))
            for gen in np.unique(gnp):
                m = gnp == gen
                idx[m] = shuffle_rng.permutation(idx[m])
            feature_sets["mismatched"] = stacked[idx]

            per_name = {}
            for name, extra in feature_sets.items():
                pred = fit_predict(np.concatenate([base, extra], axis=1), y, train, val)
                point[h][name].append(r2(pred[te], y[te]))
                _, _, _, sse_p = cluster_stats(y, pred, rows_by_gen)
                b = boot_r2(counts, n, s, q, sse_p)
                per_name[name] = b
                raw[h][name].append(b)
                draws[h][name].append(b - base_boot)
            depth[h].append(per_name[shallow] - per_name[deep])
        print(f"  seed {seed} done", flush=True)

    b_mean, b_sd = float(np.mean(base_point)), float(np.std(base_point))
    print(f"\n{len(chain)} steps, {len(np.unique(gnp))} generations, {len(SEEDS)} split seeds")
    print(f"baseline (cov + output{' + marginal' if BASE_WITH_MARGINAL else ''}): "
          f"R2 {b_mean:.4f} +/- {b_sd:.4f}")
    print(f"cluster bootstrap: {N_BOOT} draws per seed over the "
          f"{len(np.unique(gnp)) // 4} test generations\n")

    header = (f"{'h':>4} {'features':>12} {'delta':>9} {'seed +-2se':>11} "
              f"{'boot 95% CI':>21} {'wider':>7} {'P(d>0)':>8}  verdict")
    print(header)
    print("-" * (len(header) + 4))

    rows = []
    for h in horizons:
        for name in names:
            paired = np.array(point[h][name]) - np.array(base_point)
            m, sd = float(paired.mean()), float(paired.std())
            seed_half = 2 * sd / np.sqrt(len(SEEDS))     
            pooled = np.concatenate(draws[h][name])
            lo, hi, frac = interval(pooled)
            boot_half = (hi - lo) / 2
            ratio = boot_half / seed_half if seed_half > 0 else float("nan")
            verdict = "clears zero" if lo > 0 else "not clear"
            star = "*" if m - seed_half > 0 else " "
            print(f"{h:>4} {name:>12} {m:>+9.4f} {f'+-{seed_half:.4f}{star}':>11} "
                  f"{f'({lo:+.4f},{hi:+.4f})':>21} {ratio:>6.1f}x {frac:>8.3f}  {verdict}")
            rows.append([h, name, f"{m:.4f}", f"{sd:.4f}", f"{seed_half:.4f}",
                         f"{lo:.4f}", f"{hi:.4f}", f"{ratio:.2f}", f"{frac:.4f}",
                         "clears" if lo > 0 else "not_clear"])
        print("-" * (len(header) + 4))

    print(f"\n{shallow} minus {deep}, paired inside each draw:")
    print(f"{'h':>4} {'delta':>9} {'boot 95% CI':>21} {'P(d>0)':>8}")
    print("-" * 45)
    depth_rows = []
    for h in horizons:
        pooled = np.concatenate(depth[h])
        lo, hi, frac = interval(pooled)
        m = float(np.mean(point[h][shallow]) - np.mean(point[h][deep]))
        print(f"{h:>4} {m:>+9.4f} {f'({lo:+.4f},{hi:+.4f})':>21} {frac:>8.3f}")
        depth_rows.append([h, f"{m:.4f}", f"{lo:.4f}", f"{hi:.4f}", f"{frac:.4f}"])

    out = RESULTS / "rq1_horizon_bootstrap.csv"
    with open(out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["horizon", "features", "delta_mean", "delta_sd", "seed_half_width",
                         "boot_lo", "boot_hi", "width_ratio", "p_positive", "verdict"])
        writer.writerows(rows)
        writer.writerow([])
        writer.writerow(["horizon", f"{shallow} - {deep}", "boot_lo", "boot_hi", "p_positive"])
        writer.writerows(depth_rows)
    print(f"\nwrote {out}")

    make_figure(horizons, layers, names, point, raw, base_point, base_raw)

def make_figure(horizons, layers, names, point, raw, base_point, base_raw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def band(name):
        centre = np.array([np.mean(point[h][name]) for h in horizons])
        los, his = [], []
        for h in horizons:
            lo, hi = np.percentile(np.concatenate(raw[h][name]), [2.5, 97.5])
            los.append(lo)
            his.append(hi)
        lower = np.clip(centre - np.array(los), 0, None)
        upper = np.clip(np.array(his) - centre, 0, None)
        return centre, np.vstack([lower, upper])

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    palette = ["#f0a07c", "#d4693a", "#8f3b12"]          # one hue, light -> deep layer
    x = np.arange(len(horizons))

    for offset, (colour, l) in zip((-0.06, 0.0, 0.06), zip(palette, layers)):
        centre, err = band(f"layer {l}")
        ax.errorbar(x + offset, centre, yerr=err, fmt="o-", color=colour, capsize=3,
                    markersize=4, linewidth=1.6, elinewidth=1.2, label=f"layer {l}")
    centre, err = band("all layers")
    ax.errorbar(x, centre, yerr=err, fmt="^-", color="#3b1a08", capsize=3,
                markersize=5, linewidth=1.6, elinewidth=1.2, label="all layers")

    centre, _ = band("mismatched")
    ax.plot(x, centre, "x--", color="0.60", linewidth=1.4, label="mismatched states")

    b_mean = float(np.mean(base_point))
    b_lo, b_hi = np.percentile(np.concatenate(base_raw), [2.5, 97.5])
    ax.axhline(b_mean, color="0.35", ls=":", linewidth=1.4, label="cov + output only")
    ax.axhspan(b_lo, b_hi, color="0.35", alpha=0.10, linewidth=0)

    ax.set_xticks(x, [str(h) for h in horizons])
    ax.set_xlabel("steps before the decision")
    ax.set_ylabel("out-of-sample $R^2$")
    ax.set_title(TITLE, fontsize=10)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.92", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = RESULTS / "fig_rq1_horizon_bootstrap.png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")
    print("error bars and the shaded baseline band are generation-clustered "
          "bootstrap 95% intervals, pooled over split seeds")


if __name__ == "__main__":
    main()
