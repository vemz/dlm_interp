from __future__ import annotations
import csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from src.dlm_interp.paths import RESULTS, RUNS, TAG

LABELS = RUNS / "labels_waitgain.pt"
SEEDS = (0, 1, 2, 3, 4)
ALPHAS = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5)
TEST_FRACTION = VAL_FRACTION = 0.25
READINESS = "wait_gain"
OUTPUT_STATS = ("confidence", "entropy", "margin")

def three_way_split(window, seed):
    ids = np.unique(window)
    perm = np.random.default_rng(seed).permutation(len(ids))
    n_test = int(round(len(ids) * TEST_FRACTION))
    n_val = int(round(len(ids) * VAL_FRACTION))
    test_ids = set(ids[perm[:n_test]].tolist())
    val_ids = set(ids[perm[n_test:n_test + n_val]].tolist())
    test = np.array([w in test_ids for w in window])
    val = np.array([w in val_ids for w in window])
    return ~(test | val), val, test

def r2(pred, truth):
    ss_res = float(((truth - pred) ** 2).sum())
    ss_tot = float(((truth - truth.mean()) ** 2).sum())
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

def fit(x, y, train, val):
    scaler = StandardScaler().fit(x[train])
    xt, xv = scaler.transform(x[train]), scaler.transform(x[val])
    scores = [r2(Ridge(alpha=a).fit(xt, y[train]).predict(xv), y[val]) for a in ALPHAS]
    alpha = ALPHAS[int(np.argmax(scores))]
    keep = train | val
    scaler = StandardScaler().fit(x[keep])
    model = Ridge(alpha=alpha).fit(scaler.transform(x[keep]), y[keep])
    direction = model.coef_ / np.maximum(scaler.scale_, 1e-9)
    direction = direction / max(np.linalg.norm(direction), 1e-12)
    return direction, (lambda z: model.predict(scaler.transform(z))), alpha

def rank1_r2(x, y, direction, train, val, test):
    p = (x @ direction)[:, None]
    _, predict, _ = fit(p, y, train, val)
    return r2(predict(p[test]), y[test])

def orthonormal_basis(directions, tol=1e-8):
    d = np.stack(directions, axis=1)
    u, s, _ = np.linalg.svd(d, full_matrices=False)
    keep = s > tol * s[0]
    return u[:, keep], s

def deflate_subspace(x, basis):
    return x - (x @ basis) @ basis.T

def pairwise_cos(vs):
    return [abs(float(vs[i] @ vs[j]))
            for i in range(len(vs)) for j in range(i + 1, len(vs))]

def random_cosine_null(dim, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, dim))
    b = rng.standard_normal((n, dim))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    cos = np.abs((a * b).sum(1))
    return float(cos.mean()), float(np.percentile(cos, 95))

def random_inside_null(dim, k, n=20000, seed=1):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    frac = np.sqrt((v[:, :k] ** 2).sum(1))
    return float(frac.mean()), float(np.percentile(frac, 95))

def analyse(record, layers, label, rows_out):
    x = np.concatenate([record["hidden"][l].float().numpy() for l in layers], axis=1)
    y = {k: record[k].float().numpy() for k in (READINESS,) + OUTPUT_STATS}
    window = record["window"].numpy()
    dim = x.shape[1]

    cos_mean, cos_p95 = random_cosine_null(dim)

    print("\n" + "=" * 78)
    print(f"{label} — {x.shape[0]} rows, {dim} dims, {len(np.unique(window))} windows")
    print("=" * 78)
    print(f"  null |cos| in {dim} dims: mean {cos_mean:.4f}, 95th pct {cos_p95:.4f}")

    acc = {}

    def push(key, value):
        acc.setdefault(key, []).append(value)

    u_by_split, c_by_split = [], []

    for seed in SEEDS:
        train, val, test = three_way_split(window, seed)

        u, pred_u, alpha_u = fit(x, y[READINESS], train, val)
        push("r2_ready", r2(pred_u(x[test]), y[READINESS][test]))
        push("r2_ready_rank1", rank1_r2(x, y[READINESS], u, train, val, test))
        push("alpha_ready", float(alpha_u))

        dirs = []
        for stat in OUTPUT_STATS:
            d, pred, _ = fit(x, y[stat], train, val)
            dirs.append(d)
            push(f"r2_{stat}", r2(pred(x[test]), y[stat][test]))
            push(f"cos_u_{stat}", abs(float(u @ d)))

        u_by_split.append(u)
        c_by_split.append(dirs[0])

        basis, svals = orthonormal_basis(dirs)
        push("subspace_rank", float(basis.shape[1]))
        for i, sv in enumerate(svals):
            push(f"singular_{i}", float(sv))
        push("u_inside_subspace", float(np.linalg.norm(basis.T @ u)))

        x_out = deflate_subspace(x, basis)
        _, p, _ = fit(x_out, y[READINESS], train, val)
        push("r2_ready_deflated", r2(p(x_out[test]), y[READINESS][test]))

        x_no_u = x - np.outer(x @ u, u)
        for stat in OUTPUT_STATS:
            _, p, _ = fit(x_no_u, y[stat], train, val)
            push(f"r2_{stat}_deflated", r2(p(x_no_u[test]), y[stat][test]))

        print(f"  split seed {seed} done", flush=True)

    assert len(u_by_split) == len(SEEDS) == len(c_by_split), (
        f"collected {len(u_by_split)} readiness directions and "
        f"{len(c_by_split)} confidence directions for {len(SEEDS)} splits — "
        "the append is in the wrong scope and the stability table below would "
        "be comparing duplicates")

    def s(key):
        v = np.array(acc[key])
        return float(v.mean()), float(v.std())

    print(f"\n  {'quantity':>26} {'mean':>9} {'sd':>8}")
    print("  " + "-" * 45)
    for key in sorted(acc):
        m, sd = s(key)
        print(f"  {key:>26} {m:>+9.4f} {sd:>8.4f}")
        rows_out.append([TAG or "_s0", label, key, f"{m:.4f}", f"{sd:.4f}"])

    pu, pc = pairwise_cos(u_by_split), pairwise_cos(c_by_split)
    print(f"\n  direction stability across the {len(SEEDS)} splits "
          f"(|cos| over {len(pu)} pairs):")
    print(f"    {'readiness':>12}  mean {np.mean(pu):.4f}   min {min(pu):.4f}")
    print(f"    {'confidence':>12}  mean {np.mean(pc):.4f}   min {min(pc):.4f}"
          f"   <- reference, fitted at R2 ~{s('r2_confidence')[0]:.2f}")
    rows_out.append([TAG or "_s0", label, "dirstab_ready_mean",
                     f"{np.mean(pu):.4f}", f"{min(pu):.4f}"])
    rows_out.append([TAG or "_s0", label, "dirstab_conf_mean",
                     f"{np.mean(pc):.4f}", f"{min(pc):.4f}"])
    if np.mean(pu) > 0.9999 and np.mean(pc) > 0.9999:
        print("    !! both exactly 1.0000 — that is the duplicate-append bug, not")
        print("       a result. Check the scope of u_by_split.append.")

    r_full, _ = s("r2_ready")
    r_one, _ = s("r2_ready_rank1")
    r_def, _ = s("r2_ready_deflated")
    inside, _ = s("u_inside_subspace")
    rank = int(round(s("subspace_rank")[0]))
    in_mean, in_p95 = random_inside_null(dim, rank)

    print("\n  reading:")
    frac = r_one / r_full if r_full > 0 else float("nan")
    print(f"    rank-1 recovers {frac:.0%} of the full readiness R2 — "
          + ("a direction is a fair description."
             if frac > 0.6 else
             "readiness is NOT one direction; the framing needs rank > 1."))

    if np.mean(pu) < 0.7 * np.mean(pc):
        print(f"    !! the readiness direction is NOT identified: it moves between")
        print(f"       splits ({np.mean(pu):.2f}) far more than confidence does")
        print(f"       ({np.mean(pc):.2f}). Report a subspace, not a direction, and")
        print(f"       do not quote cos(u, .) as a property of the model.")

    svs = [s(f"singular_{i}")[0] for i in range(3) if f"singular_{i}" in acc]
    print(f"    output subspace: rank {rank}, singular values "
          + ", ".join(f"{v:.3f}" for v in svs))
    if len(svs) >= 2 and svs[1] < 0.2 * svs[0]:
        print("    !! the three statistics are near-collinear: deflating them is")
        print("       deflating roughly one direction, and the test is weaker than")
        print("       the count of targets suggests.")

    print(f"    ‖P_span u‖ = {inside:.4f} against a null of {in_mean:.4f} "
          f"(95th pct {in_p95:.4f}) for a random direction and a random "
          f"{rank}-dim subspace.")

    keep_r = r_def / r_full if r_full > 0 else float("nan")
    worst_c = min((s(f"r2_{k}_deflated")[0] / s(f"r2_{k}")[0])
                  for k in OUTPUT_STATS if s(f"r2_{k}")[0] > 0)
    print(f"    deflating span(confidence, entropy, margin) keeps {keep_r:.0%} of "
          f"readiness R2;")
    print(f"    deflating u keeps at worst {worst_c:.0%} of an output statistic's R2.")

    if keep_r > 0.8 and worst_c > 0.8:
        print("    -> SEPARABLE FROM ALL THREE. Readiness survives removing the")
        print("       whole instantaneous output subspace, and the control confirms")
        print("       this is not just that a few directions out of many hardly")
        print("       matter. Scope: instantaneous statistics only — temporal KL is")
        print("       untested here and needs a trajectory collection.")
    elif keep_r < 0.5:
        print("    -> NOT SEPARABLE. Most of the readiness signal lies inside the")
        print("       span of the output statistics; an output-side score can reach")
        print("       it and the branch closes here.")
    else:
        print("    -> PARTIAL. Report the fraction, not a verdict.")

def main():
    if not LABELS.exists():
        raise SystemExit(f"{LABELS} not found — run collect_waitgain.py first")
    record = torch.load(LABELS, map_location="cpu", weights_only=False)
    layers = sorted(record["hidden"])
    deepest = layers[-1]
    print(f"model tag: {TAG or '_s0'}")
    print(f"layers: {layers} (layer {deepest} is ln_f — what a sampler already has")
    print("in hand at selection time, so it is the deployable case)")

    rows = []
    analyse(record, [deepest], f"ln_f (layer {deepest})", rows)
    analyse(record, layers, "all layers", rows)

    out = RESULTS / "readiness_direction.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["model", "features", "quantity", "mean", "sd_or_min"])
        w.writerows(rows)
    print(f"\nwrote {out}")
    print("\nRun the other two model seeds with DLM_TAG / DLM_CKPT before reading")
    print("any of this as a property of masked diffusion models rather than of one")
    print("checkpoint. compare_seeds.py is where the three get put side by side.")

if __name__ == "__main__":
    main()
