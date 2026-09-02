"""Does the readiness advantage of the residual stream grow or shrink as the
model learns?

For each training checkpoint: collect wait-gain labels, fit the probe families
over several split seeds, and plot the advantage against training step.

Run collect_waitgain.py logic inline so nothing has to be staged to disk twice.
"""

from __future__ import annotations

import csv
import os
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

from src.dlm_interp.model import ModelConfig, NanoMDLM
from src.dlm_interp.samplers import HiddenCapture, score_positions

CKPT_DIR = "baseline_s0"
STEPS = (5000, 10000, 15000, 20000, 25000, 30000)
VAL_BIN = "data/tinystories/val.bin"
CACHE = "cache_waitgain"

N_WINDOWS = 150
T_VALUES = (0.8, 0.5, 0.2)
REVEAL_FRACTION = 0.5
LAYERS = (1, 3, 5)
SEEDS = (0, 1, 2)
ALPHAS = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5)
PCA_DIMS = 50
TEST_FRACTION = VAL_FRACTION = 0.25
FAMILIES = ("output + t", "hidden pca50", "output + t + hidden pca50")


# ------------------------------------------------------------- collection

def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    model = NanoMDLM(ModelConfig(**ckpt["model_cfg"]))
    model.load_state_dict(state)
    model.eval()
    return model, ckpt["model_cfg"], ckpt.get("val_nelbo")


def sample_windows(path, n_windows, seq_len, rng):
    data = np.memmap(path, dtype=np.uint16, mode="r")
    starts = rng.integers(0, len(data) - seq_len, size=n_windows)
    return torch.from_numpy(np.stack([data[s : s + seq_len].astype(np.int64) for s in starts]))


def gt_margin(logits, truth):
    probs = logits.softmax(-1)
    p_true = probs.gather(-1, truth.unsqueeze(-1)).squeeze(-1)
    p_others = probs.scatter(-1, truth.unsqueeze(-1), 0.0)
    return p_true - p_others.max(-1).values


@torch.no_grad()
def collect(model, windows, mask_id, generator):
    modules = {i: model.blocks[i] for i in LAYERS}
    modules[len(model.blocks)] = model.ln_f
    capture = HiddenCapture(modules)

    fields = ("wait_gain", "confidence", "entropy", "margin", "t", "window")
    out = {key: [] for key in fields}
    hidden_out: dict[int, list[torch.Tensor]] = {}

    with capture:
        def forward(x, t):
            capture.buffer.clear()
            h, _ = model(x)
            return model.logits(h)[0], dict(capture.buffer)

        for w_idx, clean in enumerate(windows):
            for t in T_VALUES:
                masked = torch.rand(clean.shape, generator=generator) < t
                if int(masked.sum()) < 4:
                    continue
                x_now = torch.where(masked, torch.full_like(clean, mask_id), clean)
                reveal = masked & (torch.rand(clean.shape, generator=generator) < REVEAL_FRACTION)
                still = masked & ~reveal
                if int(still.sum()) < 2:
                    continue
                x_later = torch.where(still, torch.full_like(clean, mask_id), clean)
                positions = still.nonzero(as_tuple=False).squeeze(-1)

                logits_now, hidden = forward(x_now.unsqueeze(0), float(t))
                logits_now = logits_now[positions].float()
                logits_now[:, mask_id] = float("-inf")
                logits_later, _ = forward(x_later.unsqueeze(0), float(t))
                logits_later = logits_later[positions].float()
                logits_later[:, mask_id] = float("-inf")

                truth = clean[positions]
                confidence, entropy, margin, _, _ = score_positions(logits_now)
                n = positions.numel()
                values = {
                    "wait_gain": gt_margin(logits_later, truth) - gt_margin(logits_now, truth),
                    "confidence": confidence,
                    "entropy": entropy,
                    "margin": margin,
                    "t": torch.full((n,), t),
                    "window": torch.full((n,), w_idx, dtype=torch.long),
                }
                for key, value in values.items():
                    out[key].append(value)
                for layer, h in hidden.items():
                    hidden_out.setdefault(layer, []).append(h[positions].half())

    record = {key: torch.cat(value) for key, value in out.items()}
    record["hidden"] = {layer: torch.cat(value) for layer, value in hidden_out.items()}
    return record


# ------------------------------------------------------------- probing

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


def fit_and_score(x, y, train, val, test):
    assert np.isfinite(x).all(), "features contain NaN or inf "\
        "(check the entropy fix in score_positions)"
    scaler = StandardScaler().fit(x[train])
    xt, xv = scaler.transform(x[train]), scaler.transform(x[val])
    scores = [r2(Ridge(alpha=a).fit(xt, y[train]).predict(xv), y[val]) for a in ALPHAS]
    alpha = ALPHAS[int(np.argmax(scores))]

    fit = train | val
    scaler = StandardScaler().fit(x[fit])
    pred = Ridge(alpha=alpha).fit(scaler.transform(x[fit]), y[fit]).predict(
        scaler.transform(x[test])
    )
    correlation = 0.0 if np.std(pred) == 0 else float(spearmanr(pred, y[test]).statistic)
    return r2(pred, y[test]), correlation


def build_families(record, train):
    output = torch.stack(
        [record["confidence"], record["entropy"], record["margin"]], dim=1
    ).numpy()
    t_col = record["t"][:, None].numpy()

    reduced = []
    for _, h in sorted(record["hidden"].items()):
        h = h.float().numpy()
        dims = min(PCA_DIMS, h.shape[1], int(train.sum()))
        reduced.append(PCA(n_components=dims, random_state=0).fit(h[train]).transform(h))
    hidden = np.concatenate(reduced, axis=1)

    base = np.concatenate([output, t_col], axis=1)
    return {
        "output + t": base,
        "hidden pca50": hidden,
        "output + t + hidden pca50": np.concatenate([base, hidden], axis=1),
    }


def probe(record):
    y = record["wait_gain"].float().numpy()
    scores = {name: {"r2": [], "rho": []} for name in FAMILIES}
    for seed in SEEDS:
        train, val, test = three_way_split(record["window"], seed)
        families = build_families(record, train.numpy())
        for name, x in families.items():
            score, correlation = fit_and_score(x, y, train, val, test)
            scores[name]["r2"].append(score)
            scores[name]["rho"].append(correlation)
    return scores


def summarise(values):
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


# ------------------------------------------------------------- main

def main():
    os.makedirs(CACHE, exist_ok=True)
    rows = []

    for step in STEPS:
        path = os.path.join(CKPT_DIR, f"step{step}.pt")
        if not os.path.exists(path):
            print(f"skipping step {step}: {path} not found")
            continue

        cached = os.path.join(CACHE, f"waitgain_step{step}.pt")
        if os.path.exists(cached):
            record = torch.load(cached, weights_only=False)
            model, cfg, nelbo = None, None, None
        else:
            model, cfg, nelbo = load_checkpoint(path)
            windows = sample_windows(VAL_BIN, N_WINDOWS, int(cfg["seq_len"]),
                                     np.random.default_rng(0))
            record = collect(model, windows, int(cfg["mask_id"]),
                             torch.Generator("cpu").manual_seed(0))
            torch.save(record, cached)

        scores = probe(record)
        gain_mean = float(record["wait_gain"].mean())
        row = {"step": step, "n": int(record["wait_gain"].numel()), "wait_gain": gain_mean}
        for name in FAMILIES:
            for metric in ("r2", "rho"):
                mean, std = summarise(scores[name][metric])
                row[f"{name}|{metric}"] = mean
                row[f"{name}|{metric}_sd"] = std
        rows.append(row)

        base = row["output + t|r2"]
        both = row["output + t + hidden pca50|r2"]
        print(f"step {step:6d}  n={row['n']:6d}  mean wait_gain {gain_mean:.4f}  "
              f"baseline R2 {base:.4f}  +hidden {both:.4f}  advantage {both - base:+.4f}")

    if not rows:
        print("no checkpoints processed")
        return

    with open("checkpoint_curve.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote checkpoint_curve.csv")

    steps = [r["step"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    for name, colour, marker in (
        ("output + t", "0.45", "o"),
        ("hidden pca50", "#c1440e", "s"),
        ("output + t + hidden pca50", "#7d2e00", "^"),
    ):
        means = [r[f"{name}|r2"] for r in rows]
        errs = [r[f"{name}|r2_sd"] for r in rows]
        ax1.errorbar(steps, means, yerr=errs, fmt=marker + "-", color=colour,
                     capsize=3, label=name)
    ax1.set_xlabel("training step")
    ax1.set_ylabel("out-of-sample $R^2$ on wait gain")
    ax1.set_title("Readiness decodability across training", fontsize=10)
    ax1.legend(fontsize=8, frameon=False)
    ax1.spines[["top", "right"]].set_visible(False)

    advantage = [r["output + t + hidden pca50|r2"] - r["output + t|r2"] for r in rows]
    ax2.plot(steps, advantage, "^-", color="#7d2e00")
    ax2.axhline(0, color="k", lw=0.6)
    ax2.set_xlabel("training step")
    ax2.set_ylabel("$R^2$ gain from internal states")
    ax2.set_title("What the residual stream adds over output signals", fontsize=10)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig("fig_checkpoint_curve.png", dpi=160)
    print("wrote fig_checkpoint_curve.png")


if __name__ == "__main__":
    main()