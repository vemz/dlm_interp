from __future__ import annotations
import argparse
import csv
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from scipy.stats import spearmanr

from src.dlm_interp.load import load_model, nano_forward_fn
from src.dlm_interp.paths import RESULTS, RUNS
from src.dlm_interp.samplers import score_positions

LABELS = RUNS / "labels_penalty_horizon.pt"
OUT_FEATURES = RUNS / "attention_features.pt"

# Must match the collection that produced LABELS, or the replay desynchronises.
N_GENERATIONS = 200
K = 4
PROBE_EVERY = 2
MAX_HORIZON = 16
N_ORDERS = 6
SEED = 0

GATE_RATES = (0.05, 0.10, 0.20, 0.30)
N_BOOT = 2000
BOOT_SEED = 12345

class QKVCapture:

    def __init__(self, model, layers):
        self.modules = {l: model.blocks[l].attn.qkv for l in layers}
        self.n_heads = model.cfg.n_heads
        self.buffer: dict[int, torch.Tensor] = {}
        self.handles: list = []

    def _hook(self, layer):
        def fn(_module, _inputs, output):
            self.buffer[layer] = output.detach()[0]        # (L, 3D)
        return fn

    def __enter__(self):
        for layer, module in self.modules.items():
            self.handles.append(module.register_forward_hook(self._hook(layer)))
        return self

    def __exit__(self, *_):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def clear(self):
        self.buffer.clear()

    def attention(self, layer):
        qkv = self.buffer[layer]
        L, three_d = qkv.shape
        d_model = three_d // 3
        q, k, _ = qkv.split(d_model, dim=-1)
        h = self.n_heads
        d = d_model // h
        q = q.view(L, h, d).transpose(0, 1).float()
        k = k.view(L, h, d).transpose(0, 1).float()
        return torch.softmax(q @ k.transpose(-2, -1) / (d ** 0.5), dim=-1)

def bundle_attention(att, bundle, other_masked):
    b = torch.as_tensor(bundle, dtype=torch.long)
    a = att.mean(0)                                        # (L, L), over heads
    sub = a[b][:, b]                                       # (k, k)
    k = len(bundle)
    off = (sub.sum() - sub.diagonal().sum()) / max(k * (k - 1), 1)
    pair = (sub + sub.T) / 2
    pair = pair - torch.diag(torch.diagonal(pair))
    out = {
        "attn_in_bundle": float(off),
        "attn_max_pair": float(pair.max()) if k > 1 else 0.0,
    }
    if len(other_masked):
        o = torch.as_tensor(other_masked, dtype=torch.long)
        out["attn_to_other_masked"] = float(a[b][:, o].mean())
    else:
        out["attn_to_other_masked"] = 0.0
    visible = torch.ones(a.shape[0], dtype=torch.bool)
    visible[b] = False
    if len(other_masked):
        visible[torch.as_tensor(other_masked, dtype=torch.long)] = False
    out["attn_to_visible"] = float(a[b][:, visible].mean()) if visible.any() else 0.0
    return out

@torch.no_grad()
def replay(model, capture, layers, mask_id, seq_len, stored):
    forward_fn = nano_forward_fn(model)
    generator = torch.Generator("cpu").manual_seed(SEED)
    rows, row_id = [], 0

    for gen in range(N_GENERATIONS):
        x = torch.full((seq_len,), mask_id, dtype=torch.long)
        history = deque(maxlen=MAX_HORIZON + 1)

        for step in range(seq_len // K):
            if not (x == mask_id).any():
                break
            masked = (x == mask_id).nonzero(as_tuple=False).squeeze(-1)
            if masked.numel() <= K:
                break

            capture.clear()
            logits_all, _ = forward_fn(x.unsqueeze(0), 0.0)
            logits = logits_all[masked].float()
            logits[:, mask_id] = float("-inf")
            confidence, _, _, _, _ = score_positions(logits)

            chosen = confidence.topk(K).indices
            positions = masked[chosen]
            logprobs = logits[chosen].log_softmax(-1)
            tokens = torch.multinomial(logprobs.exp(), 1, generator=generator).squeeze(-1)
            marginal = float(logprobs.gather(-1, tokens.unsqueeze(-1)).sum())

            history.append(True)
            probed = step % PROBE_EVERY == 0 and len(history) > MAX_HORIZON

            if probed:
                for _ in range(N_ORDERS - 1):              # consume, discard
                    torch.randperm(K, generator=generator)

                assert abs(marginal - stored["marginal"][row_id]) < 1e-3, (
                    f"gen {gen} step {step}: replay diverged from the stored "
                    f"trajectory (marginal {marginal:+.4f} vs "
                    f"{stored['marginal'][row_id]:+.4f}). The RNG stream does not "
                    "match the collection — do not trust any join below."
                )

                bundle = positions.tolist()
                other = [int(p) for p in masked.tolist() if p not in set(bundle)]
                feats = {"generation": gen, "step": step}
                per_layer = [bundle_attention(capture.attention(l), bundle, other)
                             for l in layers]
                for key in per_layer[0]:
                    feats[key] = float(np.mean([p[key] for p in per_layer]))

                p = sorted(bundle)
                dists = [abs(a - b) for i, a in enumerate(p) for b in p[i + 1:]]
                feats["mean_dist"] = float(np.mean(dists))
                feats["min_gap"] = float(min(dists))
                rows.append(feats)
                row_id += 1

            x[positions] = tokens

    assert row_id == len(stored["marginal"]), (
        f"replay produced {row_id} probed steps, the labels file has "
        f"{len(stored['marginal'])}")
    return rows

def capture_curve(feature, cost, gen, rates, rng, n_boot=N_BOOT):
    gens = np.unique(gen)
    rows_by_gen = [np.where(gen == g)[0] for g in gens]
    out = {}
    for f in rates:
        draws = []
        for _ in range(n_boot):
            idx = np.concatenate([rows_by_gen[i]
                                  for i in rng.integers(0, len(gens), len(gens))])
            fe, co = feature[idx], cost[idx]
            k = max(1, int(len(idx) * f))
            share = co[np.argsort(-fe)[:k]].sum() / max(co.sum(), 1e-9)
            draws.append(share / f)
        draws = np.array(draws)
        k = max(1, int(len(cost) * f))
        point = (cost[np.argsort(-feature)[:k]].sum() / cost.sum()) / f
        out[f] = (point, *np.percentile(draws, [2.5, 97.5]))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="layers to average attention over (default: all)")
    ap.add_argument("--reuse", action="store_true",
                    help="skip the replay and use a cached feature file")
    args = ap.parse_args()

    stored = torch.load(LABELS, map_location="cpu", weights_only=False)
    stored = {k: (v.numpy() if torch.is_tensor(v) else v)
              for k, v in stored.items() if k != "hidden"}

    if args.reuse and OUT_FEATURES.exists():
        rows = torch.load(OUT_FEATURES, weights_only=False)
        # Reuse replayed attention features when available.
    else:
        model, cfg = load_model()
        mask_id, seq_len = int(cfg["mask_id"]), int(cfg["seq_len"])
        layers = args.layers if args.layers is not None else list(range(len(model.blocks)))
        # Replay the stored trajectories and capture attention features.
        capture = QKVCapture(model, layers)
        with capture:
            rows = replay(model, capture, layers, mask_id, seq_len, stored)
        torch.save(rows, OUT_FEATURES)
        

    gen = np.array([r["generation"] for r in rows])
    cost = np.maximum(-stored["penalty"], 0)

    features = {
        "attn_in_bundle":       np.array([r["attn_in_bundle"] for r in rows]),
        "attn_max_pair":        np.array([r["attn_max_pair"] for r in rows]),
        "attn_to_other_masked": np.array([r["attn_to_other_masked"] for r in rows]),
        "attn_to_visible":      np.array([r["attn_to_visible"] for r in rows]),
        "mean_dist (neg)":     -np.array([r["mean_dist"] for r in rows]),
        "conf_spread":          stored["conf_spread"],
        "entropy_mean":         stored["entropy_mean"],
        "margin_mean (neg)":   -stored["margin_mean"],
        "n_masked":             stored["n_masked"],
    }

    # Rank features by their ability to target expensive steps.
    print("feature correlation")
    for name, f in features.items():
        print(f"{name}: {spearmanr(f, cost).statistic:+.4f}")

    print("\ncapture efficiency")
    head = f"  {'feature':>22}" + "".join(f"{f'@{int(r*100)}%':>22}" for r in GATE_RATES)
    print(head)

    rng = np.random.default_rng(BOOT_SEED)
    out_rows = []
    for name, f in list(features.items()) + [("ORACLE", cost.copy()),
                                             ("random", np.random.default_rng(0)
                                              .standard_normal(len(cost)))]:
        curve = capture_curve(f, cost, gen, GATE_RATES, rng)
        line = f"  {name:>22}"
        for r in GATE_RATES:
            pt, lo, hi = curve[r]
            line += f"{f'{pt:.2f}x ({lo:.2f},{hi:.2f})':>22}"
            out_rows.append([name, r, f"{pt:.4f}", f"{lo:.4f}", f"{hi:.4f}"])
        print(line)

    out = RESULTS / "attention_vs_penalty.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["feature", "gate_rate", "capture_efficiency", "boot_lo", "boot_hi"])
        w.writerows(out_rows)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
