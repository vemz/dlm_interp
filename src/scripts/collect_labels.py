from __future__ import annotations

import numpy as np
import torch

from src.dlm_interp.load import load_model, nano_forward_fn
from src.dlm_interp.samplers import HiddenCapture, score_positions

VAL_BIN = "data/tinystories/val.bin"
OUT = "labels_gtmargin.pt"

N_WINDOWS = 200
T_VALUES = (0.2, 0.5, 0.8)
LAYERS = (1, 3, 5)
SEED = 0

FIELDS = ("gt_margin", "p_true", "confidence", "entropy", "margin", "position", "t", "window")


def sample_windows(path, n_windows, seq_len, rng):
    data = np.memmap(path, dtype=np.uint16, mode="r")
    starts = rng.integers(0, len(data) - seq_len, size=n_windows)
    return torch.from_numpy(np.stack([data[s : s + seq_len].astype(np.int64) for s in starts]))


@torch.no_grad()
def collect(forward_fn, windows, mask_id, t_values, generator):
    out = {key: [] for key in FIELDS}
    hidden_out: dict[int, list[torch.Tensor]] = {}

    for w_idx, clean in enumerate(windows):
        for t in t_values:
            keep = torch.rand(clean.shape, generator=generator) >= t
            corrupted = torch.where(keep, clean, torch.full_like(clean, mask_id))
            positions = (~keep).nonzero(as_tuple=False).squeeze(-1)
            if positions.numel() < 2:
                continue

            logits_all, hidden = forward_fn(corrupted.unsqueeze(0), float(t))
            logits = logits_all[positions].float()
            logits[:, mask_id] = float("-inf")

            confidence, entropy, margin, _, _ = score_positions(logits)

            probs = logits.softmax(-1)
            truth = clean[positions]
            p_true = probs.gather(-1, truth.unsqueeze(-1)).squeeze(-1)
            p_others = probs.scatter(-1, truth.unsqueeze(-1), 0.0)

            n = positions.numel()
            values = {
                "gt_margin": p_true - p_others.max(-1).values,
                "p_true": p_true,
                "confidence": confidence,
                "entropy": entropy,
                "margin": margin,
                "position": positions,
                "t": torch.full((n,), t),
                "window": torch.full((n,), w_idx, dtype=torch.long),
            }
            for key, value in values.items():
                out[key].append(value)

            for layer, h in hidden.items():
                hidden_out.setdefault(layer, []).append(h[positions].half())

        if (w_idx + 1) % 25 == 0:
            print(f"  {w_idx + 1}/{len(windows)} windows")

    record = {key: torch.cat(value) for key, value in out.items()}
    record["hidden"] = {layer: torch.cat(value) for layer, value in hidden_out.items()}
    return record


def report(record, t_values):
    def stats(mask):
        ce = -record["p_true"][mask].clamp_min(1e-12).log()
        acc = (record["gt_margin"][mask] > 0).float().mean()
        conf = record["confidence"][mask].mean()
        return int(mask.sum()), float(ce.mean()), float(acc), float(conf)

    print(f"\n{'':>6} {'n':>7} {'CE':>7} {'acc':>7} {'conf':>7}")
    for t in t_values:
        n, ce, acc, conf = stats(record["t"] == t)
        print(f"t={t:<4} {n:7d} {ce:7.3f} {acc:7.3f} {conf:7.4f}")

    n, ce, acc, conf = stats(torch.ones_like(record["t"], dtype=torch.bool))
    print(f"{'all':>6} {n:7d} {ce:7.3f} {acc:7.3f} {conf:7.4f}")
    print("\nreference: training NELBO 5.52, uniform 10.82")

    for layer, h in record["hidden"].items():
        print(f"hidden[{layer}]: {tuple(h.shape)} {h.dtype}")


def main():
    model, cfg = load_model()
    mask_id, seq_len = int(cfg["mask_id"]), int(cfg["seq_len"])

    rng = np.random.default_rng(SEED)
    generator = torch.Generator("cpu").manual_seed(SEED)
    windows = sample_windows(VAL_BIN, N_WINDOWS, seq_len, rng)

    capture = HiddenCapture({i: model.blocks[i] for i in LAYERS})
    with capture:
        forward_fn = capture.wrap(nano_forward_fn(model))
        record = collect(forward_fn, windows, mask_id, T_VALUES, generator)

    report(record, T_VALUES)
    torch.save(record, OUT)
    print(f"\nsaved to {OUT}")


if __name__ == "__main__":
    main()