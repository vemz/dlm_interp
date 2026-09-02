from __future__ import annotations
import numpy as np
import torch
from load import load_model, nano_forward_fn
from samplers import HiddenCapture, score_positions

VAL_BIN = "data/tinystories/val.bin"
OUT = "labels_waitgain.pt"

N_WINDOWS = 200
T_VALUES = (0.8, 0.5, 0.2)
REVEAL_FRACTION = 0.5        
LAYERS = (1, 3, 5)
SEED = 0
FIELDS = ("wait_gain", "margin_now", "margin_later", "p_true_now", "p_true_later",
          "confidence", "entropy", "margin", "position", "t", "window")

def sample_windows(path, n_windows, seq_len, rng):
    data = np.memmap(path, dtype=np.uint16, mode="r")
    starts = rng.integers(0, len(data) - seq_len, size=n_windows)
    return torch.from_numpy(np.stack([data[s : s + seq_len].astype(np.int64) for s in starts]))

def gt_margin(logits, truth):
    probs = logits.softmax(-1)
    p_true = probs.gather(-1, truth.unsqueeze(-1)).squeeze(-1)
    p_others = probs.scatter(-1, truth.unsqueeze(-1), 0.0)
    return p_true - p_others.max(-1).values, p_true

@torch.no_grad()
def collect(forward_fn, windows, mask_id, t_values, generator):
    out = {key: [] for key in FIELDS}
    hidden_out: dict[int, list[torch.Tensor]] = {}

    for w_idx, clean in enumerate(windows):
        for t in t_values:
            masked = torch.rand(clean.shape, generator=generator) < t
            if int(masked.sum()) < 4:
                continue
            x_now = torch.where(masked, torch.full_like(clean, mask_id), clean)

            reveal = masked & (torch.rand(clean.shape, generator=generator) < REVEAL_FRACTION)
            still_masked = masked & ~reveal
            if int(still_masked.sum()) < 2:
                continue
            x_later = torch.where(still_masked, torch.full_like(clean, mask_id), clean)

            positions = still_masked.nonzero(as_tuple=False).squeeze(-1)

            logits_now, hidden = forward_fn(x_now.unsqueeze(0), float(t))
            logits_now = logits_now[positions].float()
            logits_now[:, mask_id] = float("-inf")

            logits_later, _ = forward_fn(x_later.unsqueeze(0), float(t))
            logits_later = logits_later[positions].float()
            logits_later[:, mask_id] = float("-inf")

            truth = clean[positions]
            m_now, p_now = gt_margin(logits_now, truth)
            m_later, p_later = gt_margin(logits_later, truth)
            confidence, entropy, margin, _, _ = score_positions(logits_now)

            n = positions.numel()
            values = {
                "wait_gain": m_later - m_now,
                "margin_now": m_now,
                "margin_later": m_later,
                "p_true_now": p_now,
                "p_true_later": p_later,
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
        return (
            int(mask.sum()),
            float(record["margin_now"][mask].mean()),
            float(record["margin_later"][mask].mean()),
            float(record["wait_gain"][mask].mean()),
            float(record["wait_gain"][mask].std()),
            float((record["wait_gain"][mask] > 0).float().mean()),
        )

    print(f"\n{'':>6} {'n':>7} {'now':>8} {'later':>8} {'gain':>8} {'sd':>8} {'>0':>6}")
    for t in t_values:
        n, now, later, gain, sd, pos = stats(record["t"] == t)
        print(f"t={t:<4} {n:7d} {now:8.4f} {later:8.4f} {gain:8.4f} {sd:8.4f} {pos:6.3f}")
    n, now, later, gain, sd, pos = stats(torch.ones_like(record["t"], dtype=torch.bool))
    print(f"{'all':>6} {n:7d} {now:8.4f} {later:8.4f} {gain:8.4f} {sd:8.4f} {pos:6.3f}")

    gain = record["wait_gain"].numpy()
    conf = record["confidence"].numpy()
    corr = np.corrcoef(conf, gain)[0, 1] if gain.std() > 0 else float("nan")
    print(f"\ncorr(confidence, wait_gain) = {corr:+.4f}")

def main():
    model, cfg = load_model()
    mask_id, seq_len = int(cfg["mask_id"]), int(cfg["seq_len"])

    rng = np.random.default_rng(SEED)
    generator = torch.Generator("cpu").manual_seed(SEED)
    windows = sample_windows(VAL_BIN, N_WINDOWS, seq_len, rng)

    modules = {i: model.blocks[i] for i in LAYERS}
    modules[len(model.blocks)] = model.ln_f        
    capture = HiddenCapture(modules)
    with capture:
        forward_fn = capture.wrap(nano_forward_fn(model))
        record = collect(forward_fn, windows, mask_id, T_VALUES, generator)

    report(record, T_VALUES)
    for layer, h in record["hidden"].items():
        print(f"hidden[{layer}]: {tuple(h.shape)} {h.dtype}")

    torch.save(record, OUT)
    print(f"\nsaved to {OUT}")

if __name__ == "__main__":
    main()