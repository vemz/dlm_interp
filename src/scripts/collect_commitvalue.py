from __future__ import annotations
import time
import numpy as np
import torch

from src.dlm_interp.commit_value import commit_value, draw_noise, rollout
from src.dlm_interp.load import load_model, nano_forward_fn
from src.dlm_interp.samplers import HiddenCapture, score_positions, uniform_schedule

VAL_BIN = "data/tinystories/val.bin"
OUT = "labels_commitvalue.pt"

N_WINDOWS = 20
N_STEPS = 64               
PROBE_FRACTIONS = (0.25, 0.5, 0.75)
POSITIONS_PER_STATE = 4
N_ROLLOUTS = 4
RULE = "top_k_confidence"
LAYERS = (1, 3, 5)
SEED = 0

FIELDS = ("commit_value", "value_sd", "confidence", "entropy", "margin",
          "position", "step", "n_masked", "window")

def sample_windows(path, n_windows, seq_len, rng):
    data = np.memmap(path, dtype=np.uint16, mode="r")
    starts = rng.integers(0, len(data) - seq_len, size=n_windows)
    return torch.from_numpy(np.stack([data[s : s + seq_len].astype(np.int64) for s in starts]))

@torch.no_grad()
def collect(forward_fn, windows, mask_id, seq_len):
    timesteps = torch.linspace(1.0, 0.0, N_STEPS + 1).tolist()
    schedule = uniform_schedule(seq_len, N_STEPS)
    out = {key: [] for key in FIELDS}
    hidden_out: dict[int, list[torch.Tensor]] = {}
    started = time.time()

    for w_idx in range(len(windows)):
        base_noise = draw_noise(N_STEPS, seq_len, SEED + 1000 * w_idx)
        x = torch.full((seq_len,), mask_id, dtype=torch.long)

        prev, x = 0, torch.full((seq_len,), mask_id, dtype=torch.long)
        for fraction in PROBE_FRACTIONS:
            step = int(fraction * N_STEPS)
            x = rollout(forward_fn, x, timesteps, mask_id, base_noise, RULE,
                        schedule, start=prev, stop=step)
            prev = step

            masked = (x == mask_id).nonzero(as_tuple=False).squeeze(-1)
            if masked.numel() < POSITIONS_PER_STATE + 2:
                continue

            logits_all, hidden = forward_fn(x.unsqueeze(0), float(timesteps[step]))
            logits = logits_all[masked].float()
            logits[:, mask_id] = float("-inf")
            confidence, entropy, margin, _, _ = score_positions(logits)

            g = torch.Generator("cpu").manual_seed(SEED + w_idx * 97 + step)
            picks = torch.randperm(masked.numel(), generator=g)[:POSITIONS_PER_STATE]

            for pick in picks.tolist():
                pos = int(masked[pick])
                mean, sd = commit_value(
                    forward_fn, x, pos, timesteps, mask_id, RULE, schedule,
                    start=step, n_rollouts=N_ROLLOUTS,
                    base_seed=SEED + 7919 * w_idx + 13 * step + pick,
                )
                values = {
                    "commit_value": torch.tensor([mean]),
                    "value_sd": torch.tensor([sd]),
                    "confidence": confidence[pick].reshape(1),
                    "entropy": entropy[pick].reshape(1),
                    "margin": margin[pick].reshape(1),
                    "position": torch.tensor([pos]),
                    "step": torch.tensor([step]),
                    "n_masked": torch.tensor([int(masked.numel())]),
                    "window": torch.tensor([w_idx]),
                }
                for key, value in values.items():
                    out[key].append(value)
                for layer, h in hidden.items():
                    hidden_out.setdefault(layer, []).append(h[pos].reshape(1, -1).half())

        done = w_idx + 1
        rate = (time.time() - started) / done
        print(f"  {done}/{len(windows)} windows  ({rate:.1f}s each, "
              f"~{rate * (len(windows) - done) / 60:.0f} min left)")

    record = {key: torch.cat(value) for key, value in out.items()}
    record["hidden"] = {layer: torch.cat(value) for layer, value in hidden_out.items()}
    return record

def report(record):
    value = record["commit_value"]
    print(f"\n{len(value)} probed decisions over {int(record['window'].max()) + 1} windows")
    print(f"commit_value: mean {value.mean():+.3f}, sd {value.std():.3f}, "
          f"share positive {float((value > 0).float().mean()):.3f}")
    print(f"within-position rollout sd: {record['value_sd'].mean():.3f} "
          f"(paired estimator noise)")

    conf = record["confidence"].numpy()
    val = value.numpy()
    if val.std() > 0:
        print(f"corr(confidence, commit_value) = {np.corrcoef(conf, val)[0, 1]:+.3f}")
    for step in sorted(set(record["step"].tolist())):
        m = record["step"] == step
        print(f"  step {step:3d}: n={int(m.sum()):4d}  V {value[m].mean():+.3f} "
              f"± {value[m].std():.3f}")

def main():
    model, cfg = load_model()
    mask_id, seq_len = int(cfg["mask_id"]), int(cfg["seq_len"])

    rng = np.random.default_rng(SEED)
    windows = sample_windows(VAL_BIN, N_WINDOWS, seq_len, rng)

    capture = HiddenCapture({i: model.blocks[i] for i in LAYERS})
    with capture:
        forward_fn = capture.wrap(nano_forward_fn(model))
        record = collect(forward_fn, windows, mask_id, seq_len)

    report(record)
    torch.save(record, OUT)
    print(f"\nsaved to {OUT}")

if __name__ == "__main__":
    main()