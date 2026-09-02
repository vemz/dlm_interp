"""How far from the end can a commit-value probe sit before the branches
decouple?

Same diagnostic as diagnose_value.py, run at several intervention steps. The
question is whether pairing ever works, and at what horizon the labels carry
signal.
"""

from __future__ import annotations

import numpy as np
import torch

from src.dlm_interp.commit_value import COMMIT, DEFER, draw_noise, rollout
from src.dlm_interp.load import load_model, nano_forward_fn
from src.dlm_interp.samplers import uniform_schedule

N_STEPS = 64
PROBE_STEPS = (60, 56, 48, 32)     # 4, 8, 16, 32 transitions from the end
N_POSITIONS = 6
N_ROLLOUTS = 6
RULE = "top_k_confidence"
SEED = 0


@torch.no_grad()
def q_masked(forward_fn, ids, affected):
    logits, _ = forward_fn(ids.unsqueeze(0), 0.0)
    per_token = logits.float().log_softmax(-1).gather(-1, ids.unsqueeze(-1)).squeeze(-1)
    return float(per_token[affected].sum())


def main():
    model, cfg = load_model()
    mask_id, seq_len = int(cfg["mask_id"]), int(cfg["seq_len"])
    forward_fn = nano_forward_fn(model)
    timesteps = torch.linspace(1.0, 0.0, N_STEPS + 1).tolist()
    schedule = uniform_schedule(seq_len, N_STEPS)
    base = draw_noise(N_STEPS, seq_len, SEED)

    print(f"{'probe':>6} {'left':>5} {'masked':>7} {'agree':>7} {'paired sd':>10} "
          f"{'indep sd':>9} {'pairing':>8} {'signal':>8} {'SNR':>6}")
    print("-" * 74)

    for probe in PROBE_STEPS:
        x = rollout(forward_fn, torch.full((seq_len,), mask_id, dtype=torch.long),
                    timesteps, mask_id, base, RULE, schedule, stop=probe)
        masked = (x == mask_id).nonzero(as_tuple=False).squeeze(-1)
        if masked.numel() < N_POSITIONS + 2:
            print(f"{probe:6d}  too few masked positions")
            continue

        g = torch.Generator("cpu").manual_seed(SEED)
        targets = masked[torch.randperm(masked.numel(), generator=g)[:N_POSITIONS]].tolist()

        paired_sd, indep_sd, means, agreement = [], [], [], []
        for target in targets:
            diffs, indep = [], []
            for r in range(N_ROLLOUTS):
                noise = draw_noise(N_STEPS, seq_len, 1000 + r)
                other = draw_noise(N_STEPS, seq_len, 5000 + r)
                a = rollout(forward_fn, x, timesteps, mask_id, noise, RULE, schedule,
                            start=probe, force=(target, COMMIT))
                b = rollout(forward_fn, x, timesteps, mask_id, noise, RULE, schedule,
                            start=probe, force=(target, DEFER))
                bi = rollout(forward_fn, x, timesteps, mask_id, other, RULE, schedule,
                             start=probe, force=(target, DEFER))
                qa, qb, qi = (q_masked(forward_fn, s, masked) for s in (a, b, bi))
                diffs.append(qa - qb)
                indep.append(qa - qi)
                if r == 0:
                    agreement.append(float((a == b).float().mean()))
            paired_sd.append(float(np.std(diffs)))
            indep_sd.append(float(np.std(indep)))
            means.append(float(np.mean(diffs)))

        noise = float(np.mean(paired_sd))
        indep_noise = float(np.mean(indep_sd))
        signal = float(np.std(means))
        se = noise / np.sqrt(N_ROLLOUTS)
        print(f"{probe:6d} {N_STEPS - probe:5d} {masked.numel():7d} "
              f"{np.mean(agreement):6.0%} {noise:10.2f} {indep_noise:9.2f} "
              f"{indep_noise / max(noise, 1e-9):7.1f}x {signal:8.2f} "
              f"{signal / max(se, 1e-9):6.2f}")

    print("\npairing > 1.5x means common random numbers are still buying something.")
    print("SNR > 2 means the labels are usable at this horizon.")


if __name__ == "__main__":
    main()