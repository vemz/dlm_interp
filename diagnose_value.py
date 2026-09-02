"""Why is the commit-value estimator so noisy?

Two questions, answered on the real model:
  1. Is the pairing actually working, or have the branches fully decoupled?
  2. Which definition of Q gives the best signal-to-noise?

Q variants, all on the same rollouts:
  full      every position (what we used, sd 64.8)
  masked    only positions still masked at the intervention — the ones the
            decision can possibly affect
  window    a +/- W band around the intervened position
  target    the intervened position alone
"""

from __future__ import annotations

import numpy as np
import torch

from src.dlm_interp.commit_value import COMMIT, DEFER, draw_noise, rollout
from src.dlm_interp.load import load_model, nano_forward_fn
from src.dlm_interp.samplers import uniform_schedule

N_STEPS = 64
PROBE_STEP = 32
N_POSITIONS = 8
N_ROLLOUTS = 6
WINDOW = 16
RULE = "top_k_confidence"
SEED = 0


@torch.no_grad()
def q_variants(forward_fn, ids, target, affected):
    """Log-likelihood of the finished sequence under several supports."""
    logits, _ = forward_fn(ids.unsqueeze(0), 0.0)
    per_token = logits.float().log_softmax(-1).gather(-1, ids.unsqueeze(-1)).squeeze(-1)
    lo, hi = max(0, target - WINDOW), min(len(ids), target + WINDOW + 1)
    return {
        "full": float(per_token.sum()),
        "masked": float(per_token[affected].sum()),
        "window": float(per_token[lo:hi].sum()),
        "target": float(per_token[target]),
    }


def main():
    model, cfg = load_model()
    mask_id, seq_len = int(cfg["mask_id"]), int(cfg["seq_len"])
    forward_fn = nano_forward_fn(model)
    timesteps = torch.linspace(1.0, 0.0, N_STEPS + 1).tolist()
    schedule = uniform_schedule(seq_len, N_STEPS)

    base = draw_noise(N_STEPS, seq_len, SEED)
    x = rollout(forward_fn, torch.full((seq_len,), mask_id, dtype=torch.long),
                timesteps, mask_id, base, RULE, schedule, stop=PROBE_STEP)
    masked = (x == mask_id).nonzero(as_tuple=False).squeeze(-1)
    print(f"state at step {PROBE_STEP}: {masked.numel()} positions still masked\n")

    g = torch.Generator("cpu").manual_seed(SEED)
    targets = masked[torch.randperm(masked.numel(), generator=g)[:N_POSITIONS]].tolist()

    names = ("full", "masked", "window", "target")
    paired = {name: [] for name in names}
    independent = {name: [] for name in names}
    per_position = {name: [] for name in names}

    for target in targets:
        diffs = {name: [] for name in names}
        indep = {name: [] for name in names}
        for r in range(N_ROLLOUTS):
            noise = draw_noise(N_STEPS, seq_len, 1000 + r)
            other = draw_noise(N_STEPS, seq_len, 5000 + r)
            a = rollout(forward_fn, x, timesteps, mask_id, noise, RULE, schedule,
                        start=PROBE_STEP, force=(target, COMMIT))
            b = rollout(forward_fn, x, timesteps, mask_id, noise, RULE, schedule,
                        start=PROBE_STEP, force=(target, DEFER))
            b_ind = rollout(forward_fn, x, timesteps, mask_id, other, RULE, schedule,
                            start=PROBE_STEP, force=(target, DEFER))

            qa, qb, qi = (q_variants(forward_fn, seq, target, masked) for seq in (a, b, b_ind))
            for name in names:
                diffs[name].append(qa[name] - qb[name])
                indep[name].append(qa[name] - qi[name])

            if r == 0:
                agree = int((a == b).sum())
                print(f"  pos {target:3d}: branches agree on {agree}/{seq_len} tokens "
                      f"({agree / seq_len:.0%})")

        for name in names:
            paired[name].append(float(np.std(diffs[name])))
            independent[name].append(float(np.std(indep[name])))
            per_position[name].append(float(np.mean(diffs[name])))

    print(f"\n{'Q variant':>10} {'rollout sd':>11} {'indep sd':>10} {'pairing':>8} "
          f"{'sd across pos':>14} {'SNR':>7}")
    print("-" * 64)
    for name in names:
        noise = float(np.mean(paired[name]))
        indep_noise = float(np.mean(independent[name]))
        signal = float(np.std(per_position[name]))
        se = noise / np.sqrt(N_ROLLOUTS)
        print(f"{name:>10} {noise:11.2f} {indep_noise:10.2f} "
              f"{indep_noise / max(noise, 1e-9):7.1f}x {signal:14.2f} {signal / max(se, 1e-9):7.2f}")

    print("\nSNR = sd across positions / standard error of one estimate.")
    print("Above ~2 the labels carry usable signal; below ~1 they are mostly noise.")


if __name__ == "__main__":
    main()