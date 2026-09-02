from __future__ import annotations
import torch
from src.dlm_interp.samplers import score_positions

COMMIT, DEFER = "commit", "defer"

def draw_noise(n_steps, length, seed):
    """Pre-drawn randomness indexed by (step, position).

    reveal: ancestral coin flips. token: inverse-CDF uniforms."""
    g = torch.Generator("cpu").manual_seed(seed)
    return {
        "reveal": torch.rand(n_steps, length, generator=g),
        "token": torch.rand(n_steps, length, generator=g),
    }

def sample_with(probs, u):
    """Inverse-transform sampling: deterministic given u."""
    cdf = probs.cumsum(-1)
    return (cdf < u.unsqueeze(-1)).sum(-1).clamp(max=probs.shape[-1] - 1)

@torch.no_grad()
def rollout(forward_fn, x, timesteps, mask_id, noise, rule, schedule,
            start=0, stop=None, force=None, token_rule="sample"):
    result = x.clone()
    transitions = list(zip(timesteps[:-1], timesteps[1:]))
    stop = len(transitions) if stop is None else stop

    for step in range(start, stop):
        t, s = transitions[step]
        masked = result == mask_id
        if not masked.any():
            break
        positions = masked.nonzero(as_tuple=False).squeeze(-1)
        n = int(positions.numel())

        logits_all, _ = forward_fn(result.unsqueeze(0), float(t))
        logits = logits_all[positions].float()
        logits[:, mask_id] = float("-inf")
        confidence, _, _, _, _ = score_positions(logits)

        if rule == "ancestral":
            p = (float(t) - float(s)) / float(t)
            chosen = (noise["reveal"][step, positions] < p).nonzero(as_tuple=False).squeeze(-1)
        else:
            k = min(schedule[step], n)
            if k <= 0:
                continue
            order = confidence.argsort(descending=True)
            chosen = order[:k]

        if force is not None and step == start:
            target, action = force
            local = (positions == target).nonzero(as_tuple=False).squeeze(-1)
            if local.numel():
                local = int(local)
                inside = bool((chosen == local).any())
                if action == COMMIT and not inside:
                    chosen = torch.cat([chosen[:-1], torch.tensor([local])]) if chosen.numel() \
                        else torch.tensor([local])
                elif action == DEFER and inside:
                    keep = chosen[chosen != local]
                    spare = [j for j in range(n) if j not in set(chosen.tolist())]
                    chosen = torch.cat([keep, torch.tensor(spare[:1])]) if spare else keep

        if chosen.numel():
            probs = logits[chosen].softmax(-1)
            if token_rule == "argmax":
                tokens = probs.argmax(-1)
            else:
                tokens = sample_with(probs, noise["token"][step, positions[chosen]])
            result[positions[chosen]] = tokens

    return result

@torch.no_grad()
def sequence_logprob(model_logits_fn, ids):
    """Q: total log-probability of the finished sequence under one full pass."""
    logits, _ = model_logits_fn(ids.unsqueeze(0), 0.0)
    logprobs = logits.float().log_softmax(-1)
    return float(logprobs.gather(-1, ids.unsqueeze(-1)).sum())

def commit_value(forward_fn, x_state, position, timesteps, mask_id, rule, schedule,
                 start, n_rollouts, base_seed):
    """Paired estimate of V for one position at one state."""
    diffs = []
    for r in range(n_rollouts):
        noise = draw_noise(len(timesteps) - 1, int(x_state.numel()), base_seed + r)
        a = rollout(forward_fn, x_state, timesteps, mask_id, noise, rule, schedule,
                    start=start, force=(position, COMMIT))
        b = rollout(forward_fn, x_state, timesteps, mask_id, noise, rule, schedule,
                    start=start, force=(position, DEFER))
        diffs.append(sequence_logprob(forward_fn, a) - sequence_logprob(forward_fn, b))
    values = torch.tensor(diffs)
    return float(values.mean()), float(values.std()) if len(diffs) > 1 else 0.0