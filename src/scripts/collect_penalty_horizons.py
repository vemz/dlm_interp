from __future__ import annotations
import sys
from collections import deque
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from src.dlm_interp.load import load_model, nano_forward_fn
from src.dlm_interp.paths import RUNS
from src.dlm_interp.samplers import HiddenCapture, score_positions

OUT = RUNS / "labels_penalty_horizon.pt"

N_GENERATIONS = 200
K = 4
PROBE_EVERY = 2
HORIZONS = (0, 1, 2, 4, 8, 16)
LAYERS = (1, 3, 5)
SEED = 0

SCALARS = ("penalty", "marginal", "chain", "order_gap",
           "conf_mean", "conf_min", "conf_max", "conf_spread",
           "entropy_mean", "margin_mean", "step", "n_masked", "generation")

@torch.no_grad()
def chain_sum(forward_fn, x, positions, tokens, mask_id, order):
    """Sum of log p(x_j | c, x_{S<j}) for one reveal order."""
    state = x.clone()
    total = 0.0
    for idx in order:
        logits, _ = forward_fn(state.unsqueeze(0), 0.0)
        total += float(logits[positions[idx]].float().log_softmax(-1)[tokens[idx]])
        state[positions[idx]] = tokens[idx]
    return total

@torch.no_grad()
def step_forward(forward_fn, x, mask_id, k, generator):
    masked = (x == mask_id).nonzero(as_tuple=False).squeeze(-1)
    if masked.numel() <= k:
        return None

    logits_all, hidden = forward_fn(x.unsqueeze(0), 0.0)
    logits = logits_all[masked].float()
    logits[:, mask_id] = float("-inf")
    confidence, entropy, margin, _, _ = score_positions(logits)

    chosen = confidence.topk(k).indices
    positions = masked[chosen]
    logprobs = logits[chosen].log_softmax(-1)
    tokens = torch.multinomial(logprobs.exp(), 1, generator=generator).squeeze(-1)

    return {
        "positions": positions,
        "tokens": tokens,
        "hidden": hidden,
        "marginal": float(logprobs.gather(-1, tokens.unsqueeze(-1)).sum()),
        "n_masked": int(masked.numel()),
        "conf_mean": float(confidence[chosen].mean()),
        "conf_min": float(confidence[chosen].min()),
        "conf_max": float(confidence[chosen].max()),
        "conf_spread": float(confidence[chosen].max() - confidence[chosen].min()),
        "entropy_mean": float(entropy[chosen].mean()),
        "margin_mean": float(margin[chosen].mean()),
    }

def pool_at(hidden, positions, still_masked):
    keep = positions[torch.isin(positions, still_masked)]
    if keep.numel() == 0:
        return None
    return {layer: h[keep].float().mean(0).half() for layer, h in hidden.items()}

@torch.no_grad()
def run_generation(forward_fn, seq_len, mask_id, k, generator, gen_id):
    x = torch.full((seq_len,), mask_id, dtype=torch.long)
    history = deque(maxlen=max(HORIZONS) + 1)      # (hidden, still_masked)
    rows = []

    for step in range(seq_len // k):
        if not (x == mask_id).any():
            break
        info = step_forward(forward_fn, x, mask_id, k, generator)
        if info is None:
            break

        history.append((info["hidden"], (x == mask_id).nonzero(as_tuple=False).squeeze(-1)))

        if step % PROBE_EVERY == 0 and len(history) > max(HORIZONS):
            orders = [list(range(k)), torch.randperm(k, generator=generator).tolist()]
            chains = [chain_sum(forward_fn, x, info["positions"], info["tokens"], mask_id, o)
                      for o in orders]

            pooled, usable = {}, True
            for h in HORIZONS:
                past_hidden, past_masked = history[-1 - h]
                at_h = pool_at(past_hidden, info["positions"], past_masked)
                if at_h is None:
                    usable = False
                    break
                pooled[h] = at_h

            if usable:
                row = {key: info[key] for key in
                       ("marginal", "n_masked", "conf_mean", "conf_min", "conf_max",
                        "conf_spread", "entropy_mean", "margin_mean")}
                row.update({
                    "chain": float(np.mean(chains)),
                    "penalty": float(np.mean(chains)) - info["marginal"],
                    "order_gap": abs(chains[0] - chains[1]),
                    "step": step,
                    "generation": gen_id,
                })
                rows.append((row, pooled))

        x[info["positions"]] = info["tokens"]

    return rows

def main():
    if OUT.exists():
        print(f"{OUT} exists; delete it to recollect, or change OUT")
        return
    model, cfg = load_model()
    mask_id, seq_len = int(cfg["mask_id"]), int(cfg["seq_len"])
    generator = torch.Generator("cpu").manual_seed(SEED)

    scalars = {key: [] for key in SCALARS}
    hidden_out = {h: {} for h in HORIZONS}

    capture = HiddenCapture({i: model.blocks[i] for i in LAYERS})
    with capture:
        forward_fn = capture.wrap(nano_forward_fn(model))
        for gen in range(N_GENERATIONS):
            for row, pooled in run_generation(forward_fn, seq_len, mask_id, K, generator, gen):
                for key in SCALARS:
                    scalars[key].append(row[key])
                for h, layers in pooled.items():
                    for layer, vector in layers.items():
                        hidden_out[h].setdefault(layer, []).append(vector.reshape(1, -1))
            if (gen + 1) % 20 == 0:
                print(f"  {gen + 1}/{N_GENERATIONS} generations, "
                      f"{len(scalars['penalty'])} rows")

    record = {key: torch.tensor(value, dtype=torch.float32) for key, value in scalars.items()}
    record["hidden"] = {
        h: {layer: torch.cat(chunks) for layer, chunks in layers.items()}
        for h, layers in hidden_out.items()
    }
    record["horizons"] = list(HORIZONS)

    first = record["generation"] == 0
    second = record["generation"] == 1
    n = min(int(first.sum()), int(second.sum()))
    assert n > 0 and not torch.allclose(record["marginal"][first][:n],
                                        record["marginal"][second][:n]), (
        "generations are identical — the sampler has no source of variation, so "
        "grouped splits separate nothing and any probe score is memorisation"
    )

    penalty = record["penalty"]
    print(f"\n{len(penalty)} probed steps over {N_GENERATIONS} generations, k={K}")
    print(f"penalty: mean {penalty.mean():+.3f}, sd {penalty.std():.3f}, "
          f"share negative {float((penalty < 0).float().mean()):.3f}")
    print(f"order_gap: mean {record['order_gap'].mean():.3f} "
          f"(vs |penalty| {penalty.abs().mean():.3f})")

    across = record["marginal"][first][:n] - record["marginal"][second][:n]
    print(f"generations differ: mean |Δmarginal| between gen 0 and 1 = "
          f"{float(across.abs().mean()):.3f}")

    for h in HORIZONS:
        shape = next(iter(record["hidden"][h].values())).shape
        print(f"  h={h}: {tuple(shape)} per layer")

    torch.save(record, OUT)
    print(f"\nsaved to {OUT}")


if __name__ == "__main__":
    main()