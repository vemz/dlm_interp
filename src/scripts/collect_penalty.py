from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from src.dlm_interp.load import load_model, nano_forward_fn
from src.dlm_interp.samplers import HiddenCapture, score_positions, uniform_schedule
from src.dlm_interp.paths import RUNS

VAL_BIN = "data/tinystories/val.bin"
OUT = RUNS / "labels_penalty.pt"

N_GENERATIONS = 40
K = 4                 
PROBE_EVERY = 4        
LAYERS = (1, 3, 5)
SEED = 0

FIELDS = ("penalty", "marginal", "chain", "order_gap",
          "conf_mean", "conf_min", "conf_max", "conf_spread",
          "entropy_mean", "margin_mean", "step", "n_masked", "generation")


@torch.no_grad()
def chain_penalty(forward_fn, x, positions, tokens, mask_id, order):
    """Sum of log p(x_j | c, x_{S<j}) for one reveal order."""
    state = x.clone()
    total = 0.0
    for idx in order:
        logits, _ = forward_fn(state.unsqueeze(0), 0.0)
        logprobs = logits[positions[idx]].float().log_softmax(-1)
        total += float(logprobs[tokens[idx]])
        state[positions[idx]] = tokens[idx]
    return total


@torch.no_grad()
def probe_step(forward_fn, x, mask_id, k, generator):
    """One probed step: choose S by confidence, measure the penalty of
    committing it together, return the features measured before the commit."""
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
    tokens = logprobs.argmax(-1)
    marginal = float(logprobs.gather(-1, tokens.unsqueeze(-1)).sum())

    orders = [list(range(k)), torch.randperm(k, generator=generator).tolist()]
    chains = [chain_penalty(forward_fn, x, positions, tokens, mask_id, o) for o in orders]

    features = {
        "penalty": float(np.mean(chains)) - marginal,
        "order_gap": abs(chains[0] - chains[1]),
        "conf_mean": float(confidence[chosen].mean()),
        "conf_min": float(confidence[chosen].min()),
        "conf_max": float(confidence[chosen].max()),
        "conf_spread": float(confidence[chosen].max() - confidence[chosen].min()),
        "entropy_mean": float(entropy[chosen].mean()),
        "margin_mean": float(margin[chosen].mean()),
        "n_masked": int(masked.numel()),
        "marginal": marginal,
        "chain": float(np.mean(chains)),
    }
    pooled = {layer: h[positions].float().mean(0).half() for layer, h in hidden.items()}
    return features, pooled, positions, tokens


@torch.no_grad()
def run_generation(forward_fn, seq_len, mask_id, k, generator):
    """Decode with confidence top-k, probing every PROBE_EVERY steps."""
    n_steps = seq_len // k
    x = torch.full((seq_len,), mask_id, dtype=torch.long)
    rows, pooled_rows = [], []

    for step in range(n_steps):
        if not (x == mask_id).any():
            break
        probing = step % PROBE_EVERY == 0
        result = probe_step(forward_fn, x, mask_id, k, generator)
        if result is None:
            break
        features, pooled, positions, tokens = result
        if probing:
            features["step"] = step
            rows.append(features)
            pooled_rows.append(pooled)
        x[positions] = tokens

    return rows, pooled_rows


def main():
    model, cfg = load_model()
    mask_id, seq_len = int(cfg["mask_id"]), int(cfg["seq_len"])
    generator = torch.Generator("cpu").manual_seed(SEED)

    out = {key: [] for key in FIELDS}
    hidden_out: dict[int, list[torch.Tensor]] = {}

    capture = HiddenCapture({i: model.blocks[i] for i in LAYERS})
    with capture:
        forward_fn = capture.wrap(nano_forward_fn(model))
        for gen in range(N_GENERATIONS):
            rows, pooled_rows = run_generation(forward_fn, seq_len, mask_id, K, generator)
            for features, pooled in zip(rows, pooled_rows):
                features["generation"] = gen
                for key in FIELDS:
                    out[key].append(features[key])
                for layer, h in pooled.items():
                    hidden_out.setdefault(layer, []).append(h.reshape(1, -1))
            if (gen + 1) % 10 == 0:
                print(f"  {gen + 1}/{N_GENERATIONS} generations, {len(out['penalty'])} rows")

    record = {key: torch.tensor(value, dtype=torch.float32) for key, value in out.items()}
    record["hidden"] = {layer: torch.cat(value) for layer, value in hidden_out.items()}
    report(record)
    torch.save(record, OUT)
    print(f"\nsaved to {OUT}")


def report(record):
    penalty, gap = record["penalty"], record["order_gap"]
    print(f"\n{len(penalty)} probed steps over {int(record['generation'].max()) + 1} generations, k={K}")
    print(f"penalty: mean {penalty.mean():+.3f}, sd {penalty.std():.3f}, "
          f"share negative {float((penalty < 0).float().mean()):.3f}")
    print(f"order_gap: mean {gap.mean():.3f} "
          f"(vs |penalty| {penalty.abs().mean():.3f} — larger means order dominates)")
    print(f"corr(conf_mean, penalty) = "
          f"{np.corrcoef(record['conf_mean'], penalty)[0, 1]:+.3f}")

    step = record["step"]
    thirds = torch.quantile(step, torch.tensor([1 / 3, 2 / 3]))
    for name, mask in (("early", step <= thirds[0]),
                       ("middle", (step > thirds[0]) & (step <= thirds[1])),
                       ("late", step > thirds[1])):
        print(f"  {name:>7}: n={int(mask.sum()):4d}  penalty {penalty[mask].mean():+.3f}")


if __name__ == "__main__":
    main()