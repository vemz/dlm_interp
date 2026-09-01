from __future__ import annotations
import statistics
import torch
from transformers import AutoTokenizer
from load import load_model, nano_forward_fn
from quality import sequence_logprob, structural_token_ids
from samplers import annotate_trace, run_sampler, uniform_schedule

RULES = ("ancestral", "random_fixed_k", "top_k_confidence", "left_to_right")
SEEDS = range(10)
METRICS = ("q_per_token", "q_content_per_token", "frac_structural", "effective_len", "eos_count")

def generate(model, forward_fn, seq_len, mask_id, structural, eos_id, rule, seed):
    timesteps = torch.linspace(1.0, 0.0, seq_len + 1).tolist()
    schedule = None if rule == "ancestral" else uniform_schedule(seq_len, seq_len)

    out, trace = run_sampler(
        forward_fn,
        torch.full((seq_len,), mask_id, dtype=torch.long),
        timesteps,
        mask_id,
        rule=rule,
        schedule=schedule,
        token_rule="sample",
        seed=seed,
        keep_hidden=False,
    )
    annotate_trace(trace, structural, eos_token_id=eos_id)

    content = trace.token_types
    n_content = int(content.sum())
    q, q_content = sequence_logprob(model, out, mask=content)

    return out, {
        "q_per_token": q / seq_len,
        "q_content_per_token": q_content / n_content if n_content else float("nan"),
        "frac_structural": 1 - n_content / seq_len,
        "effective_len": float(trace.effective_len),
        "eos_count": float((out == eos_id).sum()),
    }

def main():
    model, cfg = load_model()
    forward_fn = nano_forward_fn(model)
    tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-33M")

    mask_id, seq_len = int(cfg["mask_id"]), int(cfg["seq_len"])
    structural = structural_token_ids(tokenizer)
    eos_id = tokenizer.eos_token_id

    for rule in RULES:
        stats = {key: [] for key in METRICS}

        for seed in SEEDS:
            out, metrics = generate(
                model, forward_fn, seq_len, mask_id, structural, eos_id, rule, seed
            )
            for key, value in metrics.items():
                stats[key].append(value)
            if seed == 0:
                text = tokenizer.decode(out.tolist(), skip_special_tokens=False)
                print(f"=== {rule} ===\n{text[:220]!r}")

        for key in METRICS:
            mean = statistics.fmean(stats[key])
            std = statistics.stdev(stats[key]) if len(stats[key]) > 1 else 0.0
            print(f"  {key:22s} {mean:9.4f} ± {std:.4f}")
        print()

if __name__ == "__main__":
    main()