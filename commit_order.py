from __future__ import annotations
import statistics
import torch
from transformers import AutoTokenizer
from load import load_model, nano_forward_fn
from quality import structural_token_ids
from samplers import annotate_trace, first_commit_index, run_sampler, uniform_schedule

RULES = ("random_fixed_k", "top_k_confidence", "left_to_right", "ancestral")
TOKEN_RULES = ("sample", "argmax")
SEEDS = range(10)
EARLY = 20
KEYS = ("frac_struct_early", "frac_struct_all", "excess_early", "step_half_struct", "n_early")

def half_point(commits, structural_pos, valid, n_steps):
    """First step at which the running share of structural commits exceeds 0.5.

    Measures how fast the collapse sets in, which is what separates the rules
    once argmax drives every arm to the same end state."""
    for step in range(n_steps):
        so_far = valid & (commits <= step)
        n = int(so_far.sum())
        if n >= 10 and float(structural_pos[so_far].sum()) / n > 0.5:
            return float(step)
    return float(n_steps)


def commit_timing(
    forward_fn, seq_len, mask_id, structural, eos_id, rule, seed, token_rule, early=EARLY
):
    timesteps = torch.linspace(1.0, 0.0, seq_len + 1).tolist()
    schedule = None if rule == "ancestral" else uniform_schedule(seq_len, seq_len)

    _, trace = run_sampler(
        forward_fn,
        torch.full((seq_len,), mask_id, dtype=torch.long),
        timesteps,
        mask_id,
        rule=rule,
        schedule=schedule,
        token_rule=token_rule,
        seed=seed,
        keep_hidden=False,
    )
    annotate_trace(trace, structural, eos_token_id=eos_id)

    commits = first_commit_index(trace)
    structural_pos = ~trace.token_types
    valid = commits >= 0

    early_mask = valid & (commits < early)
    n_early = int(early_mask.sum())
    if n_early == 0:
        return None

    frac_early = float(structural_pos[early_mask].sum()) / n_early
    frac_all = float(structural_pos[valid].sum()) / int(valid.sum())

    return {
        "frac_struct_early": frac_early,
        "frac_struct_all": frac_all,
        "excess_early": frac_early - frac_all,
        "step_half_struct": half_point(commits, structural_pos, valid, trace.n_steps),
        "n_early": float(n_early),
    }

def main():
    model, cfg = load_model()
    forward_fn = nano_forward_fn(model)
    tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-33M")

    mask_id, seq_len = int(cfg["mask_id"]), int(cfg["seq_len"])
    structural = structural_token_ids(tokenizer)
    eos_id = tokenizer.eos_token_id

    print(f"first {EARLY} commits, {len(list(SEEDS))} seeds per cell")
    print("excess_early > 0 means structural positions are committed early")
    print("step_half_struct: first step where >50% of commits so far are structural\n")

    for token_rule in TOKEN_RULES:
        print(f"########## token_rule = {token_rule} ##########")
        for rule in RULES:
            rows = [
                r
                for seed in SEEDS
                if (
                    r := commit_timing(
                        forward_fn, seq_len, mask_id, structural, eos_id, rule, seed, token_rule
                    )
                )
            ]
            if not rows:
                print(f"  {rule}: no usable generation")
                continue

            print(f"  {rule}")
            for key in KEYS:
                values = [r[key] for r in rows]
                mean = statistics.fmean(values)
                std = statistics.stdev(values) if len(values) > 1 else 0.0
                print(f"    {key:20s} {mean:8.3f} ± {std:.3f}")
        print()

if __name__ == "__main__":
    main()