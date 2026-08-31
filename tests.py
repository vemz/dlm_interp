import torch
from transformers import AutoTokenizer

from load import load_model, nano_forward_fn
from samplers import run_sampler, uniform_schedule


def decode_and_report(rule: str, ids, tokenizer, mask_id: int) -> None:
    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)

    eos_count = ids.count(eos_id) if eos_id is not None else 0
    first_eos = ids.index(eos_id) if eos_id in ids else None
    pad_count = ids.count(pad_id) if pad_id is not None else 0
    mask_count = ids.count(mask_id)

    text = tokenizer.decode(ids, clean_up_tokenization_spaces=False, skip_special_tokens=False)
    print(f"=== {rule} ===")
    print(f"mask_id={mask_id} eos_id={eos_id} pad_id={pad_id}")
    print(f"counts: mask={mask_count} eos={eos_count} pad={pad_count}")
    print(f"first eos: {first_eos}")
    print(repr(text[:500]))
    print("---")


def main() -> None:
    model, cfg = load_model()
    mask_id = int(cfg["mask_id"])
    seq_len = int(cfg["seq_len"])
    forward_fn = nano_forward_fn(model)

    tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-33M")
    
    x_init = torch.full((seq_len,), mask_id, dtype=torch.long)
    timesteps = torch.linspace(1.0, 0.0, seq_len + 1).tolist()

    for rule in ("ancestral", "random_fixed_k", "top_k_confidence"):
        schedule = None if rule == "ancestral" else uniform_schedule(seq_len, len(timesteps) - 1)
        for tr in ("sample", "argmax"):
            out, _ = run_sampler(
                forward_fn,
                x_init,
                timesteps,
                mask_id,
                rule=rule,
                token_rule=tr,
                schedule=schedule,
                seed=0,
                keep_hidden=False,
            )
            ids = out.tolist()
            decode_and_report(f"{rule} / {tr}", ids, tokenizer, mask_id)


if __name__ == "__main__":
    main()
