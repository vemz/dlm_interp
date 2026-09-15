import torch
from transformers import AutoTokenizer


def structural_token_ids(tokenizer):
    ids = {tokenizer.eos_token_id}
    for s in [".", ",", "!", "?", '"', "'", "\n", " .", " ,", " !", " ?", " \"", " '"]:
        enc = tokenizer.encode(s)
        if len(enc) == 1:
            ids.add(enc[0])
    return ids


def content_mask(ids, structural):
    mask = torch.ones_like(ids, dtype=torch.bool)
    for tid in structural:
        mask &= ids != tid
    return mask


@torch.no_grad()
def sequence_logprob(model, ids, mask=None):
    h, _ = model(ids.unsqueeze(0))
    logprobs = model.logits(h)[0].log_softmax(-1)
    per_token = logprobs.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
    q = per_token.sum().item()
    q_content = per_token[mask].sum().item() if mask is not None else None
    return q, q_content


@torch.no_grad()
def nelbo(model, ids, mask_id, n_mc=16, generator=None):
    L = ids.numel()
    total = 0.0
    for j in range(n_mc):
        t = (j + 0.5) / n_mc
        keep = torch.rand(L, generator=generator) >= t
        if bool(keep.all()):
            continue
        x = torch.where(keep, ids, torch.full_like(ids, mask_id))
        pos = (~keep).nonzero(as_tuple=False).squeeze(-1)
        h, _ = model(x.unsqueeze(0))
        logp = model.logits(h)[0][pos].float().log_softmax(-1)
        nll = -logp.gather(-1, ids[pos].unsqueeze(-1)).squeeze(-1).sum()
        total += float(nll) / t
    return total / n_mc / L


if __name__ == "__main__":
    from src.dlm_interp.load import load_model, nano_forward_fn
    from src.dlm_interp.samplers import run_sampler

    tok = AutoTokenizer.from_pretrained("roneneldan/TinyStories-33M")
    s = structural_token_ids(tok)
    print(len(s), [tok.decode([i]) for i in sorted(s)])

    model, cfg = load_model()
    mask_id, L = int(cfg["mask_id"]), int(cfg["seq_len"])
    fwd = nano_forward_fn(model)
    ts = torch.linspace(1.0, 0.0, L + 1).tolist()

    out, _ = run_sampler(fwd, torch.full((L,), mask_id), ts, mask_id,
                         rule="ancestral", token_rule="sample", seed=0,
                         keep_hidden=False)
    m = content_mask(out, s)
    q, qc = sequence_logprob(model, out, m)
    print(f"sequence_logprob {q / L:+.4f} nats/token "
          f"(content {qc / int(m.sum()):+.4f})")
    print(f"nelbo            {nelbo(model, out, mask_id):+.4f} nats/token")
