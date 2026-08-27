from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

TOPK = 16
NOT_COMMITTED = -1

TOKEN_RULES = {
    "ancestral": "sample",
    "random_fixed_k": "argmax",
    "top_k_confidence": "argmax",
}


@dataclass
class StepTrace:
    step: int
    t: float
    s: float
    masked_positions: torch.Tensor
    committed: torch.Tensor
    written_tokens: torch.Tensor
    confidence: torch.Tensor
    entropy: torch.Tensor
    margin: torch.Tensor
    topk_token_ids: torch.Tensor
    topk_logits: torch.Tensor
    hidden_states: dict[int, torch.Tensor] | None = None

    @property
    def n_masked(self) -> int:
        return int(self.masked_positions.numel())

    @property
    def n_committed(self) -> int:
        return int(self.committed.sum())


@dataclass
class GenerationTrace:
    sampler: str
    token_rule: str
    seed: int | None
    mask_token_id: int
    length: int
    timesteps: list[float]
    schedule: list[int] | None
    x_init: torch.Tensor
    x_final: torch.Tensor
    steps: list[StepTrace] = field(default_factory=list)
    token_types: torch.Tensor | None = None
    effective_len: int | None = None
    canvas_limited: bool | None = None

    @property
    def n_steps(self) -> int:
        return len(self.steps)


def simple_forward_fn(model, pass_time: bool = True) -> Callable:
    def forward_fn(x, t):
        out = model(x, t) if pass_time else model(x)
        logits = out.logits if hasattr(out, "logits") else out
        return (logits[0] if logits.dim() == 3 else logits), None

    return forward_fn


class HiddenCapture:
    def __init__(self, modules: dict[int, torch.nn.Module]):
        self.modules = modules
        self.buffer: dict[int, torch.Tensor] = {}
        self.handles: list[Any] = []

    def _hook(self, layer):
        def fn(_module, _inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            self.buffer[layer] = (h[0] if h.dim() == 3 else h).detach()

        return fn

    def __enter__(self):
        for layer, module in self.modules.items():
            self.handles.append(module.register_forward_hook(self._hook(layer)))
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def wrap(self, forward_fn):
        def wrapped(x, t):
            self.buffer.clear()
            logits, _ = forward_fn(x, t)
            return logits, dict(self.buffer)

        return wrapped


def uniform_schedule(n_masked: int, n_steps: int) -> list[int]:
    base, remainder = divmod(n_masked, n_steps)
    return [base + (1 if i < remainder else 0) for i in range(n_steps)]


def score_positions(logits, topk=TOPK):
    logprobs = logits.log_softmax(-1)
    probs = logprobs.exp()
    entropy = -(probs * logprobs).sum(-1)
    top_probs, top_ids = probs.topk(min(topk, logits.shape[-1]), dim=-1)
    confidence = top_probs[:, 0]
    margin = confidence - top_probs[:, 1]
    return confidence, entropy, margin, top_ids, logits.gather(-1, top_ids)


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def run_sampler(
    forward_fn,
    x_init,
    timesteps,
    mask_token_id,
    rule,
    schedule=None,
    token_rule=None,
    seed=None,
    keep_hidden=True,
):
    token_rule = token_rule or TOKEN_RULES[rule]
    x = x_init.squeeze(0) if x_init.dim() == 2 else x_init
    result = x.clone()
    device = result.device

    generator = torch.Generator("cpu").manual_seed(seed) if seed is not None else None
    transitions = list(zip(timesteps[:-1], timesteps[1:]))

    if rule != "ancestral":
        schedule = list(schedule)
        assert len(schedule) == len(transitions)
        assert sum(schedule) >= int((result == mask_token_id).sum())

    trace = GenerationTrace(
        sampler=rule,
        token_rule=token_rule,
        seed=seed,
        mask_token_id=mask_token_id,
        length=int(result.numel()),
        timesteps=[float(v) for v in timesteps],
        schedule=schedule,
        x_init=x.clone().cpu(),
        x_final=result,
    )

    for step, (t, s) in enumerate(transitions):
        masked = result == mask_token_id
        if not masked.any():
            break
        positions = masked.nonzero(as_tuple=False).squeeze(-1)
        n = int(positions.numel())

        logits_all, hidden = forward_fn(result.unsqueeze(0), float(t))
        logits = logits_all[positions].float()
        confidence, entropy, margin, top_ids, top_logits = score_positions(logits)

        if rule == "ancestral":
            p = (float(t) - float(s)) / float(t)
            draws = torch.rand(n, generator=generator).to(device)
            selected = (draws < p).nonzero(as_tuple=False).squeeze(-1)
        else:
            k = min(schedule[step], n)
            if k <= 0:
                continue
            if rule == "random_fixed_k":
                selected = torch.randperm(n, generator=generator).to(device)[:k]
            else:
                selected = confidence.topk(k).indices

        written = torch.full((n,), NOT_COMMITTED, dtype=torch.long, device=device)
        if selected.numel():
            chosen = logits[selected]
            if token_rule == "argmax":
                tokens = chosen.argmax(-1)
            else:
                probs = chosen.softmax(-1).cpu()
                tokens = torch.multinomial(probs, 1, generator=generator).squeeze(-1).to(device)
            written[selected] = tokens
            result[positions[selected]] = tokens

        sync(device)
        trace.steps.append(
            StepTrace(
                step=step,
                t=float(t),
                s=float(s),
                masked_positions=positions.cpu(),
                committed=(written != NOT_COMMITTED).cpu(),
                written_tokens=written.cpu(),
                confidence=confidence.cpu(),
                entropy=entropy.cpu(),
                margin=margin.cpu(),
                topk_token_ids=top_ids.cpu(),
                topk_logits=top_logits.cpu(),
                hidden_states=(
                    {l: h[positions].half().cpu() for l, h in hidden.items()}
                    if keep_hidden and hidden
                    else None
                ),
            )
        )

    trace.x_final = result.clone().cpu()
    return result, trace


def ancestral_sampler(forward_fn, x_init, timesteps, mask_token_id, **kwargs):
    return run_sampler(forward_fn, x_init, timesteps, mask_token_id, "ancestral", **kwargs)


def random_fixed_k_sampler(forward_fn, x_init, timesteps, mask_token_id, schedule, **kwargs):
    return run_sampler(
        forward_fn, x_init, timesteps, mask_token_id, "random_fixed_k", schedule, **kwargs
    )


def top_k_confidence_sampler(forward_fn, x_init, timesteps, mask_token_id, schedule, **kwargs):
    return run_sampler(
        forward_fn, x_init, timesteps, mask_token_id, "top_k_confidence", schedule, **kwargs
    )


def annotate_trace(trace, structural_token_ids, eos_token_id=None):
    final = trace.x_final
    structural = torch.zeros_like(final, dtype=torch.bool)
    for token_id in structural_token_ids:
        structural |= final == token_id
    trace.token_types = ~structural

    if eos_token_id is None:
        trace.effective_len = trace.length
    else:
        hits = (final == eos_token_id).nonzero(as_tuple=False)
        trace.effective_len = int(hits[0]) if hits.numel() else trace.length

    trace.canvas_limited = trace.effective_len >= trace.length - 1
    return trace


def trace_to_rows(trace, generation_id):
    out = {}
    for step_trace in trace.steps:
        n = step_trace.n_masked
        row = {
            "step": torch.full((n,), step_trace.step, dtype=torch.long),
            "t": torch.full((n,), step_trace.t),
            "position": step_trace.masked_positions,
            "committed": step_trace.committed,
            "written_token": step_trace.written_tokens,
            "confidence": step_trace.confidence,
            "entropy": step_trace.entropy,
            "margin": step_trace.margin,
            "n_masked": torch.full((n,), n, dtype=torch.long),
        }
        for key, value in row.items():
            out.setdefault(key, []).append(value)

    out = {key: torch.cat(value) for key, value in out.items()}
    if trace.token_types is not None:
        out["is_content"] = trace.token_types[out["position"]]
    total = int(out["step"].numel())
    out["generation_id"] = [generation_id] * total
    out["sampler"] = [trace.sampler] * total
    return out


def first_commit_index(trace):
    out = torch.full((trace.length,), -1, dtype=torch.long)
    for step_trace in trace.steps:
        hit = step_trace.masked_positions[step_trace.committed]
        out[hit[out[hit] == -1]] = step_trace.step
    return out


def save_trace(trace, path):
    torch.save(trace, path)


def load_trace(path):
    return torch.load(path, weights_only=False)


def check_ancestral_rate(forward_fn, x_init, timesteps, mask_token_id, n_runs=20, tol=0.15):
    expected = int(x_init.numel()) / (len(timesteps) - 1)
    counts = []
    for run in range(n_runs):
        _, trace = run_sampler(
            forward_fn, x_init, timesteps, mask_token_id, "ancestral",
            seed=run, keep_hidden=False,
        )
        half = max(1, len(trace.steps) // 2)
        counts += [step_trace.n_committed for step_trace in trace.steps[:half]]

    observed = sum(counts) / len(counts)
    deviation = abs(observed - expected) / expected
    return {"expected": expected, "observed": observed, "dev": deviation, "passed": deviation < tol}