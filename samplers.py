from __future__ import annotations
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any
import torch

DEFAULT_TOPK = 16
NOT_COMMITTED = -1

@dataclass
class StepTrace:
    """One denoising step. All per-position tensors are aligned with
    `masked_positions` and cover every position masked at the START of the
    step, committed or not."""

    step: int
    t: float
    s: float

    masked_positions: torch.Tensor  # (n_masked,) int64 — indices into the sequence
    committed: torch.Tensor  # (n_masked,) bool
    written_tokens: torch.Tensor  # (n_masked,) int64, NOT_COMMITTED where not committed

    confidence: torch.Tensor  # (n_masked,) float32 — max softmax prob
    entropy: torch.Tensor  # (n_masked,) float32
    margin: torch.Tensor  # (n_masked,) float32 — top1 - top2

    topk_token_ids: torch.Tensor  # (n_masked, K) int64
    topk_logits: torch.Tensor  # (n_masked, K) float32

    hidden_states: dict[int, torch.Tensor] | None = None  # layer -> (n_masked, d) fp16

    @property
    def n_masked(self) -> int:
        return int(self.masked_positions.numel())

    @property
    def n_committed(self) -> int:
        return int(self.committed.sum())


@dataclass
class GenerationTrace:
    """One generation. The trajectory is reconstructable from `x_init` plus the
    commit events, so intermediate states are not stored."""

    sampler: str
    token_rule: str
    seed: int | None
    mask_token_id: int
    length: int
    timesteps: list[float]
    schedule: list[int] | None
    x_init: torch.Tensor  # (L,) int64
    x_final: torch.Tensor  # (L,) int64
    steps: list[StepTrace] = field(default_factory=list)

    # filled by `annotate_trace`
    token_types: torch.Tensor | None = None  # (L,) bool — True = content position
    effective_len: int | None = None
    canvas_limited: bool | None = None

    @property
    def n_steps(self) -> int:
        return len(self.steps)


# --------------------------------------------------------------------------
# Forward wrappers
# --------------------------------------------------------------------------


def simple_forward_fn(model, pass_time: bool = True) -> Callable:
    """Wrap a model as `forward_fn(x, t) -> (logits, None)`.

    x is (1, L); logits must come back as (1, L, V) or (L, V)."""

    def forward_fn(x: torch.Tensor, t: float):
        out = model(x, t) if pass_time else model(x)
        logits = out.logits if hasattr(out, "logits") else out
        if logits.dim() == 3:
            logits = logits[0]
        return logits, None

    return forward_fn


class HiddenCapture:
    """Forward hooks that grab the output of chosen modules.

    Usage:
        capture = HiddenCapture({6: model.blocks[6], 12: model.blocks[12]})
        forward_fn = capture.wrap(simple_forward_fn(model))
    """

    def __init__(self, modules: dict[int, torch.nn.Module]):
        self.modules = modules
        self._buffer: dict[int, torch.Tensor] = {}
        self._handles: list[Any] = []

    def _make_hook(self, layer: int):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            if tensor.dim() == 3:
                tensor = tensor[0]
            self._buffer[layer] = tensor.detach()

        return hook

    def __enter__(self) -> HiddenCapture:
        for layer, module in self.modules.items():
            self._handles.append(module.register_forward_hook(self._make_hook(layer)))
        return self

    def __exit__(self, *_exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def wrap(self, forward_fn: Callable) -> Callable:
        def wrapped(x: torch.Tensor, t: float):
            self._buffer.clear()
            logits, _ = forward_fn(x, t)
            return logits, dict(self._buffer)

        return wrapped


# --------------------------------------------------------------------------
# Schedules and scores
# --------------------------------------------------------------------------


def uniform_schedule(n_masked: int, n_steps: int) -> list[int]:
    """One k per transition, matching LLaDA's `get_num_transfer_tokens`:
    the remainder is spread over the FIRST steps, so early steps commit one
    extra token. Sums exactly to n_masked."""
    base, remainder = divmod(n_masked, n_steps)
    return [base + (1 if i < remainder else 0) for i in range(n_steps)]


def _score_positions(logits: torch.Tensor, topk: int):
    """logits: (n, V) -> confidence, entropy, margin, topk ids, topk logits."""
    logprobs = logits.log_softmax(dim=-1)
    probs = logprobs.exp()

    entropy = -(probs * logprobs).sum(dim=-1)

    k = min(topk, logits.shape[-1])
    top_probs, top_ids = probs.topk(k, dim=-1)
    confidence = top_probs[:, 0]
    margin = top_probs[:, 0] - top_probs[:, 1] if k > 1 else torch.ones_like(confidence)
    top_logits = logits.gather(-1, top_ids)

    return confidence, entropy, margin, top_ids, top_logits


def _sync(device: torch.device) -> None:
    """MPS copies are async; force completion before reading tensors back."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


# --------------------------------------------------------------------------
# Core loop
# --------------------------------------------------------------------------

_DEFAULT_TOKEN_RULE = {
    "ancestral": "sample",
    "random_fixed_k": "argmax",
    "top_k_confidence": "argmax",
}


@torch.no_grad()
def run_sampler(
    forward_fn: Callable,
    x_init: torch.Tensor,
    timesteps: Sequence[float],
    mask_token_id: int,
    rule: str,
    schedule: Sequence[int] | None = None,
    token_rule: str | None = None,
    topk: int = DEFAULT_TOPK,
    seed: int | None = None,
    keep_hidden: bool = True,
) -> tuple[torch.Tensor, GenerationTrace]:
    """Decode one sequence and return (final tokens, trace).

    timesteps runs from the noisiest time down to 0, e.g. linspace(1, 0, T+1).
    `schedule` is required for the fixed-k rules; use `uniform_schedule`.
    """
    if rule not in _DEFAULT_TOKEN_RULE:
        raise ValueError(f"unknown rule {rule!r}")
    token_rule = token_rule or _DEFAULT_TOKEN_RULE[rule]
    if token_rule not in ("sample", "argmax"):
        raise ValueError(f"unknown token_rule {token_rule!r}")

    x = x_init.squeeze(0) if x_init.dim() == 2 else x_init
    if x.dim() != 1:
        raise ValueError("x must be (L,) or (1, L)")
    result = x.clone()
    device = result.device
    length = int(result.numel())

    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)

    transitions = list(zip(timesteps[:-1], timesteps[1:]))
    for t, s in transitions:
        if not (0.0 <= float(s) < float(t)):
            raise ValueError("timesteps must be strictly decreasing and non-negative")

    if rule != "ancestral":
        if schedule is None:
            raise ValueError(f"{rule} needs a schedule")
        schedule = list(schedule)
        if len(schedule) != len(transitions):
            raise ValueError("schedule must hold one k per timestep transition")
        n_masked_init = int((result == mask_token_id).sum())
        if sum(schedule) < n_masked_init:
            raise ValueError(
                f"schedule commits {sum(schedule)} tokens but {n_masked_init} are masked; "
                "the generation would end unfinished"
            )

    trace = GenerationTrace(
        sampler=rule,
        token_rule=token_rule,
        seed=seed,
        mask_token_id=mask_token_id,
        length=length,
        timesteps=[float(v) for v in timesteps],
        schedule=list(schedule) if schedule is not None else None,
        x_init=x.clone().cpu(),
        x_final=result,  # replaced at the end
    )

    for step, (t, s) in enumerate(transitions):
        masked = result == mask_token_id
        if not masked.any():
            break
        masked_positions = masked.nonzero(as_tuple=False).squeeze(-1)
        n_masked = int(masked_positions.numel())

        logits_all, hidden = forward_fn(result.unsqueeze(0), float(t))
        logits = logits_all[masked_positions].float()

        confidence, entropy, margin, top_ids, top_logits = _score_positions(logits, topk)

        # ---- where to unmask -------------------------------------------------
        if rule == "ancestral":
            reveal_p = (float(t) - float(s)) / float(t)
            draws = torch.rand(n_masked, generator=generator).to(device)
            selected = (draws < reveal_p).nonzero(as_tuple=False).squeeze(-1)
        else:
            k = min(int(schedule[step]), n_masked)
            if k <= 0:
                continue
            if rule == "random_fixed_k":
                order = torch.randperm(n_masked, generator=generator).to(device)
                selected = order[:k]
            else:  # top_k_confidence
                selected = confidence.topk(k).indices

        # ---- what to write ---------------------------------------------------
        written = torch.full((n_masked,), NOT_COMMITTED, dtype=torch.long, device=device)
        if selected.numel() > 0:
            chosen_logits = logits[selected]
            if token_rule == "argmax":
                tokens = chosen_logits.argmax(dim=-1)
            else:
                probs = chosen_logits.softmax(dim=-1).cpu()
                tokens = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
                tokens = tokens.to(device)
            written[selected] = tokens
            result[masked_positions[selected]] = tokens

        committed = written != NOT_COMMITTED

        _sync(device)
        trace.steps.append(
            StepTrace(
                step=step,
                t=float(t),
                s=float(s),
                masked_positions=masked_positions.cpu(),
                committed=committed.cpu(),
                written_tokens=written.cpu(),
                confidence=confidence.cpu(),
                entropy=entropy.cpu(),
                margin=margin.cpu(),
                topk_token_ids=top_ids.cpu(),
                topk_logits=top_logits.cpu(),
                hidden_states=(
                    {
                        layer: h[masked_positions].to(torch.float16).cpu()
                        for layer, h in hidden.items()
                    }
                    if (keep_hidden and hidden)
                    else None
                ),
            )
        )

    trace.x_final = result.clone().cpu()
    return result, trace


# --------------------------------------------------------------------------
# Post-processing
# --------------------------------------------------------------------------


def annotate_trace(
    trace: GenerationTrace,
    structural_token_ids: set[int],
    eos_token_id: int | None = None,
) -> GenerationTrace:
    """Fill token_types, effective_len and canvas_limited.

    A position is *content* iff its final token is not structural. effective_len
    is the number of tokens before the first EOS, and canvas_limited uses the
    corrected test (>= L - 1), which catches an answer that fills the whole
    canvas and puts EOS in the last slot."""
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


def trace_to_rows(trace: GenerationTrace, generation_id: str) -> dict[str, torch.Tensor]:
    """Flatten to one row per (step, masked position) — the format the probes
    and the tau_b estimator both want. Hidden states stay out; keep them in the
    trace file and join on (generation_id, step, position)."""
    columns: dict[str, list[torch.Tensor]] = {
        "step": [],
        "t": [],
        "position": [],
        "committed": [],
        "written_token": [],
        "confidence": [],
        "entropy": [],
        "margin": [],
        "n_masked": [],
    }
    for st in trace.steps:
        n = st.n_masked
        columns["step"].append(torch.full((n,), st.step, dtype=torch.long))
        columns["t"].append(torch.full((n,), st.t, dtype=torch.float32))
        columns["position"].append(st.masked_positions)
        columns["committed"].append(st.committed)
        columns["written_token"].append(st.written_tokens)
        columns["confidence"].append(st.confidence)
        columns["entropy"].append(st.entropy)
        columns["margin"].append(st.margin)
        columns["n_masked"].append(torch.full((n,), n, dtype=torch.long))

    out = {key: torch.cat(value) for key, value in columns.items()}

    if trace.token_types is not None:
        out["is_content"] = trace.token_types[out["position"]]

    out["generation_id"] = [generation_id] * int(out["step"].numel())
    out["sampler"] = [trace.sampler] * int(out["step"].numel())
    return out


def first_commit_index(trace: GenerationTrace) -> torch.Tensor:
    """(L,) int64 — the step at which each position was FIRST committed, or -1.

    First-acceptance, not last: commits are not monotone in practice, so the
    tau_b estimator needs the first one."""
    out = torch.full((trace.length,), -1, dtype=torch.long)
    for st in trace.steps:
        hit = st.masked_positions[st.committed]
        fresh = out[hit] == -1
        out[hit[fresh]] = st.step
    return out


def save_trace(trace: GenerationTrace, path: str) -> None:
    torch.save(trace, path)


def load_trace(path: str) -> GenerationTrace:
    return torch.load(path, weights_only=False)


# --------------------------------------------------------------------------
# Unit test from M1: the ancestral rate
# --------------------------------------------------------------------------


def check_ancestral_rate(
    forward_fn: Callable,
    x_init: torch.Tensor,
    timesteps: Sequence[float],
    mask_token_id: int,
    n_runs: int = 20,
    tolerance: float = 0.15,
) -> dict[str, float]:
    """M1 predicts E[commits per step] = L / T, constant over time.

    Deviating means the reveal probability is wrong — a bug, not a finding.
    Late steps are noisy by construction (few masked positions left), so the
    check is on the mean over the first half of the trajectory."""
    length = int(x_init.numel())
    n_steps = len(timesteps) - 1
    expected = length / n_steps

    counts: list[float] = []
    for run in range(n_runs):
        _, trace = run_sampler(
            forward_fn,
            x_init,
            timesteps,
            mask_token_id,
            rule="ancestral",
            seed=run,
            keep_hidden=False,
        )
        half = max(1, len(trace.steps) // 2)
        counts.extend(float(st.n_committed) for st in trace.steps[:half])

    observed = sum(counts) / len(counts)
    deviation = abs(observed - expected) / expected
    return {
        "expected_per_step": expected,
        "observed_per_step": observed,
        "relative_deviation": deviation,
        "passed": float(deviation < tolerance),
        "n_step_samples": float(len(counts)),
    }


__all__ = [
    "StepTrace",
    "GenerationTrace",
    "simple_forward_fn",
    "HiddenCapture",
    "uniform_schedule",
    "run_sampler",
    "annotate_trace",
    "trace_to_rows",
    "first_commit_index",
    "save_trace",
    "load_trace",
    "check_ancestral_rate",
]