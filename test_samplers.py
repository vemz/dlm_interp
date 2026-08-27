"""Smoke test: toy model, three rules, trace integrity, M1 rate check."""

import torch

from samplers import (
    HiddenCapture,
    annotate_trace,
    check_ancestral_rate,
    first_commit_index,
    run_sampler,
    trace_to_rows,
    uniform_schedule,
)

VOCAB, LENGTH, STEPS = 32, 24, 12
MASK = VOCAB - 1
EOS = VOCAB - 2


class ToyModel(torch.nn.Module):
    """Tiny bidirectional stack, enough to exercise shapes and hooks."""

    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB, 16)
        self.blocks = torch.nn.ModuleList(
            [torch.nn.TransformerEncoderLayer(16, 2, 32, batch_first=True) for _ in range(3)]
        )
        self.head = torch.nn.Linear(16, VOCAB)

    def forward(self, x, t=None):
        h = self.embed(x)
        for block in self.blocks:
            h = block(h)
        return self.head(h)


def main() -> None:
    torch.manual_seed(0)
    model = ToyModel().eval()
    x_init = torch.full((LENGTH,), MASK, dtype=torch.long)
    timesteps = torch.linspace(1.0, 0.0, STEPS + 1).tolist()
    schedule = uniform_schedule(LENGTH, STEPS)
    assert sum(schedule) == LENGTH, schedule

    capture = HiddenCapture({1: model.blocks[1], 2: model.blocks[2]})
    with capture:
        from samplers import simple_forward_fn

        forward_fn = capture.wrap(simple_forward_fn(model, pass_time=False))

        for rule in ("ancestral", "random_fixed_k", "top_k_confidence"):
            kwargs = {} if rule == "ancestral" else {"schedule": schedule}
            out, trace = run_sampler(
                forward_fn, x_init, timesteps, MASK, rule=rule, seed=7, **kwargs
            )
            annotate_trace(trace, structural_token_ids={MASK, EOS}, eos_token_id=EOS)
            rows = trace_to_rows(trace, generation_id=f"toy/{rule}")
            commits = first_commit_index(trace)

            step0 = trace.steps[0]
            assert step0.n_masked == LENGTH, "step 0 must see every position as masked"
            assert step0.confidence.shape == (LENGTH,), "scores cover ALL masked positions"
            assert step0.topk_logits.shape == (LENGTH, 16)
            assert step0.hidden_states is not None and set(step0.hidden_states) == {1, 2}
            assert step0.hidden_states[1].shape == (LENGTH, 16)
            assert rows["confidence"].numel() == rows["position"].numel()
            assert "is_content" in rows

            n_left = int((out == MASK).sum())
            print(
                f"{rule:>18} | steps={trace.n_steps:2d} "
                f"| committed={int((commits >= 0).sum()):2d}/{LENGTH} "
                f"| masked_left={n_left} "
                f"| rows={rows['position'].numel():4d} "
                f"| eff_len={trace.effective_len:2d} "
                f"| canvas_limited={trace.canvas_limited}"
            )
            if rule != "ancestral":
                assert n_left == 0, "fixed-k rules must finish the canvas"

        # R4 decoupled from R1-R3: same 'where', different 'what'
        _, sampled = run_sampler(
            forward_fn, x_init, timesteps, MASK, rule="top_k_confidence",
            schedule=schedule, token_rule="sample", seed=7, keep_hidden=False,
        )
        print(f"{'top_k + sampling':>18} | token_rule={sampled.token_rule}")

        # Paired rollouts (M4): same seed => identical trajectory
        _, a = run_sampler(forward_fn, x_init, timesteps, MASK, rule="ancestral",
                           seed=42, keep_hidden=False)
        _, b = run_sampler(forward_fn, x_init, timesteps, MASK, rule="ancestral",
                           seed=42, keep_hidden=False)
        assert torch.equal(a.x_final, b.x_final), "same seed must reproduce the run"
        print(f"{'paired seeds':>18} | reproducible: True")

        report = check_ancestral_rate(forward_fn, x_init, timesteps, MASK, n_runs=30)
        print(
            f"{'M1 rate check':>18} | expected={report['expected_per_step']:.2f} "
            f"observed={report['observed_per_step']:.2f} "
            f"dev={report['relative_deviation']:.1%} "
            f"passed={bool(report['passed'])}"
        )


if __name__ == "__main__":
    main()