"""The same single-substitution experiment on Dream, with its native logit alignment.
The shared decoder, fixed examples, controls and checkpointing live in llada.py.
Dream predicts the next position: shift its logits once before selecting tokens.
This is an adapted block schedule, not Dream's stock diffusion sampler.
"""
from llada import make_app, launch, remote_experiment

SETTINGS = dict(name='dream', model='Dream-org/Dream-v0-Instruct-7B',
                revision='05334cb9faaf763692dcf9d8737c642be2b2a6ae',
                mask_id=151666, end_ids=[151643, 151645],
                model_kwargs={'attn_implementation': 'sdpa'})


def aligned_logits(model, x):
    import torch
    raw = model(x, attention_mask='full', use_cache=False).logits
    if raw.ndim != 3 or raw.shape[:2] != x.shape:
        raise ValueError('Unexpected model output shape')
    return torch.cat((raw[:, :1], raw[:, :-1]), dim=1)


def run_experiment(phase: str, run_id: str):
    return remote_experiment(phase, run_id, 'dream')


def main(phase: str = 'pilot', run_id: str = ''):
    launch(run, 'dream', phase, run_id)


if __name__ == '__main__':
    print(__doc__)
else:
    try:
        import modal
    except ImportError:
        pass
    else:
        app, run = make_app('dream', run_experiment)
        main = app.local_entrypoint()(main)
