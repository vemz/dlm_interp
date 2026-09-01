import torch
from load import load_model, nano_forward_fn
from samplers import HiddenCapture, check_ancestral_rate

model, cfg = load_model()
MASK_ID, L = cfg["mask_id"], cfg["seq_len"]

x_init = torch.full((L,), MASK_ID, dtype=torch.long)
timesteps = torch.linspace(1.0, 0.0, L + 1).tolist()

capture = HiddenCapture({1: model.blocks[1], 3: model.blocks[3], 5: model.blocks[5]})
with capture:
    forward_fn = capture.wrap(nano_forward_fn(model))
    print(check_ancestral_rate(forward_fn, x_init, timesteps, MASK_ID, n_runs=5))