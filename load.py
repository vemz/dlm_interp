import torch
from model import ModelConfig, NanoMDLM

CKPT = "baseline_s0/best.pt"

def load_model(path=CKPT):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    model = NanoMDLM(ModelConfig(**ckpt["model_cfg"]))
    model.load_state_dict(state)
    model.eval()
    return model, ckpt["model_cfg"]

def nano_forward_fn(model):
    def forward_fn(x, t):
        h, _ = model(x)
        return model.logits(h)[0], None
    return forward_fn