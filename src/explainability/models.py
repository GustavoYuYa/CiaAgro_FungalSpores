"""Load existing checkpoints without changing the trained architectures."""

from functools import partial

import torch

from src.models.cnn import CNN
from src.models.simplevit import SimpleViT


def reshape_tokens(tensor, height, width):
    """SimpleViT has no CLS token: keep every spatial token."""
    if tensor.ndim != 3 or tensor.shape[1] != height * width:
        raise ValueError(f"Unexpected token shape: {tuple(tensor.shape)}")
    return tensor.reshape(tensor.shape[0], height, width, tensor.shape[2]).permute(0, 3, 1, 2)


def load_model(checkpoint_path, architecture, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    cfg = config["model"]
    # The reused project transforms are fixed at 64 x 64.
    if cfg["image_size"] != 64:
        raise ValueError("Existing dataset transforms require image_size=64.")
    if architecture == "cnn":
        model = CNN(cfg["num_classes"], cfg["channels"], cfg["dropout"])
    elif architecture == "simplevit":
        model = SimpleViT(**{name: cfg[name] for name in (
            "image_size", "patch_size", "num_classes", "dim", "depth", "heads",
            "mlp_dim", "dim_head",
        )})
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, config


def cam_layer(model, architecture, config):
    if architecture == "cnn":
        # Last convolution, before BatchNorm/ReLU and final max pooling (8 x 8).
        return model.block4[3], None, "block4.3"
    if architecture == "simplevit":
        # Same attention projection used by the reference notebook.
        side = config["model"]["image_size"] // config["model"]["patch_size"]
        reshape = partial(reshape_tokens, height=side, width=side)
        return model.transformer.layers[-1][0].to_out, reshape, "transformer.layers.-1.0.to_out"
    raise ValueError(f"Unknown architecture: {architecture}")
