"""Small adapter around jacobgil/pytorch-grad-cam."""

import numpy as np
import torch
from pytorch_grad_cam import EigenCAM, EigenGradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


METHODS = {"EigenCAM": EigenCAM, "EigenGradCAM": EigenGradCAM}


def generate_cam(model, layer, reshape, input_tensor, target_class, method):
    """Release hooks before deletion inference; do not retain its activations.

    Use try/finally rather than BaseCAM.__exit__, which suppresses IndexError
    in some package versions. EigenCAM ignores the category by design.
    """
    algorithm = METHODS[method](model=model, target_layers=[layer], reshape_transform=reshape)
    try:
        with torch.enable_grad():
            result = algorithm(
                input_tensor=input_tensor.detach().clone().requires_grad_(True),
                targets=[ClassifierOutputTarget(int(target_class))],
                aug_smooth=False, eigen_smooth=False,
            )[0]
        result = np.asarray(result, dtype=np.float32)
        if result.shape != tuple(input_tensor.shape[-2:]) or not np.isfinite(result).all():
            raise ValueError("CAM has an invalid shape or contains nonfinite values.")
        return np.maximum(result, 0)
    finally:
        algorithm.activations_and_grads.release()
        model.zero_grad(set_to_none=True)
