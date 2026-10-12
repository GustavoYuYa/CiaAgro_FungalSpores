"""Localization metrics and deletion AUC with an explicit perturbation protocol."""

import numpy as np
import torch


def localization_metrics(cam, mask, percentile=90.0):
    """Notebook definitions: sum inside / sum total, and percentile-threshold IoU.

    Ties at the threshold are retained (>=), as in the reference notebook.
    A zero CAM yields zero energy and zero IoU, not an all-positive mask.
    Empty foreground masks are invalid references and return NaN metrics.
    """
    cam = np.asarray(cam, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if cam.ndim != 2 or cam.shape != mask.shape:
        raise ValueError("CAM and foreground mask must have the same H x W shape.")
    if not 0 <= percentile <= 100 or not np.isfinite(cam).all() or (cam < 0).any():
        raise ValueError("Expected finite nonnegative CAM and percentile in [0, 100].")
    total = cam.sum()
    threshold = float(np.percentile(cam, percentile))
    binary = cam >= threshold if total > 0 else np.zeros_like(mask)
    valid = bool(mask.any())
    union = np.logical_or(binary, mask).sum()
    return {
        "energy_inside": float(cam[mask].sum() / total) if total > 0 and valid else (0.0 if valid else float("nan")),
        "iou": float(np.logical_and(binary, mask).sum() / union) if valid and union else float("nan"),
        "mask_fraction": float(mask.mean()),
        "mask_valid": valid,
        "cam_zero": bool(total == 0),
        "cam_constant": bool(np.ptp(cam) == 0),
        "iou_threshold": threshold,
        "cam_selected_fraction": float(binary.mean()),
    }


def deletion_baseline(rgb, mean, kind="mean", seed=0):
    """Replacement RGB values in [0,1], applied BEFORE normalization."""
    if kind == "mean":
        return mean.expand_as(rgb).clone()
    if kind == "black":
        return torch.zeros_like(rgb)
    if kind == "gaussian":
        noise = np.random.RandomState(seed).normal(0.5, 0.2, tuple(rgb.shape))
        return torch.as_tensor(np.clip(noise, 0, 1), dtype=rgb.dtype, device=rgb.device)
    raise ValueError(f"Unknown deletion baseline: {kind}")


def deletion_auc(model, rgb, cam, target_class, mean, std, *, steps=100,
                 batch_size=32, baseline="mean", seed=0):
    """Delete highest-CAM spatial pixels (all RGB channels together).

    Integrate the fixed target's softmax probability over the ACTUAL fraction
    removed, from 0 to 1 inclusive. No division by the initial probability.
    Tied saliency values are ordered by flattened pixel index (stable sort).
    Lower AUC means faster loss of target confidence for this chosen baseline.
    """
    if steps < 1 or batch_size < 1:
        raise ValueError("Deletion steps and batch size must be positive.")
    if rgb.ndim != 3 or rgb.shape[0] != 3 or tuple(rgb.shape[1:]) != np.shape(cam):
        raise ValueError("Expected RGB C x H x W and matching CAM.")
    if not np.isfinite(cam).all():
        raise ValueError("CAM must be finite.")
    if model.training:
        raise ValueError("Deletion requires model.eval().")
    count = rgb.shape[-1] * rgb.shape[-2]
    order = np.argsort(-np.asarray(cam).reshape(-1), kind="stable")
    order = torch.as_tensor(order, device=rgb.device)
    counts = np.unique(np.rint(np.linspace(0, count, steps + 1)).astype(int))
    replacement = deletion_baseline(rgb, mean, baseline, seed).reshape(3, -1)
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(counts), batch_size):
            current = counts[start:start + batch_size]
            images = rgb.unsqueeze(0).repeat(len(current), 1, 1, 1)
            flat = images.view(len(current), 3, -1)
            for j, n_deleted in enumerate(current):
                pixels = order[:n_deleted]
                flat[j, :, pixels] = replacement[:, pixels]
            logits = model((images - mean) / std)
            probabilities.extend(logits.softmax(dim=1)[:, target_class].cpu().tolist())
    fractions = counts / count
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(probabilities).all():
        raise ValueError("Deletion produced nonfinite probabilities.")
    # Explicit trapezoidal integration works with NumPy 1.x and 2.x.
    auc = np.sum(np.diff(fractions) * (probabilities[:-1] + probabilities[1:]) / 2)
    return float(auc), fractions, probabilities
