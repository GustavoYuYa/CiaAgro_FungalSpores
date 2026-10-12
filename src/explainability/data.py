"""Paired test images using the exact preprocessing of the existing datasets."""

from pathlib import Path

import numpy as np
import torch

from src.datasets.spore_dataset import EsporosDataset, EsporosDatasetNoise
from src.datasets.transforms import transform_wout_normalize


def _sample_key(filename, folder):
    parts = str(filename).replace("\\", "/").split("/")
    if folder not in parts:
        raise ValueError(f"Expected {folder!r} in split path: {filename}")
    return tuple(parts[parts.index(folder) + 1:])


class PairedTestDataset:
    """Preserve split order: the original index is also the Noise RNG seed.

    The mask is reconstructed from the transformed black-background image,
    exactly as in EsporosDatasetNoise.__getitem__, NOT from the clean image,
    the normalized input, or a newly estimated segmentation.
    """

    def __init__(self, dataset_root, splits_root=None):
        self.root = Path(dataset_root)
        splits_root = Path(splits_root) if splits_root else self.root / "splits"
        arrays = {}
        for condition in ("normal", "segmented_black"):
            for kind in ("files", "labels"):
                path = splits_root / condition / f"test_{kind}.npy"
                arrays[condition, kind] = np.load(path, allow_pickle=True)
        self.clean_files = arrays["normal", "files"]
        self.segmented_files = arrays["segmented_black", "files"]
        self.labels = arrays["normal", "labels"]
        if any(a.ndim != 1 for a in arrays.values()):
            raise ValueError("All test split arrays must be one-dimensional.")
        if not len(self.labels) or len({len(a) for a in arrays.values()}) != 1:
            raise ValueError("Test splits must be nonempty and have equal lengths.")
        if not np.array_equal(self.labels, arrays["segmented_black", "labels"]):
            raise ValueError("Clean and segmented labels differ in split order.")
        if not np.issubdtype(self.labels.dtype, np.integer) or np.any(self.labels < 0):
            raise ValueError("Labels must be nonnegative integers.")
        self.class_names = {}
        seen = set()
        for clean, segmented, label in zip(
            self.clean_files, self.segmented_files, self.labels
        ):
            key = _sample_key(clean, "Normal")
            if key != _sample_key(segmented, "Segmented"):
                raise ValueError(f"Unpaired test images: {clean} / {segmented}")
            if len(key) < 2 or key in seen:
                raise ValueError(f"Invalid or duplicate sample: {clean}")
            seen.add(key)
            class_name = key[0]
            if self.class_names.setdefault(int(label), class_name) != class_name:
                raise ValueError(f"Conflicting class names for label {label}.")
            for filename in (clean, segmented):
                if not (self.root / str(filename)).is_file():
                    raise FileNotFoundError(self.root / str(filename))
        if len(set(self.class_names.values())) != len(self.class_names):
            raise ValueError("One class name is assigned to multiple labels.")
        transform = transform_wout_normalize()
        self.clean = EsporosDataset(
            self.clean_files, self.labels, self.root, transform=transform
        )
        self.segmented = EsporosDataset(
            self.segmented_files, self.labels, self.root, transform=transform
        )
        self.noise = EsporosDatasetNoise(
            self.segmented_files, self.labels, self.root, transform=transform,
            deterministic_noise=True, apply_masked_jitter=False,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        clean, label = self.clean[index]
        segmented, _ = self.segmented[index]
        noise, _ = self.noise[index]
        mask = (segmented.max(dim=0, keepdims=True).values > 0).float()
        return {
            "index": index,
            "label": int(label),
            "class_name": self.class_names[int(label)],
            "clean_path": str(self.clean_files[index]),
            "segmented_path": str(self.segmented_files[index]),
            "Clean": clean,
            "Noise": noise,
            "mask": mask.squeeze(0).numpy().astype(bool),
        }


def normalization_tensors(config, device="cpu"):
    normalization = config["normalization"]
    if normalization.get("mean") is None or normalization.get("std") is None:
        raise ValueError("Checkpoint must contain training mean and std.")
    mean = torch.as_tensor(normalization["mean"], dtype=torch.float32, device=device)
    std = torch.as_tensor(normalization["std"], dtype=torch.float32, device=device)
    if mean.shape != (3,) or std.shape != (3,):
        raise ValueError("Normalization must contain three RGB channels.")
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Normalization must be finite, with positive std.")
    return mean.view(3, 1, 1), std.view(3, 1, 1)
