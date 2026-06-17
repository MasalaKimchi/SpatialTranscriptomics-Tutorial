"""ImageNet normalization for CNN training and inference."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
    """Normalize CHW or NCHW float tensor in [0, 1] with ImageNet statistics."""
    if x.ndim == 3:
        mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(3, 1, 1)
    else:
        mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


class NormalizedDataset(Dataset):
    """Wrap a patch dataset with ImageNet normalization on inputs."""

    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y_cls, y_reg = self.base[idx]
        return imagenet_normalize(x), y_cls, y_reg
