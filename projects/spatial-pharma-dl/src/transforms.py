"""ImageNet normalization and training-time augmentation for H&E patches."""

from __future__ import annotations

import random

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


def augment_patch(x: torch.Tensor) -> torch.Tensor:
    """Random flip, 90° rotation, and mild color jitter (train only)."""
    if random.random() < 0.5:
        x = torch.flip(x, dims=[2])
    if random.random() < 0.5:
        x = torch.flip(x, dims=[1])
    if random.random() < 0.5:
        x = torch.rot90(x, k=random.randint(1, 3), dims=(1, 2))
    # brightness / contrast jitter in [0,1] space
    if random.random() < 0.5:
        b = 1.0 + random.uniform(-0.08, 0.08)
        x = (x * b).clamp(0.0, 1.0)
    if random.random() < 0.5:
        c = 1.0 + random.uniform(-0.08, 0.08)
        mean = x.mean(dim=(1, 2), keepdim=True)
        x = ((x - mean) * c + mean).clamp(0.0, 1.0)
    return x


class NormalizedDataset(Dataset):
    """Wrap a patch dataset with ImageNet normalization on inputs."""

    def __init__(self, base: Dataset, augment: bool = False):
        self.base = base
        self.augment = augment

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y_cls, y_reg = self.base[idx]
        if self.augment:
            x = augment_patch(x)
        return imagenet_normalize(x), y_cls, y_reg
