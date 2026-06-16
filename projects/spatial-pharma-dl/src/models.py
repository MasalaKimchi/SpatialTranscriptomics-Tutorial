"""ResNet18 multi-task model for domain classification and gene regression."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class MultiTaskResNet18(nn.Module):
    """Shared ResNet18 backbone with classification and regression heads."""

    def __init__(
        self,
        n_classes: int,
        n_genes: int,
        pretrained: bool = True,
    ):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        feat_dim = backbone.fc.in_features
        self.cls_head = nn.Linear(feat_dim, n_classes)
        self.reg_head = nn.Linear(feat_dim, n_genes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.features(x).flatten(1)
        return self.cls_head(h), self.reg_head(h)


def build_model(n_classes: int, n_genes: int, pretrained: bool = True) -> MultiTaskResNet18:
    return MultiTaskResNet18(n_classes, n_genes, pretrained=pretrained)
