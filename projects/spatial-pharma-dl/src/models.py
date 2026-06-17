"""Multi-task image models for domain classification and gene regression."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision import models

# Backbones exposed via configs/default.yaml training.model
SUPPORTED_BACKBONES = (
    "resnet18",       # fast baseline; good for tutorials and smoke tests
    "resnet50",       # stronger ImageNet CNN; common in computational pathology
    "efficientnet_b0",  # better accuracy/param trade-off than ResNet18
    "convnext_tiny",  # modern CNN foundation model (ConvNeXt family)
    "vit_b_16",       # ViT foundation model; captures global patch context
)

_DEFAULT_BACKBONE = "resnet18"


class MultiTaskImageModel(nn.Module):
    """Shared torchvision backbone with classification and regression heads."""

    def __init__(
        self,
        n_classes: int,
        n_genes: int,
        backbone: str = _DEFAULT_BACKBONE,
        pretrained: bool = True,
    ):
        super().__init__()
        backbone = backbone.lower()
        if backbone not in SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unknown backbone {backbone!r}; choose from {SUPPORTED_BACKBONES}"
            )
        self.backbone_name = backbone
        self.pretrained = pretrained

        weights = _imagenet_weights(backbone, pretrained)
        self.features, self.pool, feat_dim, self.gradcam_layer = _build_encoder(
            backbone, weights
        )
        self.cls_head = nn.Linear(feat_dim, n_classes)
        self.reg_head = nn.Linear(feat_dim, n_genes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone_name.startswith("vit"):
            return self._encode_vit(x)
        h = self.features(x)
        if self.pool is not None:
            h = self.pool(h)
        return h.flatten(1)

    def _encode_vit(self, x: torch.Tensor) -> torch.Tensor:
        vit = self.features
        x = vit._process_input(x)
        n = x.shape[0]
        batch_class_token = vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = vit.encoder(x)
        return x[:, 0]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encode(x)
        return self.cls_head(h), self.reg_head(h)


# Backward-compatible alias used in notebooks and docs.
MultiTaskResNet18 = MultiTaskImageModel


def _imagenet_weights(backbone: str, pretrained: bool) -> Any:
    if not pretrained:
        return None
    weight_map = {
        "resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
        "resnet50": models.ResNet50_Weights.IMAGENET1K_V1,
        "efficientnet_b0": models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        "convnext_tiny": models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
        "vit_b_16": models.ViT_B_16_Weights.IMAGENET1K_V1,
    }
    return weight_map[backbone]


def _last_conv2d(module: nn.Module) -> nn.Module:
    last: nn.Module | None = None
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            last = child
    if last is None:
        raise ValueError(f"No Conv2d layer found in {type(module).__name__}")
    return last


def _build_encoder(
    backbone: str, weights: Any
) -> tuple[nn.Module, nn.Module | None, int, nn.Module]:
    """Return (features, pool, feat_dim, gradcam_layer)."""
    if backbone == "resnet18":
        net = models.resnet18(weights=weights)
        features = nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu,
            net.maxpool,
            net.layer1,
            net.layer2,
            net.layer3,
            net.layer4,
        )
        gradcam = net.layer4[-1].conv2
        return features, net.avgpool, net.fc.in_features, gradcam

    if backbone == "resnet50":
        net = models.resnet50(weights=weights)
        features = nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu,
            net.maxpool,
            net.layer1,
            net.layer2,
            net.layer3,
            net.layer4,
        )
        gradcam = net.layer4[-1].conv2
        return features, net.avgpool, net.fc.in_features, gradcam

    if backbone == "efficientnet_b0":
        net = models.efficientnet_b0(weights=weights)
        features = net.features
        pool = net.avgpool
        gradcam = _last_conv2d(features)
        return features, pool, net.classifier[1].in_features, gradcam

    if backbone == "convnext_tiny":
        net = models.convnext_tiny(weights=weights)
        features = net.features
        pool = net.avgpool
        gradcam = _last_conv2d(features)
        return features, pool, net.classifier[2].in_features, gradcam

    if backbone == "vit_b_16":
        net = models.vit_b_16(weights=weights)
        # Patch-embedding conv is the best 2-D hook for Grad-CAM on ViT.
        gradcam = net.conv_proj
        return net, None, net.hidden_dim, gradcam

    raise ValueError(f"Unsupported backbone {backbone!r}")


def build_model(
    n_classes: int,
    n_genes: int,
    model_name: str | None = None,
    pretrained: bool = True,
) -> MultiTaskImageModel:
    """Build a multi-task model; ``model_name`` defaults to resnet18."""
    return MultiTaskImageModel(
        n_classes,
        n_genes,
        backbone=model_name or _DEFAULT_BACKBONE,
        pretrained=pretrained,
    )


def get_gradcam_layer(model: nn.Module) -> nn.Module:
    """Return the convolutional layer used for Grad-CAM."""
    if hasattr(model, "gradcam_layer"):
        return model.gradcam_layer
    # Legacy ResNet checkpoints wrapped avgpool inside features.
    if hasattr(model, "features") and hasattr(model.features[-1], "__getitem__"):
        return model.features[-1][-1].conv2
    raise TypeError("Model has no gradcam_layer; use MultiTaskImageModel.")


def load_model_from_checkpoint(
    path: str | Path,
    map_location: str | torch.device | None = "cpu",
) -> tuple[MultiTaskImageModel, dict[str, Any]]:
    """Restore model weights and metadata from a training checkpoint."""
    ckpt = torch.load(Path(path), map_location=map_location, weights_only=False)
    model_name = ckpt.get("model_name", _DEFAULT_BACKBONE)
    model = build_model(
        ckpt["n_classes"],
        ckpt["n_genes"],
        model_name=model_name,
        pretrained=False,
    )
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt
