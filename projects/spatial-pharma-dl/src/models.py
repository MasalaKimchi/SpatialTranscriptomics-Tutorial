"""Multi-task image models for domain classification and gene regression."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision import models

from .data import load_config
from .validation import resolve_config
from utils.artifacts import (
    ARTIFACT_CONTRACT_VERSIONS,
    ArtifactFingerprint,
    ArtifactValidationError,
    admit_artifact,
    build_fingerprint,
    publish_artifact,
    read_artifact_manifest,
)

# Backbones exposed via configs/default.yaml training.model
SUPPORTED_BACKBONES = (
    "resnet18",       # fast baseline; good for tutorials and smoke tests
    "resnet50",       # stronger ImageNet CNN; common in computational pathology
    "efficientnet_b0",  # better accuracy/param trade-off than ResNet18
    "convnext_tiny",  # modern CNN foundation model (ConvNeXt family)
    "vit_b_16",       # ViT foundation model; captures global patch context
)

_DEFAULT_BACKBONE = "resnet18"

_CHECKPOINT_METADATA_KEYS = frozenset(
    {
        "model_name",
        "experiment",
        "pretrained",
        "n_classes",
        "n_reg_targets",
        "cls_col",
        "reg_cols",
        "val_slide",
        "train_slides",
        "fold",
    }
)


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


def _checkpoint_metadata(value: object) -> dict[str, Any]:
    if type(value) is not dict or frozenset(value) != _CHECKPOINT_METADATA_KEYS:
        raise ArtifactValidationError("reader_validation_failed", artifact_kind="checkpoint")
    metadata = value
    string_keys = ("model_name", "experiment", "cls_col", "val_slide")
    if any(type(metadata[key]) is not str or not metadata[key] for key in string_keys):
        raise ArtifactValidationError("reader_validation_failed", artifact_kind="checkpoint")
    if metadata["model_name"] not in SUPPORTED_BACKBONES:
        raise ArtifactValidationError("reader_validation_failed", artifact_kind="checkpoint")
    if type(metadata["pretrained"]) is not bool:
        raise ArtifactValidationError("reader_validation_failed", artifact_kind="checkpoint")
    for key in ("n_classes", "n_reg_targets"):
        if type(metadata[key]) is not int or metadata[key] < 1:
            raise ArtifactValidationError("reader_validation_failed", artifact_kind="checkpoint")
    if type(metadata["fold"]) is not int or metadata["fold"] < 0:
        raise ArtifactValidationError("reader_validation_failed", artifact_kind="checkpoint")
    for key in ("reg_cols", "train_slides"):
        values = metadata[key]
        if type(values) is not list or any(type(item) is not str or not item for item in values):
            raise ArtifactValidationError("reader_validation_failed", artifact_kind="checkpoint")
    if len(metadata["reg_cols"]) != metadata["n_reg_targets"] or not metadata["train_slides"]:
        raise ArtifactValidationError("reader_validation_failed", artifact_kind="checkpoint")
    if metadata["val_slide"] in metadata["train_slides"]:
        raise ArtifactValidationError("reader_validation_failed", artifact_kind="checkpoint")
    return {key: metadata[key] for key in sorted(metadata)}


def _state_schema(state_dict: object) -> dict[str, object]:
    if type(state_dict) is not dict or not state_dict:
        raise ArtifactValidationError("reader_validation_failed", artifact_kind="checkpoint")
    tensors: list[dict[str, object]] = []
    for key in sorted(state_dict):
        value = state_dict[key]
        if type(key) is not str or type(value) is not torch.Tensor:
            raise ArtifactValidationError("reader_validation_failed", artifact_kind="checkpoint")
        tensors.append(
            {
                "key": key,
                "shape": [int(item) for item in value.shape],
                "dtype": str(value.dtype),
            }
        )
    return {"state_keys": [item["key"] for item in tensors], "tensors": tensors}


def checkpoint_fingerprint(
    *,
    cfg: dict[str, Any],
    metadata: dict[str, Any],
    upstream_lineage: dict[str, object],
) -> ArtifactFingerprint:
    """Fingerprint model/training/target/fold and exact upstream lineage."""
    resolved = resolve_config(cfg).to_dict()
    admitted = _checkpoint_metadata(metadata)
    return build_fingerprint(
        "checkpoint",
        {
            "configuration": resolved,
            "source": {
                "model": admitted["model_name"],
                "pretrained": admitted["pretrained"],
                "experiment": admitted["experiment"],
                "targets": {
                    "classification": admitted["cls_col"],
                    "regression": admitted["reg_cols"],
                    "n_classes": admitted["n_classes"],
                },
            },
            "upstream": upstream_lineage,
            "identity": {
                "fold": admitted["fold"],
                "train_slides": admitted["train_slides"],
                "val_slide": admitted["val_slide"],
            },
        },
    )


def _read_trusted_local_checkpoint(path: Path) -> tuple[dict[str, Any], dict[str, object]]:
    """Decode only a contract-admitted checkpoint emitted by this writer.

    ``weights_only=False`` is intentionally retained for local compatibility. A
    checksum is not authenticity; hostile-checkpoint safety remains Phase 5.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if type(payload) is not dict or frozenset(payload) != {
        "metadata", "state_dict", "state_schema"
    }:
        raise ArtifactValidationError("reader_validation_failed", artifact_kind="checkpoint")
    metadata = _checkpoint_metadata(payload["metadata"])
    observed_schema = _state_schema(payload["state_dict"])
    if type(payload["state_schema"]) is not dict or payload["state_schema"] != observed_schema:
        raise ArtifactValidationError("reader_validation_failed", artifact_kind="checkpoint")
    return {**metadata, "state_dict": payload["state_dict"]}, {
        "metadata_keys": sorted(_CHECKPOINT_METADATA_KEYS),
        **observed_schema,
        "decode_policy": "trusted-local-writer-only-weights-only-false",
    }


def save_model_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    metadata: dict[str, Any],
    cfg: dict[str, Any] | None = None,
    upstream_lineage: dict[str, object],
) -> Path:
    """Atomically publish a locally generated checkpoint and exact state schema."""
    resolved = load_config() if cfg is None else resolve_config(cfg).to_dict()
    admitted_metadata = _checkpoint_metadata(metadata)
    state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    state_schema = _state_schema(state_dict)
    schema = {
        "metadata_keys": sorted(_CHECKPOINT_METADATA_KEYS),
        **state_schema,
        "decode_policy": "trusted-local-writer-only-weights-only-false",
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    publish_artifact(
        destination,
        artifact_kind="checkpoint",
        contract_version=ARTIFACT_CONTRACT_VERSIONS["checkpoint"],
        fingerprint=checkpoint_fingerprint(
            cfg=resolved,
            metadata=admitted_metadata,
            upstream_lineage=upstream_lineage,
        ),
        payload_format="pytorch-local-checkpoint",
        payload_schema=schema,
        write_payload=lambda temporary: torch.save(
            {
                "metadata": admitted_metadata,
                "state_dict": state_dict,
                "state_schema": state_schema,
            },
            temporary,
        ),
        reader=_read_trusted_local_checkpoint,
    )
    return destination


def _expected_checkpoint_fingerprint(
    path: Path,
    *,
    cfg: dict[str, Any],
    upstream_lineage: dict[str, object] | None,
) -> ArtifactFingerprint:
    sidecar = read_artifact_manifest(path)
    inputs = sidecar.fingerprint.to_dict()["inputs"]
    if upstream_lineage is not None:
        inputs["upstream"] = upstream_lineage
    inputs["configuration"] = resolve_config(cfg).to_dict()
    return build_fingerprint("checkpoint", {
        "configuration": inputs["configuration"],
        "source": inputs["source"],
        "upstream": inputs["upstream"],
        "identity": inputs["identity"],
    })


def load_local_checkpoint_payload(
    path: str | Path,
    *,
    cfg: dict[str, Any] | None = None,
    upstream_lineage: dict[str, object] | None = None,
) -> dict[str, Any]:
    """Admit contract/checksum/schema before the local pickle-compatible decode."""
    resolved = load_config() if cfg is None else resolve_config(cfg).to_dict()
    checkpoint_path = Path(path)
    expected = _expected_checkpoint_fingerprint(
        checkpoint_path, cfg=resolved, upstream_lineage=upstream_lineage
    )
    admission = admit_artifact(
        checkpoint_path,
        expected_kind="checkpoint",
        expected_contract_version=ARTIFACT_CONTRACT_VERSIONS["checkpoint"],
        expected_fingerprint=expected,
        reader=_read_trusted_local_checkpoint,
    )
    payload, schema = admission.value
    import json

    if json.loads(admission.manifest.payload_schema_json) != schema:
        raise ArtifactValidationError(
            "payload_schema_mismatch", artifact_kind="checkpoint", basename=checkpoint_path.name
        )
    return payload


def load_model_from_checkpoint(
    path: str | Path,
    map_location: str | torch.device | None = "cpu",
    *,
    cfg: dict[str, Any] | None = None,
    upstream_lineage: dict[str, object] | None = None,
) -> tuple[MultiTaskImageModel, dict[str, Any]]:
    """Restore a contract-admitted local model; hostile safety is Phase 5."""
    ckpt = load_local_checkpoint_payload(
        path, cfg=cfg, upstream_lineage=upstream_lineage
    )
    model_name = ckpt["model_name"]
    model = build_model(
        ckpt["n_classes"],
        ckpt["n_reg_targets"],
        model_name=model_name,
        pretrained=False,
    )
    expected_schema = _state_schema(dict(model.state_dict()))
    observed_schema = _state_schema(ckpt["state_dict"])
    if expected_schema != observed_schema:
        raise ArtifactValidationError("reader_validation_failed", artifact_kind="checkpoint")
    model.load_state_dict(ckpt["state_dict"])
    if map_location is not None:
        model.to(map_location)
    return model, ckpt
