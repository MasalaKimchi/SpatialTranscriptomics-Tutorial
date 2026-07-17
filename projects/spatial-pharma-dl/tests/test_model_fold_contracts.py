"""Checkpoint contract evidence for locally produced fold models."""

from __future__ import annotations

import torch
import pytest

from src import data, models
from utils.artifacts import ArtifactValidationError, manifest_path

pytestmark = pytest.mark.offline


def _tiny_model() -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Linear(3, 2))


def test_local_checkpoint_round_trip_requires_contract_before_decode(tmp_path, monkeypatch):
    cfg = data.load_config()
    model = _tiny_model()
    path = tmp_path / "resnet18_v2_fold0.pt"
    metadata = {
        "model_name": "resnet18",
        "experiment": "v2",
        "pretrained": False,
        "n_classes": 2,
        "n_reg_targets": 1,
        "cls_col": "tme_class_id",
        "reg_cols": ["gene_A"],
        "val_slide": "slide_b",
        "train_slides": ["slide_a"],
        "fold": 0,
    }
    models.save_model_checkpoint(
        path,
        model=model,
        metadata=metadata,
        cfg=cfg,
        upstream_lineage={"patches": ["p"], "labels": ["l"]},
    )
    assert manifest_path(path).is_file()

    calls: list[str] = []
    real_load = models.torch.load

    def observed_load(*args, **kwargs):
        calls.append("torch.load")
        return real_load(*args, **kwargs)

    monkeypatch.setattr(models.torch, "load", observed_load)
    payload = models.load_local_checkpoint_payload(path, cfg=cfg)
    assert payload["fold"] == 0
    assert calls == ["torch.load"]

    original_decoder = models._read_trusted_local_checkpoint

    def aba_decoder(snapshot):
        held = path.with_name("held-checkpoint.pt")
        path.rename(held)
        path.write_bytes(b"unadmitted-checkpoint-bytes")
        path.unlink()
        held.rename(path)
        return original_decoder(snapshot)

    monkeypatch.setattr(models, "_read_trusted_local_checkpoint", aba_decoder)
    assert models.load_local_checkpoint_payload(path, cfg=cfg)["fold"] == 0

    manifest_path(path).unlink()
    calls.clear()
    with pytest.raises(ArtifactValidationError, match="legacy_artifact"):
        models.load_local_checkpoint_payload(path, cfg=cfg)
    assert calls == []


def test_checkpoint_fingerprint_tracks_fold_target_model_and_input_not_presentation():
    cfg = data.load_config()
    base = dict(
        cfg=cfg,
        metadata={
            "model_name": "resnet18",
            "experiment": "v2",
            "pretrained": False,
            "n_classes": 2,
            "n_reg_targets": 1,
            "reg_cols": ["gene_A"],
            "cls_col": "tme_class_id",
            "fold": 0,
            "val_slide": "slide_b",
            "train_slides": ["slide_a"],
        },
        upstream_lineage={"patches": ["p"], "labels": ["l"]},
    )
    first = models.checkpoint_fingerprint(**base)
    cfg_presentation = {
        **cfg,
        "foundation": {**cfg["foundation"], "cache": not cfg["foundation"]["cache"]},
    }
    assert models.checkpoint_fingerprint(**{**base, "cfg": cfg_presentation}).digest == first.digest
    for changed in (
        {**base["metadata"], "fold": 1},
        {**base["metadata"], "reg_cols": ["gene_B"]},
        {**base["metadata"], "model_name": "resnet50"},
    ):
        assert models.checkpoint_fingerprint(**{**base, "metadata": changed}).digest != first.digest
    assert models.checkpoint_fingerprint(
        **{**base, "upstream_lineage": {"patches": ["q"], "labels": ["l"]}}
    ).digest != first.digest
