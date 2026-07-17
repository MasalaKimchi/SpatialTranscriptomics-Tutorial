"""Offline orchestration evidence for typed artifact reuse decisions."""

from __future__ import annotations

import pytest

import numpy as np
import pandas as pd

from src import data, patches
from utils import st_helpers as st
from utils.artifacts import ArtifactValidationError

pytestmark = pytest.mark.offline


def test_available_processed_status_is_pure_for_missing_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "project_root", lambda: tmp_path)
    assert data.available_processed_slide_ids(["slide_a"], cfg=data.load_config()) == set()
    assert not (tmp_path / "data").exists()


def test_legacy_patch_stops_before_local_object_decode(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "project_root", lambda: tmp_path)
    path = patches.patch_cache_path("slide_a", data.load_config())
    path.parent.mkdir(parents=True)
    path.write_bytes(b"legacy-without-sidecar")
    calls: list[str] = []

    def forbidden(*_args, **_kwargs):
        calls.append("np.load")
        raise AssertionError("unsafe decoder reached")

    monkeypatch.setattr(patches.np, "load", forbidden)
    with pytest.raises(ArtifactValidationError, match="legacy_artifact"):
        patches.load_patch_arrays("slide_a", cfg=data.load_config())
    assert calls == []


def test_patch_lineage_ignores_training_but_rejects_patch_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "project_root", lambda: tmp_path)
    cfg = data.load_config()
    values = np.zeros((1, 3, 4, 4), dtype=np.float32)
    meta = pd.DataFrame(
        {
            "slide_id": ["slide_a"],
            "spot_id": ["spot_0"],
            "x": [1.0],
            "y": [2.0],
            "native_patch_px": [8],
        }
    )
    patches.save_patch_arrays("slide_a", values, meta, cfg=cfg)
    training_only = {**cfg, "training": {**cfg["training"], "epochs": 99}}
    assert patches.load_patch_arrays("slide_a", cfg=training_only)[0].shape == values.shape
    changed = {**cfg, "patches": {**cfg["patches"], "output_size": 112}}
    with pytest.raises(ArtifactValidationError, match="stale_fingerprint"):
        patches.load_patch_arrays("slide_a", cfg=changed)
