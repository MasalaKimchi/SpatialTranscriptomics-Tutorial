"""Regression tests for lightweight imports and lifecycle refactors."""

from __future__ import annotations

import sys

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.offline


def test_package_exports_are_lazy_and_scanpy_independent() -> None:
    import src

    assert "scanpy" not in sys.modules
    assert src.load_config()["experiment"] == "v2_remediation"
    assert "scanpy" not in sys.modules
    assert callable(src.run_loso_benchmark)
    assert "scanpy" not in sys.modules


def test_notebook_patches_are_idempotent() -> None:
    from scripts.patch_notebooks import patch_00

    notebook = {"cells": [], "metadata": {}}
    patch_00(notebook)
    first_cells = list(notebook["cells"])

    patch_00(notebook)

    assert notebook["cells"] == first_cells


class _TinyCamModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.gradcam_layer = self.conv
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.classifier = torch.nn.Linear(4, 2)
        self.regressor = torch.nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.relu(self.conv(x))
        encoded = self.pool(features).flatten(1)
        return self.classifier(encoded), self.regressor(encoded)


def test_gradcam_removes_temporary_hooks() -> None:
    from src.eval import grad_cam_for_patch

    model = _TinyCamModel()
    patch = np.full((3, 8, 8), 0.5, dtype=np.float32)
    hooks_before = (len(model.conv._forward_hooks), len(model.conv._backward_hooks))

    cam = grad_cam_for_patch(model, patch)

    hooks_after = (len(model.conv._forward_hooks), len(model.conv._backward_hooks))
    assert hooks_after == hooks_before
    assert cam.shape == (8, 8)
    assert np.isfinite(cam).all()
