"""Unit tests for the inference-only foundation-model benchmark arm."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

pytestmark = pytest.mark.offline

from src.foundation import (  # noqa: E402
    FOUNDATION_MODELS,
    extract_frozen_embeddings,
    train_eval_linear_probe,
)
from src.foundation_eval import (  # noqa: E402
    ProbeCandidate,
    nested_loso_classification,
    preprocess_slide_embeddings,
)


class MeanEncoder(torch.nn.Module):
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return batch.mean(dim=(2, 3))


def test_embedding_extraction_uses_model_normalization_and_no_grad() -> None:
    spec = FOUNDATION_MODELS["kaiko_vits16"]
    patches = np.full((3, 3, 8, 8), 0.75, dtype=np.float32)
    embeddings = extract_frozen_embeddings(
        patches, MeanEncoder(), spec, device="cpu", batch_size=2
    )

    assert embeddings.shape == (3, 3)
    # (0.75 - 0.5) / 0.5 = 0.5 for the documented Kaiko transform.
    np.testing.assert_allclose(embeddings, 0.5)


def test_linear_probe_learns_without_mutating_embeddings() -> None:
    rng = np.random.default_rng(7)
    train_x = rng.normal(size=(80, 6))
    val_x = rng.normal(size=(30, 6))
    train_copy = train_x.copy()

    train_labels = pd.DataFrame(
        {
            "tme_class_id": (train_x[:, 0] > 0).astype(int),
            "module_signal": 2.0 * train_x[:, 1] - train_x[:, 2],
        }
    )
    val_labels = pd.DataFrame(
        {
            "tme_class_id": (val_x[:, 0] > 0).astype(int),
            "module_signal": 2.0 * val_x[:, 1] - val_x[:, 2],
        }
    )
    cfg = {
        "labels": {
            "classification_col": "tme_class_id",
            "regression_targets": "modules",
        }
    }

    metrics = train_eval_linear_probe(
        train_x, train_labels, val_x, val_labels, cfg=cfg, seed=0
    )

    np.testing.assert_array_equal(train_x, train_copy)
    assert metrics["balanced_accuracy"] > 0.9
    assert metrics["mean_pearson_r"] > 0.99


def test_registry_contains_independent_pathology_encoders() -> None:
    assert FOUNDATION_MODELS["kaiko_vits16"].embedding_dim == 384
    assert FOUNDATION_MODELS["phikon"].embedding_dim == 768
    assert (
        FOUNDATION_MODELS["kaiko_vits16"].backend != FOUNDATION_MODELS["phikon"].backend
    )


def test_slide_zscore_l2_removes_shift_and_normalizes_rows() -> None:
    rng = np.random.default_rng(11)
    embeddings = rng.normal(loc=8.0, scale=3.0, size=(40, 8))
    transformed = preprocess_slide_embeddings(embeddings, "slide_zscore_l2")
    shifted = preprocess_slide_embeddings(embeddings + 100.0, "slide_zscore_l2")

    np.testing.assert_allclose(np.linalg.norm(transformed, axis=1), 1.0)
    np.testing.assert_allclose(transformed, shifted, atol=1e-12)


def test_nested_loso_returns_one_unseen_slide_per_fold() -> None:
    rng = np.random.default_rng(12)
    slide_data = {}
    class_names = ["tumor_epithelial", "immune_enriched", "stromal"]
    for slide_idx in range(4):
        y = np.tile(np.arange(3), 20)
        embeddings = np.eye(3)[y] + rng.normal(scale=0.05, size=(len(y), 3))
        embeddings += slide_idx * 2.0
        labels = pd.DataFrame(
            {
                "spot_id": [f"s{slide_idx}_{i}" for i in range(len(y))],
                "tme_class": [class_names[i] for i in y],
            }
        )
        slide_data[f"slide_{slide_idx}"] = (embeddings, labels)

    candidates = (ProbeCandidate("slide_zscore_l2", c=0.1),)
    rows, details = nested_loso_classification(
        list(slide_data),
        slide_data,
        task="confident_3class",
        candidates=candidates,
    )

    assert len(rows) == len(details) == 4
    assert rows["held_out_slide"].nunique() == 4
    assert rows["macro_f1"].min() > 0.95
