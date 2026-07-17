"""Offline evidence for exact compound identity and complete alignment."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.identity import (
    IdentityValidationError,
    align_labels_with_metadata,
    validate_anndata_spot_identity,
)
from src import labels as label_module
from src import patches as patch_module
from src import foundation as foundation_module
from src import train as train_module
from src.foundation import FOUNDATION_MODELS

pytestmark = pytest.mark.offline


def _align(case, **kwargs):
    return align_labels_with_metadata(
        case["labels"],
        case["patch_index"],
        stage="test_alignment",
        **kwargs,
    )


@pytest.mark.parametrize(
    ("case_name", "code"),
    [
        ("missing_label_slide", "missing_column"),
        ("missing_metadata_spot", "missing_column"),
        ("null", "invalid_type"),
        ("blank", "blank_value"),
        ("wrong_type", "invalid_type"),
        ("duplicate", "duplicate_key"),
        ("duplicate_metadata", "duplicate_key"),
        ("reserved_label", "reserved_column"),
        ("reserved_metadata", "reserved_column"),
    ],
)
def test_key_defects_are_structured(key_adversary_factory, case_name, code):
    with pytest.raises(IdentityValidationError) as caught:
        _align(key_adversary_factory()[case_name])

    assert code in {issue.code for issue in caught.value.issues}
    assert isinstance(caught.value, ValueError)
    assert all(issue.count >= 1 for issue in caught.value.issues)


@pytest.mark.parametrize("case_name", ["hostile_label", "hostile_metadata"])
def test_hostile_key_subclasses_execute_no_hooks(key_adversary_factory, case_name):
    with pytest.raises(IdentityValidationError, match="invalid_type"):
        _align(key_adversary_factory()[case_name])


def test_value_row_mismatch_is_rejected(key_adversary_factory):
    case = key_adversary_factory()["value_row_mismatch"]
    with pytest.raises(IdentityValidationError) as caught:
        _align(case, value_row_count=case["value_row_count"])

    issue = caught.value.issues[0]
    assert issue.code == "cardinality_mismatch"
    assert issue.side == "values"


def test_wrong_slide_metadata_is_rejected(key_adversary_factory):
    case = key_adversary_factory()["wrong_slide_metadata"]
    with pytest.raises(IdentityValidationError) as caught:
        _align(case, expected_slide_id="slide_a")

    assert "wrong_slide" in {issue.code for issue in caught.value.issues}


def test_cross_slide_mismatch_is_distinguished(key_adversary_factory):
    case = key_adversary_factory()["cross_slide"]
    with pytest.raises(IdentityValidationError) as caught:
        _align(case)

    assert "cross_slide" in {issue.code for issue in caught.value.issues}


def test_diagnostics_are_deterministic_and_bounded():
    labels = pd.DataFrame(
        {
            "slide_id": ["slide_a"] * 7,
            "spot_id": [f"label_{index}" for index in range(7)],
        }
    )
    metadata = pd.DataFrame(
        {
            "slide_id": ["slide_a"] * 7,
            "spot_id": [f"meta_{index}" for index in reversed(range(7))],
        }
    )

    messages = []
    issues = []
    for frame in (metadata, metadata.iloc[::-1].reset_index(drop=True)):
        with pytest.raises(IdentityValidationError) as caught:
            align_labels_with_metadata(labels, frame, stage="stable")
        messages.append(str(caught.value))
        issues.append(caught.value.issues)

    assert messages[0] == messages[1]
    assert issues[0] == issues[1]
    assert all(len(issue.sample_keys) <= 5 for issue in issues[0])


def test_shuffled_complete_alignment_preserves_metadata_order_and_pairing(
    key_adversary_factory,
):
    case = key_adversary_factory()["shuffled_complete"]
    aligned = _align(case)

    assert aligned["spot_id"].tolist() == case["patch_index"]["spot_id"].tolist()
    expected = case["labels"].set_index(["slide_id", "spot_id"])
    for row in aligned.itertuples(index=False):
        source = expected.loc[(row.slide_id, row.spot_id)]
        assert row.tme_class_id == source["tme_class_id"]
        assert row.module_signal == source["module_signal"]
    assert aligned["_patch_source_row"].tolist() == list(range(len(aligned)))
    assert sorted(aligned["_label_source_row"].tolist()) == list(range(len(aligned)))


def test_repeated_barcode_across_slides_remains_compound_scoped():
    labels = pd.DataFrame(
        {
            "slide_id": ["slide_a", "slide_b"],
            "spot_id": ["shared", "shared"],
            "target": [1, 2],
        }
    )
    metadata = labels[["slide_id", "spot_id"]].iloc[::-1].reset_index(drop=True)

    aligned = align_labels_with_metadata(labels, metadata, stage="compound")

    assert aligned["target"].tolist() == [2, 1]


def test_anndata_identity_rejects_invalid_names_before_row_construction():
    adata = SimpleNamespace(obs_names=pd.Index(["spot_a", None], dtype=object))

    with pytest.raises(IdentityValidationError, match="invalid_type"):
        validate_anndata_spot_identity(adata, "slide_a", stage="producer")


def test_public_alignment_facade_preserves_dataframe_contract(
    key_adversary_factory,
):
    case = key_adversary_factory()["shuffled_complete"]

    aligned = label_module.align_labels_with_patches(
        case["labels"], case["patch_index"]
    )

    assert isinstance(aligned, pd.DataFrame)
    assert aligned["spot_id"].tolist() == case["patch_index"]["spot_id"].tolist()


def test_label_producer_rejects_anndata_identity_before_scientific_work(monkeypatch):
    invalid = SimpleNamespace(obs_names=pd.Index(["spot_a", None], dtype=object))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("label scientific seam was reached")

    fake_scanpy = SimpleNamespace(
        tl=SimpleNamespace(rank_genes_groups=forbidden),
        get=SimpleNamespace(rank_genes_groups_df=forbidden),
    )
    monkeypatch.setitem(__import__("sys").modules, "scanpy", fake_scanpy)
    monkeypatch.setattr(label_module, "load_slide", lambda _slide_id: invalid)
    monkeypatch.setattr(label_module, "tme_class_to_id", forbidden)
    monkeypatch.setattr(label_module, "marker_genes_for_slide", forbidden)
    monkeypatch.setattr(label_module, "compute_module_scores", forbidden)
    monkeypatch.setattr(label_module.st, "genes_present", forbidden)

    with pytest.raises(IdentityValidationError, match="invalid_type"):
        label_module.build_labels_for_slide("slide_a", cfg={"seed": 0})


def test_patch_producer_rejects_anndata_identity_before_coordinates_or_stack(
    monkeypatch,
):
    invalid = SimpleNamespace(obs_names=pd.Index(["spot_a", None], dtype=object))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("patch scientific seam was reached")

    monkeypatch.setattr(patch_module, "coords_hires", forbidden)
    monkeypatch.setattr(patch_module.st, "get_image", forbidden)
    monkeypatch.setattr(patch_module.np, "stack", forbidden)

    with pytest.raises(IdentityValidationError, match="invalid_type"):
        patch_module._extract_spot_patches(
            invalid,
            "slide_a",
            np.eye(2, 3),
            {"patches": {}},
        )


def test_ordinary_consumer_uses_patch_source_order(monkeypatch):
    labels = pd.DataFrame(
        {
            "slide_id": ["slide_a", "slide_a", "slide_b"],
            "spot_id": ["p1", "p0", "other"],
            "target": [11, 10, 99],
        }
    )
    metadata = pd.DataFrame(
        {"slide_id": ["slide_a", "slide_a"], "spot_id": ["p0", "p1"]}
    )
    patch_values = np.asarray([[[[10.0]]], [[[11.0]]]], dtype=np.float32)
    monkeypatch.setattr(
        train_module,
        "load_patch_arrays",
        lambda *_args, **_kwargs: (patch_values, metadata),
    )

    patches, aligned = train_module.load_slide_patches(
        "slide_a", labels, cfg={}
    )

    np.testing.assert_array_equal(patches, patch_values)
    assert aligned["spot_id"].tolist() == ["p0", "p1"]
    assert aligned["target"].tolist() == [10, 11]
    assert aligned["_patch_source_row"].tolist() == [0, 1]


def test_ordinary_consumer_rejects_cardinality_before_indexing(monkeypatch):
    labels = pd.DataFrame(
        {"slide_id": ["slide_a"], "spot_id": ["p0"], "target": [1]}
    )
    metadata = labels[["slide_id", "spot_id"]].copy()
    patch_values = np.zeros((2, 1, 1, 1), dtype=np.float32)
    monkeypatch.setattr(
        train_module,
        "load_patch_arrays",
        lambda *_args, **_kwargs: (patch_values, metadata),
    )

    with pytest.raises(IdentityValidationError, match="cardinality_mismatch"):
        train_module.load_slide_patches("slide_a", labels, cfg={})


def _foundation_test_config() -> dict:
    return {
        "foundation": {
            "model": "kaiko_vits16",
            "cache": True,
            "batch_size": 2,
        },
        "patches": {"version": "v1"},
    }


def test_foundation_cache_hit_rejects_subset_before_patch_or_encoder(
    monkeypatch, tmp_path
):
    cache_path = tmp_path / "cached.npz"
    np.savez_compressed(
        cache_path,
        embeddings=np.zeros((1, 3), dtype=np.float32),
        spot_ids=np.asarray(["p0"], dtype=np.str_),
    )
    labels = pd.DataFrame(
        {
            "slide_id": ["slide_a", "slide_a"],
            "spot_id": ["p0", "p1"],
            "target": [0, 1],
        }
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("foundation fallback or encoder seam was reached")

    monkeypatch.setattr(foundation_module, "_resolved_config_arg", lambda cfg: cfg)
    monkeypatch.setattr(
        foundation_module, "_embedding_cache_path", lambda *_args: cache_path
    )
    monkeypatch.setattr(foundation_module, "load_slide_patches", forbidden)
    monkeypatch.setattr(foundation_module, "load_frozen_encoder", forbidden)

    with pytest.raises(IdentityValidationError, match="label_only"):
        foundation_module.load_or_extract_slide_embeddings(
            "slide_a", labels, cfg=_foundation_test_config()
        )


def test_foundation_cache_hit_preserves_cache_order(monkeypatch, tmp_path):
    cache_path = tmp_path / "cached.npz"
    embeddings = np.asarray([[11.0, 1.0], [10.0, 0.0]], dtype=np.float32)
    np.savez_compressed(
        cache_path,
        embeddings=embeddings,
        spot_ids=np.asarray(["p1", "p0"], dtype=np.str_),
    )
    labels = pd.DataFrame(
        {
            "slide_id": ["slide_a", "slide_a", "slide_b"],
            "spot_id": ["p0", "p1", "other"],
            "target": [10, 11, 99],
        }
    ).iloc[::-1].reset_index(drop=True)
    monkeypatch.setattr(foundation_module, "_resolved_config_arg", lambda cfg: cfg)
    monkeypatch.setattr(
        foundation_module, "_embedding_cache_path", lambda *_args: cache_path
    )

    actual, aligned = foundation_module.load_or_extract_slide_embeddings(
        "slide_a", labels, cfg=_foundation_test_config()
    )

    np.testing.assert_array_equal(actual, embeddings)
    assert aligned["spot_id"].tolist() == ["p1", "p0"]
    assert aligned["target"].tolist() == [11, 10]


def test_foundation_cache_miss_checks_extracted_cardinality_before_write(
    monkeypatch, tmp_path
):
    cache_path = tmp_path / "missing.npz"
    labels = pd.DataFrame(
        {
            "slide_id": ["slide_a", "slide_a"],
            "spot_id": ["p0", "p1"],
            "target": [10, 11],
            "_label_source_row": [0, 1],
            "_patch_source_row": [0, 1],
        }
    )
    patches = np.zeros((2, 3, 4, 4), dtype=np.float32)
    spec = FOUNDATION_MODELS["kaiko_vits16"]
    monkeypatch.setattr(foundation_module, "_resolved_config_arg", lambda cfg: cfg)
    monkeypatch.setattr(
        foundation_module, "_embedding_cache_path", lambda *_args: cache_path
    )
    monkeypatch.setattr(
        foundation_module,
        "load_slide_patches",
        lambda *_args, **_kwargs: (patches, labels),
    )
    monkeypatch.setattr(
        foundation_module,
        "extract_frozen_embeddings",
        lambda *_args, **_kwargs: np.zeros((1, spec.embedding_dim), dtype=np.float32),
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("cache write was reached")

    monkeypatch.setattr(foundation_module.np, "savez_compressed", forbidden)

    with pytest.raises(IdentityValidationError, match="cardinality_mismatch"):
        foundation_module.load_or_extract_slide_embeddings(
            "slide_a",
            labels.drop(columns=["_label_source_row", "_patch_source_row"]),
            cfg=_foundation_test_config(),
            encoder_bundle=(object(), "cpu", spec),
        )
