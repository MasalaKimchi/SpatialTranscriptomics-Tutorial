"""Offline evidence for exact compound identity and complete alignment."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from src.identity import (
    IdentityValidationError,
    align_labels_with_metadata,
    validate_anndata_spot_identity,
)

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
