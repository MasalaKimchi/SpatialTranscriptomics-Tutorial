"""Offline evidence for adaptive preprocessing dimensions and provenance."""

from __future__ import annotations

import json

import pytest

from src.validation import (
    PreprocessingValidationError,
    finalize_preprocessing_resolution,
    resolve_post_qc_preprocessing,
)

pytestmark = pytest.mark.offline


def _resolve(**overrides):
    values = {
        "slide_id": "slide_a",
        "input_spots": 12,
        "input_genes": 10,
        "after_filter_cells_spots": 10,
        "after_filter_cells_genes": 10,
        "after_filter_genes_spots": 10,
        "after_filter_genes_genes": 8,
        "post_qc_spots": 8,
        "post_qc_genes": 8,
        "requested_hvg": 6,
        "requested_pcs": 4,
        "requested_neighbors": 5,
        "requested_graph_pcs": 3,
    }
    values.update(overrides)
    return resolve_post_qc_preprocessing(**values)


def test_resolve_and_finalize_accept_legal_requests() -> None:
    resolution = finalize_preprocessing_resolution(_resolve(), actual_hvgs=6)
    record = resolution.to_dict()

    assert record["resolved"] == {
        "graph_pcs": 3,
        "hvg_call": 6,
        "neighbors": 5,
        "pca": 4,
    }
    assert set(record["reasons"].values()) == {"requested_value_accepted"}
    assert record["counts"]["actual_hvgs"] == 6


def test_independent_and_joint_caps_use_actual_hvgs() -> None:
    first = _resolve(
        requested_hvg=100,
        requested_pcs=100,
        requested_neighbors=100,
        requested_graph_pcs=100,
    )
    resolution = finalize_preprocessing_resolution(first, actual_hvgs=3)
    record = resolution.to_dict()

    assert record["resolved"] == {
        "graph_pcs": 2,
        "hvg_call": 8,
        "neighbors": 7,
        "pca": 2,
    }
    assert record["reasons"] == {
        "graph_pcs": "requested_value_capped_to_resolved_pcs",
        "hvg": "requested_value_capped_to_post_qc_genes",
        "neighbors": "requested_value_capped_to_spot_limit",
        "pca": "requested_value_capped_to_rank_limit",
    }


@pytest.mark.parametrize(
    ("overrides", "stage", "reason"),
    [
        ({"post_qc_spots": 0}, "post_qc", "insufficient_post_qc_spots"),
        ({"post_qc_spots": 1}, "post_qc", "insufficient_post_qc_spots"),
        ({"post_qc_spots": 2}, "post_qc", "insufficient_post_qc_spots"),
        ({"post_qc_genes": 0}, "post_qc", "insufficient_post_qc_genes"),
        ({"post_qc_genes": 1}, "post_qc", "insufficient_post_qc_genes"),
    ],
)
def test_nonviable_post_qc_counts_raise_structured_error(overrides, stage, reason) -> None:
    with pytest.raises(PreprocessingValidationError) as caught:
        _resolve(**overrides)
    assert caught.value.stage == stage
    assert caught.value.reason_code == reason
    assert type(caught.value.counts) is dict
    assert type(caught.value.requested) is dict
    assert type(caught.value.guidance) is str


@pytest.mark.parametrize("actual_hvgs", [0, 1])
def test_nonviable_actual_hvgs_fail_before_pca(actual_hvgs: int) -> None:
    with pytest.raises(PreprocessingValidationError) as caught:
        finalize_preprocessing_resolution(_resolve(), actual_hvgs=actual_hvgs)
    assert caught.value.stage == "post_hvg"
    assert caught.value.reason_code == "insufficient_actual_hvgs"


def test_resolution_is_canonical_and_returns_mutation_isolated_primitives() -> None:
    first = finalize_preprocessing_resolution(_resolve(), actual_hvgs=6)
    second = finalize_preprocessing_resolution(_resolve(), actual_hvgs=6)
    assert first.canonical_json == second.canonical_json
    assert json.dumps(first.to_dict(), allow_nan=False)
    mutated = first.to_dict()
    mutated["counts"]["actual_hvgs"] = 999
    assert first.to_dict()["counts"]["actual_hvgs"] == 6
    assert all(
        type(value) is int
        for section in ("counts", "exclusions", "requested", "resolved")
        for value in first.to_dict()[section].values()
    )


@pytest.mark.parametrize("field", ["post_qc_spots", "requested_hvg"])
def test_resolver_rejects_non_exact_primitive_before_arithmetic(field: str) -> None:
    class HostileInt(int):
        def __lt__(self, _other):
            raise AssertionError("hostile comparison executed")

        def bit_length(self):
            raise AssertionError("hostile integer hook executed")

    with pytest.raises(PreprocessingValidationError) as caught:
        _resolve(**{field: HostileInt(4)})
    assert caught.value.reason_code == "invalid_preprocessing_input"
