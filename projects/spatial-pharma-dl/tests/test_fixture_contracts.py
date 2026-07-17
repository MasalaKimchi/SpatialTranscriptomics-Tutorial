"""Contracts for deterministic, fresh, CPU-small verification fixtures."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.offline


def _repo_file_state(path: Path) -> dict[Path, tuple[int, int]]:
    if not path.exists():
        return {}
    return {
        candidate: (candidate.stat().st_size, candidate.stat().st_mtime_ns)
        for candidate in path.rglob("*")
        if candidate.is_file()
    }


def test_anndata_factory_is_deterministic_and_fresh(
    synthetic_anndata_factory: Callable[..., Any],
) -> None:
    first = synthetic_anndata_factory()
    second = synthetic_anndata_factory()
    np.testing.assert_array_equal(first.X, second.X)
    np.testing.assert_array_equal(first.obsm["spatial"], second.obsm["spatial"])
    np.testing.assert_array_equal(
        next(iter(first.uns["spatial"].values()))["images"]["hires"],
        next(iter(second.uns["spatial"].values()))["images"]["hires"],
    )
    assert first.obs_names.tolist() == second.obs_names.tolist()
    assert first.var_names.tolist() == second.var_names.tolist()

    first.X[0, 0] = -999
    first.obs.iloc[0, first.obs.columns.get_loc("slide_id")] = "mutated"
    third = synthetic_anndata_factory()
    assert third.X[0, 0] != -999
    assert third.obs.iloc[0]["slide_id"] == "slide_a"


def test_cohort_factory_preserves_order_and_is_fresh(
    cohort_factory: Callable[..., dict[str, Any]],
) -> None:
    first = cohort_factory()
    second = cohort_factory()
    assert first["slide_ids"] == second["slide_ids"] == [
        "slide_a",
        "slide_b",
        "slide_c",
    ]
    assert first["folds"] == second["folds"]
    pd.testing.assert_frame_equal(first["labels"], second["labels"])
    pd.testing.assert_frame_equal(first["patch_index"], second["patch_index"])

    first["labels"].loc[0, "spot_id"] = "mutated"
    first["slide_ids"].append("mutated")
    third = cohort_factory()
    assert third["labels"].loc[0, "spot_id"] == "slide_a_spot_00"
    assert third["slide_ids"] == ["slide_a", "slide_b", "slide_c"]


def test_all_key_and_fold_adversaries_are_constructed(
    key_adversary_factory: Callable[[], dict[str, Any]],
    fold_adversary_factory: Callable[[], dict[str, Any]],
) -> None:
    keys = key_adversary_factory()
    folds = fold_adversary_factory()
    assert set(keys) == {
        "null",
        "duplicate",
        "unmatched_label",
        "unmatched_patch",
        "cross_slide",
        "shuffled_complete",
        "missing_label_slide",
        "missing_metadata_spot",
        "blank",
        "wrong_type",
        "hostile_label",
        "hostile_metadata",
        "duplicate_metadata",
        "wrong_slide_metadata",
        "value_row_mismatch",
        "reserved_label",
        "reserved_metadata",
    }
    assert set(folds) == {
        "empty",
        "one_slide",
        "single_class",
        "unseen_held_out_class",
    }
    assert keys["null"]["labels"]["spot_id"].isna().any()
    assert keys["duplicate"]["labels"].duplicated(["slide_id", "spot_id"]).any()
    assert folds["empty"]["labels"].empty
    assert folds["unseen_held_out_class"]["held_out"] == "slide_c"

    keys["unmatched_label"]["labels"].loc[0, "spot_id"] = "mutated"
    assert (
        key_adversary_factory()["unmatched_label"]["labels"].loc[0, "spot_id"]
        == "label_only_spot"
    )


def test_all_image_adversaries_are_deterministic_and_fresh(
    image_adversary_factory: Callable[..., dict[str, Any]],
) -> None:
    first = image_adversary_factory()
    second = image_adversary_factory()
    assert set(first) == {
        "grayscale",
        "wrong_channel",
        "invalid_range",
        "all_white",
        "rank_deficient",
        "border",
    }
    for name in {"grayscale", "wrong_channel", "invalid_range", "all_white"}:
        np.testing.assert_array_equal(first[name], second[name])
    np.testing.assert_array_equal(
        first["border"]["coordinates"], second["border"]["coordinates"]
    )

    first["grayscale"][0, 0] = 0
    third = image_adversary_factory()
    np.testing.assert_array_equal(third["grayscale"], second["grayscale"])


def test_artifact_adversaries_are_tmp_isolated_and_fresh(
    artifact_adversary_factory: Callable[[], dict[str, dict[str, Any]]],
    tmp_path: Path,
) -> None:
    first = artifact_adversary_factory()
    assert set(first) == {
        "missing_key",
        "wrong_shape_dtype",
        "object_npz",
        "corrupt_json",
        "row_mismatch",
        "corrupt_bytes",
    }
    for descriptor in first.values():
        assert descriptor["path"].is_relative_to(tmp_path)
        assert descriptor["path"].exists()

    first["missing_key"]["kind"] = "mutated"
    second = artifact_adversary_factory()
    assert second["missing_key"]["kind"] == "missing_key"
    with np.load(second["object_npz"]["path"], allow_pickle=False) as unsafe:
        with pytest.raises(ValueError, match="Object arrays"):
            _ = unsafe["payload"]


def test_factories_do_not_write_repository_data_or_outputs(
    synthetic_anndata_factory: Callable[..., Any],
    cohort_factory: Callable[..., dict[str, Any]],
    key_adversary_factory: Callable[[], dict[str, Any]],
    fold_adversary_factory: Callable[[], dict[str, Any]],
    image_adversary_factory: Callable[..., dict[str, Any]],
    artifact_adversary_factory: Callable[[], dict[str, Any]],
) -> None:
    root = Path(__file__).resolve().parents[3]
    before = {
        name: _repo_file_state(root / name) for name in ("data", "outputs")
    }
    synthetic_anndata_factory()
    cohort_factory()
    key_adversary_factory()
    fold_adversary_factory()
    image_adversary_factory()
    artifact_adversary_factory()
    after = {name: _repo_file_state(root / name) for name in ("data", "outputs")}
    assert after == before
