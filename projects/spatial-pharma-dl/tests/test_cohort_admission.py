"""Offline evidence for deterministic fail-closed cohort admission."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
import yaml

from src.validation import (
    AdmittedRun,
    CohortAdmissionError,
    CohortManifest,
    SlideAdmission,
    admit_run,
)

pytestmark = pytest.mark.offline

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def _config(*, allow_partial: bool = False) -> dict[str, object]:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    cfg["cohorts"] = {
        "oncology": ["oncology_b", "oncology_a"],
        "external": ["external_a"],
        "benchmark": ["benchmark_a"],
    }
    cfg["cohort_policy"]["allow_partial"] = allow_partial
    return cfg


def test_admission_records_are_frozen_slotted_and_ordered() -> None:
    admitted = admit_run(_config())
    assert isinstance(admitted, AdmittedRun)
    assert admitted.slide_ids == (
        "oncology_b",
        "oncology_a",
        "external_a",
        "benchmark_a",
    )
    assert admitted.manifest.schema_version == "cohort-manifest-v1"
    assert all(dataclasses.is_dataclass(cls) for cls in (SlideAdmission, CohortManifest))
    with pytest.raises(dataclasses.FrozenInstanceError):
        admitted.manifest.allow_partial = True


def test_strict_admission_aggregates_every_missing_and_failed_member() -> None:
    with pytest.raises(CohortAdmissionError) as caught:
        admit_run(
            _config(),
            available_slide_ids={"oncology_b", "benchmark_a"},
            failures={"external_a": "Remote source could not be loaded."},
        )

    manifest = caught.value.manifest
    assert [item.slide_id for item in manifest.skipped] == [
        "oncology_a",
        "external_a",
    ]
    assert [item.slide_id for item in manifest.failed] == ["external_a"]
    assert caught.value.unavailable_slide_ids == ("oncology_a", "external_a")


def test_partial_admission_is_explicit_complete_and_order_stable() -> None:
    admitted = admit_run(
        _config(allow_partial=True),
        available_slide_ids={"benchmark_a", "oncology_b", "oncology_a"},
        failures={"external_a": "Remote source could not be loaded."},
    )

    assert admitted.slide_ids == ("oncology_b", "oncology_a", "benchmark_a")
    manifest = admitted.manifest
    assert [item.slide_id for item in manifest.configured] == [
        "oncology_b",
        "oncology_a",
        "external_a",
        "benchmark_a",
    ]
    assert [(item.status, item.reason_code) for item in manifest.skipped] == [
        ("skipped", "source_load_failed")
    ]
    assert [(item.status, item.reason_code) for item in manifest.failed] == [
        ("failed", "source_load_failed")
    ]
    assert manifest.skipped[0].reason != manifest.failed[0].reason


@pytest.mark.parametrize(
    "available",
    [set(), {"oncology_b", "external_a", "benchmark_a"}],
)
def test_partial_admission_rejects_unusable_cohorts(available: set[str]) -> None:
    with pytest.raises(CohortAdmissionError):
        admit_run(_config(allow_partial=True), available_slide_ids=available)


def test_manifest_json_is_deterministic_safe_and_mutation_isolated() -> None:
    first = admit_run(_config()).manifest
    second = admit_run(_config()).manifest
    assert first.canonical_json == second.canonical_json
    decoded = json.loads(first.canonical_json)
    assert decoded["schema_version"] == "cohort-manifest-v1"
    assert not any(
        token in first.canonical_json.lower()
        for token in ("timestamp", "traceback", "/users/", "nan", "infinity")
    )
    mutable = first.to_dict()
    pristine = first.to_dict()
    mutable["included"].append({"slide_id": "mutation"})
    assert mutable != pristine
