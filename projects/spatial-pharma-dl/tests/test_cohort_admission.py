"""Offline evidence for deterministic fail-closed cohort admission."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import pandas as pd
import yaml
from requests.exceptions import ConnectionError as RequestsConnectionError

from src.validation import (
    AdmittedRun,
    CohortAdmissionError,
    CohortAdmissionInputError,
    CohortManifest,
    SlideAdmission,
    StageValidationError,
    admit_run,
    resolve_config,
)

pytestmark = pytest.mark.offline

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_pipeline.py"


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


def test_admission_sanitizes_hostile_string_details_before_manifest_return() -> None:
    admitted = admit_run(
        _config(allow_partial=True),
        available_slide_ids={"oncology_b", "oncology_a", "benchmark_a"},
        failures={
            "external_a": (
                "Traceback (most recent call last): /Users/private/token "
                "RuntimeError: secret"
            )
        },
    )

    manifest = admitted.manifest.to_dict()
    assert manifest["failed"] == [
        {
            "cohort": "external",
            "reason": (
                "Source acquisition failed for the configured slide; verify "
                "network access and the public dataset identifier before retrying."
            ),
            "reason_code": "source_load_failed",
            "slide_id": "external_a",
            "status": "failed",
        }
    ]
    assert "traceback" not in admitted.manifest.canonical_json.lower()
    assert "/users/" not in admitted.manifest.canonical_json.lower()


@pytest.mark.parametrize(
    "failures",
    [
        {"unknown_slide": "failed"},
        {1: "failed"},
        {"external_a": Path("/private/source")},
    ],
)
def test_admission_rejects_unknown_ids_and_non_string_failure_details(
    failures: dict[object, object],
) -> None:
    with pytest.raises(CohortAdmissionInputError):
        admit_run(_config(allow_partial=True), failures=failures)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "available",
    [
        "oncology_b",
        [["oncology_b"]],
        ["unknown_slide"],
    ],
)
def test_admission_rejects_malformed_availability_before_normalization(
    available: object,
) -> None:
    with pytest.raises(CohortAdmissionInputError):
        admit_run(
            _config(),
            available_slide_ids=available,  # type: ignore[arg-type]
        )


def test_admission_does_not_hash_non_string_availability_members() -> None:
    class HostileHash:
        def __hash__(self) -> int:
            raise AssertionError("availability validation executed hostile hash")

    with pytest.raises(CohortAdmissionInputError, match="non-string"):
        admit_run(
            _config(),
            available_slide_ids=[HostileHash()],  # type: ignore[list-item]
        )


def test_admission_accepts_duplicate_valid_availability_members() -> None:
    admitted = admit_run(
        _config(),
        available_slide_ids=[
            "oncology_b",
            "oncology_b",
            "oncology_a",
            "external_a",
            "benchmark_a",
        ],
    )
    assert admitted.slide_ids == (
        "oncology_b",
        "oncology_a",
        "external_a",
        "benchmark_a",
    )


def _load_runner():
    spec = importlib.util.spec_from_file_location("cohort_admission_runner", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_import_keeps_heavy_stages_deferred() -> None:
    heavy = {"torch", "torchvision", "sklearn", "skimage", "timm", "transformers"}
    before = set(sys.modules)
    _load_runner()
    loaded = {name.split(".", 1)[0] for name in set(sys.modules) - before}
    assert heavy.isdisjoint(loaded)


def test_local_availability_scan_does_not_create_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src import data

    monkeypatch.setattr(data.st, "project_root", lambda: tmp_path)
    assert data.available_processed_slide_ids(["slide/a", "slide_b"]) == set()
    assert not (tmp_path / "data").exists()


def test_preprocess_cohort_wraps_only_documented_source_acquisition_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src import data

    monkeypatch.setattr(data, "pharma_processed_dir", lambda: tmp_path)
    monkeypatch.setattr(
        data.st,
        "load_visium_sample",
        lambda _sid: (_ for _ in ()).throw(RequestsConnectionError("private host")),
    )

    with pytest.raises(data.SourceAcquisitionError) as caught:
        data.preprocess_cohort(["slide_a"], cfg={"unused": True})
    assert isinstance(caught.value.__cause__, RequestsConnectionError)
    assert "private host" not in str(caught.value)

    monkeypatch.setattr(data.st, "load_visium_sample", lambda _sid: object())
    monkeypatch.setattr(
        data,
        "preprocess_slide",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("writer failed")),
    )
    with pytest.raises(OSError, match="writer failed"):
        data.preprocess_cohort(["slide_a"], cfg={"unused": True})


def test_runner_resolves_quick_and_foundation_overrides_before_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    cfg = _config()
    observed: dict[str, object] = {}

    def inspect_resolution(raw):
        observed["epochs"] = raw["training"]["epochs"]
        observed["patience"] = raw["training"]["patience"]
        observed["foundation"] = raw["foundation"]["enabled"]
        raise RuntimeError("stop after final resolution")

    monkeypatch.setenv("PHARMA_QUICK", "1")
    monkeypatch.setenv("PHARMA_FOUNDATION", "1")
    monkeypatch.setattr(runner, "load_config", lambda: cfg)
    monkeypatch.setattr(runner, "resolve_config", inspect_resolution)
    monkeypatch.setattr(
        runner, "admit_run", lambda *_args, **_kwargs: pytest.fail("admission ran")
    )

    with pytest.raises(RuntimeError, match="final resolution"):
        runner.main()
    assert observed == {"epochs": 2, "patience": 1, "foundation": True}


def test_runner_strict_source_failure_has_no_false_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    runner = _load_runner()
    cfg = _config()
    monkeypatch.delenv("PHARMA_TRAIN_ONLY", raising=False)
    monkeypatch.delenv("PHARMA_QUICK", raising=False)
    monkeypatch.delenv("PHARMA_FOUNDATION", raising=False)
    monkeypatch.setattr(runner, "load_config", lambda: cfg)
    attempted: list[str] = []

    def preprocess(ids, cfg):
        attempted.extend(ids)
        if ids == ["oncology_b"]:
            raise runner.SourceAcquisitionError("host-specific detail")

    monkeypatch.setattr(runner, "preprocess_cohort", preprocess)
    monkeypatch.setattr(
        runner, "_load_stages", lambda: (_ for _ in ()).throw(AssertionError("stage"))
    )
    monkeypatch.setattr(
        runner,
        "pharma_outputs_dir",
        lambda: (_ for _ in ()).throw(AssertionError("writer")),
    )

    with pytest.raises(CohortAdmissionError) as caught:
        runner.main()
    manifest = caught.value.manifest
    assert attempted == ["oncology_b"]
    assert [item.slide_id for item in manifest.configured] == [
        "oncology_b",
        "oncology_a",
        "external_a",
        "benchmark_a",
    ]
    assert manifest.included == ()
    assert [item.slide_id for item in manifest.skipped] == [
        "oncology_b",
        "oncology_a",
        "external_a",
        "benchmark_a",
    ]
    assert [item.reason_code for item in manifest.skipped] == [
        "source_load_failed",
        "source_not_attempted",
        "source_not_attempted",
        "source_not_attempted",
    ]
    assert [item.slide_id for item in manifest.failed] == ["oncology_b"]
    assert manifest.failed[0].reason_code == "source_load_failed"
    assert "host-specific detail" not in manifest.canonical_json
    assert isinstance(caught.value.__cause__, runner.SourceAcquisitionError)
    assert "Pipeline complete." not in capsys.readouterr().out
    assert not (tmp_path / "cohort_manifest.json").exists()


def test_runner_strict_failure_forbids_later_preprocessing_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    cfg = _config()
    attempted: list[str] = []

    def preprocess(ids, cfg):
        attempted.extend(ids)
        if ids == ["oncology_b"]:
            raise runner.SourceAcquisitionError("first source failed")
        raise AssertionError("later preprocessing or cache publication was reached")

    monkeypatch.setattr(runner, "preprocess_cohort", preprocess)

    with pytest.raises(CohortAdmissionError) as caught:
        runner._curate_sources(
            cfg,
            ["oncology_b", "oncology_a", "external_a", "benchmark_a"],
        )

    assert attempted == ["oncology_b"]
    assert [item.reason_code for item in caught.value.manifest.skipped] == [
        "source_load_failed",
        "source_not_attempted",
        "source_not_attempted",
        "source_not_attempted",
    ]


def test_runner_does_not_convert_programming_or_storage_defects_to_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    cfg = _config(allow_partial=True)
    attempted: list[str] = []

    def broken_preprocess(ids, cfg):
        attempted.extend(ids)
        raise OSError("disk write failed")

    monkeypatch.setattr(runner, "preprocess_cohort", broken_preprocess)
    monkeypatch.setattr(
        runner,
        "admit_run",
        lambda *_args, **_kwargs: pytest.fail("storage error became admission policy"),
    )

    with pytest.raises(OSError, match="disk write failed"):
        runner._curate_sources(cfg, ["oncology_b", "oncology_a"])
    assert attempted == ["oncology_b"]


def test_runner_partial_remote_outcomes_are_readmitted_once_and_propagated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner()
    cfg = _config(allow_partial=True)
    attempted: list[str] = []
    admission_calls: list[tuple[object, object]] = []
    original_admit = admit_run

    def tracked_admit(
        raw,
        *,
        available_slide_ids=None,
        failures=None,
        unattempted_slide_ids=None,
    ):
        admission_calls.append((available_slide_ids, failures))
        return original_admit(
            raw,
            available_slide_ids=available_slide_ids,
            failures=failures,
            unattempted_slide_ids=unattempted_slide_ids,
        )

    def preprocess(ids, cfg):
        attempted.extend(ids)
        if ids == ["external_a"]:
            raise runner.SourceAcquisitionError("source failed")

    seen: dict[str, object] = {}

    labels = pd.DataFrame(
        {
            "slide_id": ["oncology_b", "oncology_a", "benchmark_a"],
            "tme_class_id": [0, 1, 0],
        }
    )
    report = pd.DataFrame(
        {
            "model": ["cnn", "rf"],
            "fold": [0, 0],
            "val_slide": ["oncology_b", "oncology_b"],
            "balanced_accuracy": [0.5, 0.5],
            "mean_pearson_r": [0.1, 0.1],
        }
    )

    def summary_stage(ids):
        seen["summary_ids"] = ids
        return pd.DataFrame({"slide_id": ids})

    def label_stage(ids, cfg):
        seen["label_ids"] = ids
        return labels

    def stain_stage(ids, cfg):
        seen["stain_ids"] = ids
        return object()

    def patch_stage(ids, ref_stain, cfg):
        seen["patch_ids"] = ids

    def benchmark_stage(ids, selected_labels, cfg):
        seen["benchmark_ids"] = ids
        seen["benchmark_label_ids"] = list(selected_labels["slide_id"])
        return tmp_path / "report.csv", []

    stages = SimpleNamespace(
        st=SimpleNamespace(set_seeds=lambda seed: seen.setdefault("seed", seed)),
        cohort_summary=summary_stage,
        build_labels_cohort=label_stage,
        fit_reference_stain=stain_stage,
        build_patch_cohort=patch_stage,
        save_patch_index=lambda labels: tmp_path / "patch_index.parquet",
        patch_cache_path=lambda sid, cfg: tmp_path / "unused",
        run_and_save_benchmark=benchmark_stage,
        evaluate_fold=lambda result: result,
        pd=SimpleNamespace(read_csv=lambda path: report),
    )
    monkeypatch.delenv("PHARMA_TRAIN_ONLY", raising=False)
    monkeypatch.setattr(runner, "load_config", lambda: cfg)
    monkeypatch.setattr(runner, "resolve_config", resolve_config)
    monkeypatch.setattr(runner, "admit_run", tracked_admit)
    monkeypatch.setattr(runner, "preprocess_cohort", preprocess)
    monkeypatch.setattr(runner, "_load_stages", lambda: stages)
    monkeypatch.setattr(runner, "pharma_outputs_dir", lambda: tmp_path)

    runner.main()

    assert attempted == ["oncology_b", "oncology_a", "external_a", "benchmark_a"]
    assert len(admission_calls) == 2
    assert admission_calls[0] == (None, None)
    assert admission_calls[1][0] == ["oncology_b", "oncology_a", "benchmark_a"]
    assert list(admission_calls[1][1]) == ["external_a"]
    assert seen["summary_ids"] == ["oncology_b", "oncology_a", "benchmark_a"]
    assert seen["label_ids"] == seen["patch_ids"]
    assert seen["stain_ids"] == ["oncology_b", "oncology_a"]
    assert seen["benchmark_ids"] == ["oncology_b", "oncology_a"]
    manifest = json.loads((tmp_path / "cohort_manifest.json").read_text())
    assert [item["slide_id"] for item in manifest["included"]] == seen["label_ids"]


@pytest.mark.parametrize("helper", ["summary", "labels", "stain", "patches"])
def test_downstream_helpers_propagate_first_missing_admitted_member(
    helper: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from src import data, labels, patches

    calls: list[str] = []

    def missing_first(slide_id: str, *_args):
        calls.append(slide_id)
        if slide_id == "missing":
            raise FileNotFoundError("admitted slide disappeared")
        raise AssertionError("later member must not be processed")

    monkeypatch.setattr(data, "load_slide", missing_first)
    monkeypatch.setattr(labels, "pharma_outputs_dir", lambda: tmp_path)
    monkeypatch.setattr(labels, "build_labels_for_slide", missing_first)

    cfg = _config()
    with pytest.raises(FileNotFoundError, match="admitted slide disappeared"):
        if helper == "summary":
            data.cohort_summary(["missing", "later"])
        elif helper == "labels":
            labels.build_labels_cohort(["missing", "later"], cfg)
        elif helper == "stain":
            patches.fit_reference_stain(["missing", "later"], cfg)
        else:
            patches.build_patch_cohort(
                ["missing", "later"],
                ref_stain=object(),
                cfg=cfg,
            )

    assert calls == ["missing"]
    output = capsys.readouterr().out
    assert "Skipping" not in output
    assert "Saved" not in output


def test_explicit_empty_values_are_not_replaced_by_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src import data, labels, patches

    def forbidden_default(*_args, **_kwargs):
        raise AssertionError("explicit empty value reloaded defaults")

    monkeypatch.setattr(data, "load_config", forbidden_default)
    monkeypatch.setattr(labels, "load_config", forbidden_default)
    monkeypatch.setattr(patches, "load_config", forbidden_default)

    with pytest.raises(StageValidationError, match="cohort_preprocessing"):
        data.preprocess_cohort([], cfg={})
    with pytest.raises(StageValidationError, match="cohort_summary"):
        data.cohort_summary([])
    assert labels.tme_class_names({"labels": {"tme_classes": []}}) == []
    marker = object()
    with pytest.raises(StageValidationError, match="patch_cohort"):
        patches.build_patch_cohort([], ref_stain=marker, cfg={})
