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

from src.validation import (
    AdmittedRun,
    CohortAdmissionError,
    CohortManifest,
    SlideAdmission,
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
    monkeypatch.setattr(
        runner,
        "preprocess_cohort",
        lambda ids, cfg: (_ for _ in ()).throw(RuntimeError("host-specific detail")),
    )
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
    assert caught.value.manifest.failed[0].slide_id == "oncology_b"
    assert caught.value.manifest.failed[0].reason_code == "source_load_failed"
    assert "host-specific detail" not in caught.value.manifest.canonical_json
    assert "Pipeline complete." not in capsys.readouterr().out
    assert not (tmp_path / "cohort_manifest.json").exists()


def test_runner_partial_remote_outcomes_are_readmitted_once_and_propagated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner()
    cfg = _config(allow_partial=True)
    attempted: list[str] = []
    admission_calls: list[tuple[object, object]] = []
    original_admit = admit_run

    def tracked_admit(raw, *, available_slide_ids=None, failures=None):
        admission_calls.append((available_slide_ids, failures))
        return original_admit(
            raw,
            available_slide_ids=available_slide_ids,
            failures=failures,
        )

    def preprocess(ids, cfg):
        attempted.extend(ids)
        if ids == ["external_a"]:
            raise RuntimeError("source failed")

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
