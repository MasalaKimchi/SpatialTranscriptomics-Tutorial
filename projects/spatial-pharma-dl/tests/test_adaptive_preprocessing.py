"""Offline evidence for adaptive preprocessing dimensions and provenance."""

from __future__ import annotations

import json
import importlib.util
import sys
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import yaml

from src import data
from src.identity import IdentityValidationError
from src.validation import (
    PreprocessingManifest,
    PreprocessingValidationError,
    finalize_preprocessing_resolution,
    resolve_post_qc_preprocessing,
)

pytestmark = pytest.mark.offline

CONFIG_PATH = __import__("pathlib").Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
RUNNER_PATH = CONFIG_PATH.parents[1] / "scripts" / "run_pipeline.py"


def _valid_config():
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    cfg["preprocessing"].update(min_counts=1, min_cells=1, max_pct_mito=100)
    return cfg


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


def _record(slide_id: str):
    return finalize_preprocessing_resolution(
        _resolve(slide_id=slide_id), actual_hvgs=6
    ).to_dict()


def _load_runner():
    spec = importlib.util.spec_from_file_location("adaptive_runner", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def _fake_scanpy(call_log: list[tuple[str, dict[str, object]]], actual_hvgs: int = 4):
    def record(name, mutation=None):
        def call(adata, **kwargs):
            call_log.append((name, kwargs.copy()))
            if mutation is not None:
                mutation(adata, kwargs)
        return call

    def qc(adata, _kwargs):
        adata.obs["pct_counts_mt"] = np.zeros(adata.n_obs)

    def hvg(adata, _kwargs):
        mask = np.zeros(adata.n_vars, dtype=bool)
        mask[: min(actual_hvgs, adata.n_vars)] = True
        adata.var["highly_variable"] = mask

    def pca(adata, kwargs):
        adata.obsm["X_pca"] = np.zeros((adata.n_obs, kwargs["n_comps"]))
        adata.uns["pca"] = {"variance": np.ones(kwargs["n_comps"])}

    pp = SimpleNamespace(
        calculate_qc_metrics=record("qc", qc),
        filter_cells=record("filter_cells"),
        filter_genes=record("filter_genes"),
        normalize_total=record("normalize_total"),
        log1p=record("log1p"),
        highly_variable_genes=record("hvg", hvg),
        scale=record("scale"),
        pca=record("pca", pca),
        neighbors=record("neighbors"),
    )
    return SimpleNamespace(pp=pp, tl=SimpleNamespace(umap=record("umap")))


def test_scanpy_orchestration_passes_finalized_values_once(monkeypatch, synthetic_anndata_factory) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setitem(sys.modules, "scanpy", _fake_scanpy(calls, actual_hvgs=4))
    monkeypatch.setattr(data.st, "set_seeds", lambda seed: calls.append(("seed", {"seed": seed})))
    monkeypatch.setattr(data.st, "run_leiden", lambda adata, **kwargs: (calls.append(("leiden", kwargs)), adata.obs.__setitem__("clusters", "0")))
    cfg = _valid_config()
    cfg["preprocessing"].update(
        n_top_genes_hvg=100,
        n_pcs=50,
        n_neighbors=50,
        n_pcs_neighbors=30,
    )

    result = data.preprocess_slide(
        synthetic_anndata_factory(n_spots=8, n_genes=8), "slide_a", cfg=cfg, seed=7
    )

    names = [name for name, _ in calls]
    assert names == [
        "seed", "qc", "filter_cells", "filter_genes", "normalize_total", "log1p",
        "hvg", "scale", "pca", "neighbors", "umap", "leiden",
    ]
    by_name = {name: kwargs for name, kwargs in calls}
    assert by_name["hvg"]["n_top_genes"] == 8
    assert by_name["pca"]["n_comps"] == 3
    assert by_name["neighbors"]["n_neighbors"] == 7
    assert by_name["neighbors"]["n_pcs"] == 3
    assert result.uns["spatial_pharma_preprocessing"]["counts"]["actual_hvgs"] == 4
    assert {"counts", "raw", "X_pca", "pca", "clusters"} <= {
        *result.layers.keys(), *result.obsm.keys(), *result.uns.keys(), *result.obs.columns,
        "raw" if result.raw is not None else "",
    }


def test_config_and_identity_guard_before_scanpy_import_copy_or_seed(monkeypatch, synthetic_anndata_factory) -> None:
    class CopyForbidden:
        obs_names = pd.Index(["spot_a", None], dtype=object)

        def copy(self):
            raise AssertionError("copy reached")

    monkeypatch.delitem(sys.modules, "scanpy", raising=False)
    real_import = __import__("builtins").__import__

    def guarded_import(name, *args, **kwargs):
        if name == "scanpy":
            raise AssertionError("scanpy import reached")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    monkeypatch.setattr(data.st, "set_seeds", lambda *_args: (_ for _ in ()).throw(AssertionError("seed reached")))

    bad_cfg = _valid_config()
    bad_cfg["preprocessing"]["n_pcs"] = 0
    with pytest.raises(Exception):
        data.preprocess_slide(synthetic_anndata_factory(), "slide_a", cfg=bad_cfg)
    with pytest.raises(IdentityValidationError):
        data.preprocess_slide(CopyForbidden(), "slide_a", cfg=_valid_config())


def test_preprocessing_metadata_h5ad_round_trip(tmp_path, monkeypatch, synthetic_anndata_factory) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setitem(sys.modules, "scanpy", _fake_scanpy(calls, actual_hvgs=4))
    monkeypatch.setattr(data.st, "set_seeds", lambda _seed: None)
    monkeypatch.setattr(data.st, "run_leiden", lambda adata, **_kwargs: adata.obs.__setitem__("clusters", "0"))
    result = data.preprocess_slide(
        synthetic_anndata_factory(n_spots=8, n_genes=8), "slide_a", cfg=_valid_config()
    )
    expected = result.uns["spatial_pharma_preprocessing"]
    canonical = json.dumps(expected, sort_keys=True, separators=(",", ":"), allow_nan=False)
    result.obs_names = pd.Index(result.obs_names.to_numpy(dtype=object), dtype=object)
    result.var_names = pd.Index(result.var_names.to_numpy(dtype=object), dtype=object)
    result.obs["slide_id"] = result.obs["slide_id"].astype(object)
    result.obs["clusters"] = result.obs["clusters"].astype(object)
    raw = result.raw.to_adata()
    raw.obs_names = pd.Index(raw.obs_names.to_numpy(dtype=object), dtype=object)
    raw.var_names = pd.Index(raw.var_names.to_numpy(dtype=object), dtype=object)
    result.raw = raw
    path = tmp_path / "processed.h5ad"
    result.write_h5ad(path)
    restored = ad.read_h5ad(path).uns["spatial_pharma_preprocessing"]
    assert restored == expected
    assert json.dumps(expected, sort_keys=True, separators=(",", ":"), allow_nan=False) == canonical


def test_preprocessing_manifest_is_admitted_order_and_canonical() -> None:
    first = PreprocessingManifest(
        slide_ids=["slide_b", "slide_a"],
        records=[_record("slide_b"), _record("slide_a")],
    )
    second = PreprocessingManifest(
        slide_ids=["slide_b", "slide_a"],
        records=[_record("slide_b"), _record("slide_a")],
    )
    assert first.canonical_json == second.canonical_json
    assert first.to_dict()["slide_ids"] == ["slide_b", "slide_a"]
    assert [record["slide_id"] for record in first.to_dict()["records"]] == [
        "slide_b", "slide_a"
    ]
    assert first.to_dict()["schema_version"] == "spatial-pharma-preprocessing-manifest-v1"


def test_manifest_rejects_hostile_values_before_hooks_or_serialization() -> None:
    calls: list[str] = []

    class HostileInt(int):
        def bit_length(self):
            calls.append("bit_length")
            raise AssertionError("hostile hook")

        def __repr__(self):
            calls.append("repr")
            raise AssertionError("hostile hook")

    record = _record("slide_a")
    record["counts"]["input_spots"] = HostileInt(12)
    with pytest.raises(PreprocessingValidationError, match="malformed_preprocessing_manifest"):
        PreprocessingManifest(slide_ids=["slide_a"], records=[record])
    assert calls == []


def test_manifest_rejects_primitive_but_semantically_malformed_record() -> None:
    record = _record("slide_a")
    record["counts"]["actual_hvgs"] = True
    with pytest.raises(PreprocessingValidationError, match="malformed_preprocessing_manifest"):
        PreprocessingManifest(slide_ids=["slide_a"], records=[record])


def test_pipeline_manifest_failure_precedes_write_and_downstream(tmp_path, monkeypatch) -> None:
    runner = _load_runner()
    cfg = _valid_config()
    cfg["cohorts"] = {"oncology": ["slide_a", "slide_b"], "external": [], "benchmark": []}
    malformed = _record("slide_a")
    malformed["schema_version"] = "wrong"
    monkeypatch.setenv("PHARMA_TRAIN_ONLY", "1")
    monkeypatch.delenv("PHARMA_QUICK", raising=False)
    monkeypatch.delenv("PHARMA_FOUNDATION", raising=False)
    monkeypatch.setattr(runner, "load_config", lambda: cfg)
    monkeypatch.setattr(runner, "available_processed_slide_ids", lambda ids: set(ids))
    monkeypatch.setattr(runner, "pharma_outputs_dir", lambda: tmp_path)
    monkeypatch.setattr(
        runner,
        "load_slide",
        lambda sid: SimpleNamespace(uns={"spatial_pharma_preprocessing": malformed if sid == "slide_a" else _record(sid)}),
    )
    monkeypatch.setattr(
        runner,
        "_load_stages",
        lambda: SimpleNamespace(
            st=SimpleNamespace(set_seeds=lambda _seed: None),
            build_labels_cohort=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("labels reached")),
        ),
    )
    with pytest.raises(PreprocessingValidationError):
        runner.main()
    assert not (tmp_path / "preprocessing_manifest.json").exists()
    assert not (tmp_path / "cohort_manifest.json").exists()
