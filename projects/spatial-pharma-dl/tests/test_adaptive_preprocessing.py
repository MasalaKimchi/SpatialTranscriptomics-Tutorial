"""Offline evidence for adaptive preprocessing dimensions and provenance."""

from __future__ import annotations

import json
import importlib.util
import os
from pathlib import Path
import subprocess
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
    ConfigValidationError,
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


@pytest.mark.parametrize(
    "overrides",
    [
        {"after_filter_cells_genes": 9},
        {"after_filter_genes_spots": 9},
        {"post_qc_genes": 7},
    ],
)
def test_resolver_rejects_impossible_cross_axis_stage_histories(overrides) -> None:
    with pytest.raises(PreprocessingValidationError) as caught:
        _resolve(**overrides)
    assert caught.value.reason_code == "invalid_preprocessing_input"


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
    reached = {"scanpy_import": False, "copy": False, "seed": False}

    class CopyForbidden:
        obs_names = pd.Index(["spot_a", None], dtype=object)

        def copy(self):
            reached["copy"] = True
            raise AssertionError("copy reached")

    monkeypatch.delitem(sys.modules, "scanpy", raising=False)
    real_import = __import__("builtins").__import__

    def guarded_import(name, *args, **kwargs):
        if name == "scanpy":
            reached["scanpy_import"] = True
            raise AssertionError("scanpy import reached")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    def forbidden_seed(*_args):
        reached["seed"] = True
        raise AssertionError("seed reached")

    monkeypatch.setattr(data.st, "set_seeds", forbidden_seed)

    bad_cfg = _valid_config()
    bad_cfg["preprocessing"]["n_pcs"] = 0
    with pytest.raises(ConfigValidationError) as caught:
        data.preprocess_slide(synthetic_anndata_factory(), "slide_a", cfg=bad_cfg)
    assert any(issue.path == "preprocessing.n_pcs" for issue in caught.value.issues)
    assert reached == {"scanpy_import": False, "copy": False, "seed": False}
    with pytest.raises(IdentityValidationError):
        data.preprocess_slide(CopyForbidden(), "slide_a", cfg=_valid_config())
    assert reached == {"scanpy_import": False, "copy": False, "seed": False}


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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("after_filter_cells_genes", 9),
        ("after_filter_genes_spots", 9),
        ("post_qc_genes", 7),
    ],
)
def test_manifest_rejects_impossible_cross_axis_stage_histories(
    field, value
) -> None:
    record = _record("slide_a")
    record["counts"][field] = value

    with pytest.raises(
        PreprocessingValidationError, match="malformed_preprocessing_manifest"
    ):
        PreprocessingManifest(slide_ids=["slide_a"], records=[record])


def test_manifest_rejects_forged_stage_exclusion_history() -> None:
    record = _record("slide_a")
    record["exclusions"]["gene_filter"] += 1

    with pytest.raises(
        PreprocessingValidationError, match="malformed_preprocessing_manifest"
    ):
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
            build_labels_cohort=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("labels reached")
            ),
            build_patch_cohort=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("patches reached")
            ),
            run_and_save_benchmark=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("model reached")
            ),
            evaluate_fold=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("report reached")
            ),
        ),
    )
    with pytest.raises(PreprocessingValidationError):
        runner.main()
    assert not (tmp_path / "preprocessing_manifest.json").exists()
    assert not (tmp_path / "cohort_manifest.json").exists()


def _declared_scanpy_python(tmp_path: Path) -> Path:
    """Return a working declared-environment interpreter or fail actionably."""
    candidates = [Path(sys.executable)]
    conda_root = Path(sys.executable).resolve().parents[1]
    candidates.append(conda_root / "envs" / "spatial-tx" / "bin" / "python")
    failures: list[str] = []
    environment = os.environ.copy()
    environment.update(
        {
            "MPLCONFIGDIR": str(tmp_path / "matplotlib"),
            "NUMBA_DISABLE_JIT": "1",
            "XDG_CACHE_HOME": str(tmp_path / "cache"),
        }
    )
    for candidate in dict.fromkeys(candidates):
        if not candidate.is_file():
            failures.append(f"{candidate}: interpreter not found")
            continue
        probe = subprocess.run(
            [str(candidate), "-c", "import scanpy; print(scanpy.__version__)"],
            capture_output=True,
            check=False,
            env=environment,
            text=True,
            timeout=120,
        )
        if probe.returncode == 0:
            return candidate
        detail = (probe.stderr or probe.stdout).strip().splitlines()
        failures.append(
            f"{candidate}: {detail[-1] if detail else 'Scanpy import failed'}"
        )
    pytest.fail(
        "Real Scanpy is required for the fast scientific integration gate. "
        "Activate/install the declared spatial-tx environment from environment.yml. "
        + " | ".join(failures)
    )


def _run_real_scanpy_preprocessing(tmp_path: Path) -> Path:
    interpreter = _declared_scanpy_python(tmp_path)
    output_path = tmp_path / "slide_real_clustered.h5ad"
    pharma_root = CONFIG_PATH.parent.parent
    repository_root = pharma_root.parents[1]
    script = r'''
from pathlib import Path
import sys

import anndata as ad
import numpy as np
import pandas as pd
import yaml
import numba

# The managed sandbox makes installed package sources non-cacheable to Numba.
# Keep real vectorized execution while disabling only decorator disk caching.
_vectorize = numba.vectorize
def _sandbox_vectorize(*args, **kwargs):
    kwargs["cache"] = False
    return _vectorize(*args, **kwargs)
numba.vectorize = _sandbox_vectorize

pharma_root = Path(sys.argv[1])
repository_root = Path(sys.argv[2])
output_path = Path(sys.argv[3])
sys.path[:0] = [str(pharma_root), str(repository_root)]

from src.data import preprocess_slide

rng = np.random.default_rng(903)
genes = ["MT-CO1", "EPCAM", "COL1A1", "CD3D", "MS4A1", "VIM", "MKI67", "CXCL9"]
counts = rng.poisson(4.0, size=(10, len(genes))).astype(np.int32) + 1
obs_names = [f"slide_real_spot_{index:02d}" for index in range(len(counts))]
adata = ad.AnnData(
    X=counts,
    obs=pd.DataFrame({"slide_id": "slide_real"}, index=obs_names),
    var=pd.DataFrame(index=genes),
)
config_path = pharma_root / "configs" / "default.yaml"
cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
cfg["preprocessing"].update(
    min_counts=1,
    min_cells=1,
    max_pct_mito=100,
    n_top_genes_hvg=100,
    n_pcs=50,
    n_neighbors=50,
    n_pcs_neighbors=30,
)
result = preprocess_slide(adata, "slide_real", cfg=cfg, seed=903)
# Older Scanpy records an optional null log base using an H5AD encoding that
# the repository's newer reader cannot decode. Omit only that optional null.
if result.uns.get("log1p", {}).get("base") is None:
    result.uns["log1p"].pop("base", None)
result.obs_names = pd.Index(result.obs_names.to_numpy(dtype=object), dtype=object)
result.var_names = pd.Index(result.var_names.to_numpy(dtype=object), dtype=object)
for column in ("slide_id", "clusters"):
    result.obs[column] = result.obs[column].astype(object)
raw = result.raw.to_adata()
raw.obs_names = pd.Index(raw.obs_names.to_numpy(dtype=object), dtype=object)
raw.var_names = pd.Index(raw.var_names.to_numpy(dtype=object), dtype=object)
result.raw = raw
result.write_h5ad(output_path)
'''
    environment = os.environ.copy()
    environment.update(
        {
            "MPLCONFIGDIR": str(tmp_path / "matplotlib"),
            "NUMBA_DISABLE_JIT": "1",
            "XDG_CACHE_HOME": str(tmp_path / "cache"),
        }
    )
    completed = subprocess.run(
        [
            str(interpreter),
            "-c",
            script,
            str(pharma_root),
            str(repository_root),
            str(output_path),
        ],
        capture_output=True,
        check=False,
        cwd=repository_root,
        env=environment,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, (
        "Real Scanpy preprocessing failed in the declared spatial-tx environment. "
        "Repair environment.yml compatibility before accepting this phase.\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    assert output_path.is_file()
    return output_path


def test_real_scanpy_capped_preprocessing_round_trip_matches_run_provenance(
    tmp_path, monkeypatch
) -> None:
    """Execute the scientific stack and compare both persisted provenance surfaces."""
    path = _run_real_scanpy_preprocessing(tmp_path)
    monkeypatch.setattr(data, "pharma_processed_dir", lambda: tmp_path)
    restored = data.load_slide("slide_real")
    record = restored.uns["spatial_pharma_preprocessing"]
    resolved = record["resolved"]

    assert resolved["hvg_call"] < record["requested"]["hvg"]
    assert resolved["pca"] < record["requested"]["pca"]
    assert resolved["neighbors"] < record["requested"]["neighbors"]
    assert resolved["graph_pcs"] < record["requested"]["graph_pcs"]
    assert restored.obsm["X_pca"].shape == (restored.n_obs, resolved["pca"])
    assert restored.uns["neighbors"]["params"]["n_neighbors"] == resolved["neighbors"]
    assert restored.uns["neighbors"]["params"]["n_pcs"] == resolved["graph_pcs"]
    assert "counts" in restored.layers
    assert restored.raw is not None
    assert "pca" in restored.uns
    assert "clusters" in restored.obs

    runner = _load_runner()
    monkeypatch.setattr(runner, "load_slide", data.load_slide)
    admitted = SimpleNamespace(slide_ids=("slide_real",))
    first = runner._assemble_preprocessing_manifest(admitted)
    second = runner._assemble_preprocessing_manifest(admitted)
    manifest = first.to_dict()
    assert manifest["slide_ids"] == ["slide_real"]
    assert manifest["records"] == [record]
    assert first.canonical_json == second.canonical_json
    assert first.canonical_json.encode("utf-8") == second.canonical_json.encode("utf-8")
    assert json.loads(restored.uns["spatial_pharma_preprocessing_canonical_json"]) == record
    assert path == tmp_path / "slide_real_clustered.h5ad"


def test_nonviable_preprocessing_stops_before_graph_save_or_downstream(
    monkeypatch, synthetic_anndata_factory
) -> None:
    """Two retained spots cannot reach normalization, graph, or publication seams."""
    calls: list[str] = []

    def qc(adata, **_kwargs):
        calls.append("qc")
        adata.obs["pct_counts_mt"] = np.zeros(adata.n_obs)

    def pass_through(_adata, **_kwargs):
        calls.append("filter")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("nonviable preprocessing reached a downstream seam")

    fake_scanpy = SimpleNamespace(
        pp=SimpleNamespace(
            calculate_qc_metrics=qc,
            filter_cells=pass_through,
            filter_genes=pass_through,
            normalize_total=forbidden,
            log1p=forbidden,
            highly_variable_genes=forbidden,
            scale=forbidden,
            pca=forbidden,
            neighbors=forbidden,
        ),
        tl=SimpleNamespace(umap=forbidden),
    )
    monkeypatch.setitem(sys.modules, "scanpy", fake_scanpy)
    monkeypatch.setattr(data.st, "set_seeds", lambda _seed: None)
    monkeypatch.setattr(data.st, "run_leiden", forbidden)
    monkeypatch.setattr(data, "save_slide", forbidden)

    with pytest.raises(
        PreprocessingValidationError, match="insufficient_post_qc_spots"
    ):
        data.preprocess_slide(
            synthetic_anndata_factory(n_spots=2, n_genes=8),
            "slide_a",
            cfg=_valid_config(),
        )
    assert calls == ["qc", "filter", "filter"]
