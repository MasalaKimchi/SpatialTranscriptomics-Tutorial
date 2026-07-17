"""Offline orchestration evidence for typed artifact reuse decisions."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import numpy as np
import pandas as pd

from src import data, eval as evaluation, patches
from utils import st_helpers as st
from utils.artifacts import (
    ARTIFACT_CONTRACT_VERSIONS,
    ArtifactValidationError,
    admit_artifact,
    build_fingerprint,
    manifest_path,
    publish_artifact,
)

pytestmark = pytest.mark.offline


def test_available_processed_status_is_pure_for_missing_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "project_root", lambda: tmp_path)
    assert data.available_processed_slide_ids(["slide_a"], cfg=data.load_config()) == set()
    assert not (tmp_path / "data").exists()


def test_legacy_patch_stops_before_local_object_decode(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "project_root", lambda: tmp_path)
    path = patches.patch_cache_path("slide_a", data.load_config())
    path.parent.mkdir(parents=True)
    path.write_bytes(b"legacy-without-sidecar")
    calls: list[str] = []

    def forbidden(*_args, **_kwargs):
        calls.append("np.load")
        raise AssertionError("unsafe decoder reached")

    monkeypatch.setattr(patches.np, "load", forbidden)
    with pytest.raises(ArtifactValidationError, match="legacy_artifact"):
        patches.load_patch_arrays("slide_a", cfg=data.load_config())
    assert calls == []


def test_patch_lineage_ignores_training_but_rejects_patch_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "project_root", lambda: tmp_path)
    cfg = data.load_config()
    monkeypatch.setattr(
        patches,
        "_patch_artifact_context",
        lambda _slide_id, _cfg: {
            "processed_slide": {"fingerprint": "a" * 64},
            "stain_reference": None,
        },
    )
    values = np.zeros((1, 3, 4, 4), dtype=np.float32)
    meta = pd.DataFrame(
        {
            "slide_id": ["slide_a"],
            "spot_id": ["spot_0"],
            "x": [1.0],
            "y": [2.0],
            "native_patch_px": [8],
        }
    )
    patches.save_patch_arrays("slide_a", values, meta, cfg=cfg)
    training_only = {**cfg, "training": {**cfg["training"], "epochs": 99}}
    assert patches.load_patch_arrays("slide_a", cfg=training_only)[0].shape == values.shape
    changed = {**cfg, "patches": {**cfg["patches"], "output_size": 112}}
    with pytest.raises(ArtifactValidationError, match="stale_fingerprint"):
        patches.load_patch_arrays("slide_a", cfg=changed)


PIPELINE_ARTIFACT_KINDS = {
    "root_raw_h5ad": "root_h5ad",
    "root_qc_h5ad": "root_h5ad",
    "root_clustered_h5ad": "root_h5ad",
    "root_features_h5ad": "root_h5ad",
    "processed_h5ad": "processed_slide",
    "label_parquet": "label_table",
    "domain_csv": "domain_table",
    "patch_npz": "patch",
    "patch_index_parquet": "patch_index",
    "foundation_embedding_npz": "embedding",
    "local_checkpoint": "checkpoint",
    "benchmark_report": "report",
    "cohort_manifest": "cohort_manifest",
    "preprocessing_manifest": "preprocessing_manifest",
    "cohort_summary": "summary",
    "experiment_summary": "summary",
    "training_history": "summary",
    "nested_loso_results": "summary",
    "model_task_summary": "summary",
}


def test_real_nineteen_kind_pipeline_fixture_roundtrips_and_invalidates(tmp_path):
    assert len(PIPELINE_ARTIFACT_KINDS) == 19
    parent = "root-source"
    for logical_name, kind in PIPELINE_ARTIFACT_KINDS.items():
        path = tmp_path / f"{logical_name}.bin"
        inputs = {
            "configuration": {},
            "source": {"logical_name": logical_name},
            "upstream": {"parent": parent},
            "identity": {"logical_name": logical_name},
        }
        fingerprint = build_fingerprint(kind, inputs)
        payload = f"payload:{logical_name}".encode()
        schema = {"logical_name": logical_name}
        publish_artifact(
            path,
            artifact_kind=kind,
            contract_version=ARTIFACT_CONTRACT_VERSIONS[kind],
            fingerprint=fingerprint,
            payload_format="test-bytes",
            payload_schema=schema,
            write_payload=lambda temporary, value=payload: temporary.write_bytes(value),
            reader=lambda candidate: candidate.read_bytes(),
            observed_schema=lambda _decoded, value=schema: value,
        )
        admission = admit_artifact(
            path,
            expected_kind=kind,
            expected_contract_version=ARTIFACT_CONTRACT_VERSIONS[kind],
            expected_fingerprint=fingerprint,
            reader=lambda candidate: candidate.read_bytes(),
        )
        assert admission.value == payload
        stale = json.loads(json.dumps(inputs))
        stale["upstream"]["parent"] = f"stale:{parent}"
        with pytest.raises(ArtifactValidationError, match="stale_fingerprint"):
            admit_artifact(
                path,
                expected_kind=kind,
                expected_contract_version=ARTIFACT_CONTRACT_VERSIONS[kind],
                expected_fingerprint=build_fingerprint(kind, stale),
                reader=lambda _candidate: pytest.fail("stale payload decoded"),
            )
        parent = admission.manifest.fingerprint.digest


def test_summary_reader_rejects_mixed_and_truncated_generations(tmp_path):
    cfg = data.load_config()
    path = tmp_path / "training_history.csv"
    first = pd.DataFrame(
        {"fold": [0], "val_slide": ["slide_a"], "epoch": [0], "train_loss": [1.0], "val_loss": [2.0]}
    )
    second = pd.DataFrame(
        {"fold": [1], "val_slide": ["slide_b"], "epoch": [0], "train_loss": [0.5], "val_loss": [1.0]}
    )
    evaluation.save_result_table(
        first, path, table_name="training_history", cfg=cfg,
        upstream_lineage={"checkpoint": "old"},
    )
    old_manifest = manifest_path(path).read_bytes()
    evaluation.save_result_table(
        second, path, table_name="training_history", cfg=cfg,
        upstream_lineage={"checkpoint": "new"},
    )
    manifest_path(path).write_bytes(old_manifest)
    with pytest.raises(ArtifactValidationError, match="stale_fingerprint"):
        evaluation.load_result_table(
            path, table_name="training_history", cfg=cfg,
            upstream_lineage={"checkpoint": "new"}, expected_rows=1,
        )

    evaluation.save_result_table(
        second, path, table_name="training_history", cfg=cfg,
        upstream_lineage={"checkpoint": "new"},
    )
    path.write_bytes(path.read_bytes()[:8])
    with pytest.raises(ArtifactValidationError):
        evaluation.load_result_table(
            path, table_name="training_history", cfg=cfg,
            upstream_lineage={"checkpoint": "new"}, expected_rows=1,
        )


_RAW_IO_PATTERN = re.compile(
    r"ad\.read_h5ad|\.write_h5ad\(|np\.load\(|np\.savez(?:_compressed)?\(|"
    r"pd\.read_(?:csv|parquet)\(|\.to_(?:csv|parquet)\(|torch\.(?:load|save)\(|"
    r"\.write_(?:text|bytes)\(|json\.dump\(|\.(?:exists|is_file)\(\)"
)

_RAW_IO_ALLOWLIST = {
    "utils/artifacts.py": {"path.exists()", "path.is_file()"},
    "utils/st_helpers.py": {
        "ad.read_h5ad(path)",
        "adata.write_h5ad(temporary)",
        "pd.read_csv(candidate",
        "frame.to_csv(",
    },
    "projects/spatial-pharma-dl/src/data.py": {
        "ad.read_h5ad(path)", "adata.write_h5ad(temporary)",
    },
    "projects/spatial-pharma-dl/src/patches.py": {
        "np.load(path, allow_pickle=True)", "np.savez_compressed(handle",
        "pd.read_parquet(path)", "labels.to_parquet(temporary, index=False)",
    },
    "projects/spatial-pharma-dl/src/foundation.py": {
        "np.load(path, allow_pickle=False)", "np.savez_compressed(",
    },
    "projects/spatial-pharma-dl/src/labels.py": {
        "pd.read_parquet(path)", "pd.read_csv(",
        "frame.to_parquet(temporary, index=False)", "frame.to_csv(temporary, index=False)",
    },
    "projects/spatial-pharma-dl/src/models.py": {
        "torch.load(path, map_location=\"cpu\", weights_only=False)",
        "torch.save(", "path.is_file()",
    },
    "projects/spatial-pharma-dl/src/eval.py": {
        "df.to_csv(temporary, index=False)", "pd.read_csv(", "path.is_file()",
        "frame.to_csv(temporary, index=False)", "temporary.write_bytes(raw)",
    },
    "projects/spatial-pharma-dl/scripts/run_pipeline.py": {"report.to_csv(index=False)"},
    "projects/spatial-pharma-dl/scripts/build_notebooks.py": {"json.dump(nb(cells), f, indent=1)"},
    "projects/spatial-pharma-dl/scripts/build_foundation_notebook.py": {
        '(ROOT / "projects" / "spatial-pharma-dl").exists()'
    },
    "00_overview_spatial_transcriptomics.ipynb": {"(ROOT / 'utils').exists()"},
    "01_environment_setup.ipynb": {"(ROOT / 'utils').exists()"},
    "02_fetch_public_visium_data.ipynb": {
        "(ROOT / 'utils').exists()",
        "p.is_file()",
        "entry.is_file()",
        "'adata_raw.h5ad').exists()",
    },
    "03_load_expression_and_spatial_metadata.ipynb": {"(ROOT / 'utils').exists()"},
    "04_qc_and_preprocessing.ipynb": {"(ROOT / 'utils').exists()"},
    "05_histology_image_loading_and_preprocessing.ipynb": {"(ROOT / 'utils').exists()"},
    "06_spatial_visualization.ipynb": {"(ROOT / 'utils').exists()"},
    "07_clustering_and_spatial_domains.ipynb": {"(ROOT / 'utils').exists()"},
    "08_spatially_variable_genes.ipynb": {"(ROOT / 'utils').exists()"},
    "09_image_feature_extraction_from_histology.ipynb": {"(ROOT / 'utils').exists()"},
    "10_integrating_histology_features_with_gene_expression.ipynb": {"(ROOT / 'utils').exists()"},
    "11_cell_type_annotation_and_deconvolution_optional.ipynb": {"(ROOT / 'utils').exists()"},
}


def _raw_io_inventory(root: Path, sources: dict[str, str] | None = None):
    if sources is None:
        candidates = [
            *sorted((root / "utils").glob("*.py")),
            *sorted((root / "projects/spatial-pharma-dl/src").glob("*.py")),
            *sorted((root / "projects/spatial-pharma-dl/scripts").glob("*.py")),
            root / "scripts/generate_gallery_figures.py",
        ]
        sources = {
            str(path.relative_to(root)): path.read_text(encoding="utf-8")
            for path in candidates
        }
        for path in sorted(root.glob("*.ipynb")):
            notebook = json.loads(path.read_text(encoding="utf-8"))
            sources[path.name] = "\n".join(
                "".join(cell.get("source", []))
                for cell in notebook.get("cells", [])
                if cell.get("cell_type") == "code"
            )
    violations = []
    for relative, source in sources.items():
        allowed = _RAW_IO_ALLOWLIST.get(relative, set())
        for line_number, line in enumerate(source.splitlines(), 1):
            if not _RAW_IO_PATTERN.search(line):
                continue
            if not any(marker in line for marker in allowed):
                violations.append((relative, line_number, line.strip()))
    return violations


def test_static_raw_io_inventory_is_narrow_and_detects_new_bypass():
    root = Path(__file__).resolve().parents[3]
    assert _raw_io_inventory(root) == []
    synthetic = {
        "projects/spatial-pharma-dl/src/new_bypass.py": "pd.read_csv(path)\n"
    }
    assert _raw_io_inventory(root, synthetic) == [
        ("projects/spatial-pharma-dl/src/new_bypass.py", 1, "pd.read_csv(path)")
    ]


@pytest.mark.parametrize(
    ("result_name", "frame", "include_index"),
    [
        (
            "qc_summary",
            pd.DataFrame(
                {"total_counts": [10], "n_genes_by_counts": [5], "pct_counts_mt": [1.0]},
                index=["spot_0"],
            ),
            True,
        ),
        (
            "cluster_markers",
            pd.DataFrame(
                {"group": ["0"], "names": ["GENE"], "scores": [1.0],
                 "logfoldchanges": [1.0], "pvals": [0.01], "pvals_adj": [0.02]}
            ),
            False,
        ),
        (
            "image_features",
            pd.DataFrame(
                [[0.1] * 15], columns=list(st._ROOT_RESULT_COLUMNS["image_features"]),
                index=["spot_0"],
            ),
            True,
        ),
        (
            "integration_metrics",
            pd.DataFrame(
                {"task": ["regression"], "target": ["GENE"], "metric": ["R2"], "value": [0.5]}
            ),
            False,
        ),
    ],
)
def test_root_notebook_tables_publish_named_atomic_sidecars(
    tmp_path, result_name, frame, include_index
):
    path = tmp_path / f"{result_name}.csv"
    st.save_root_result_table(
        frame,
        path,
        result_name=result_name,
        include_index=include_index,
        upstream_lineage={"source": "current"},
    )
    assert path.is_file()
    assert manifest_path(path).is_file()
