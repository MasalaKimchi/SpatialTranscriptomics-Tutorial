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
    ArtifactValidationError,
    manifest_path,
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
        lambda _slide_id, _cfg, **_kwargs: {
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
    "utils/artifacts.py": {
        'reason = "legacy_artifact" if path.is_file() else "missing_payload"',
        "if snapshot_path is not None and not snapshot_path.exists():",
        'reason = "legacy_artifact" if path.exists() else "missing_payload"',
    },
    "utils/st_helpers.py": {
        "value = ad.read_h5ad(path)",
        "write_payload=lambda temporary: adata.write_h5ad(temporary),",
        "restored = pd.read_csv(candidate, index_col=0 if include_index else None)",
        "write_payload=lambda temporary: frame.to_csv(",
    },
    "projects/spatial-pharma-dl/src/data.py": {
        "value = ad.read_h5ad(path)",
        "write_payload=lambda temporary: adata.write_h5ad(temporary),",
    },
    "projects/spatial-pharma-dl/src/patches.py": {
        "with np.load(path, allow_pickle=True) as data:",
        'np.savez_compressed(handle, patches=patches, meta=meta.to_dict("list"))',
        "frame = pd.read_parquet(path)",
        "write_payload=lambda temporary: labels.to_parquet(temporary, index=False),",
    },
    "projects/spatial-pharma-dl/src/foundation.py": {
        "with np.load(path, allow_pickle=False) as cached:",
        "np.savez_compressed(",
    },
    "projects/spatial-pharma-dl/src/labels.py": {
        "pd.read_parquet(path)",
        "else pd.read_csv(",
        "write_payload=lambda temporary: frame.to_parquet(temporary, index=False),",
        "write_payload=lambda temporary: frame.to_csv(temporary, index=False),",
    },
    "projects/spatial-pharma-dl/src/models.py": {
        'payload = torch.load(path, map_location="cpu", weights_only=False)',
        "write_payload=lambda temporary: torch.save(",
    },
    "projects/spatial-pharma-dl/src/eval.py": {
        "write_payload=lambda temporary: df.to_csv(temporary, index=False),",
        "frame = pd.read_csv(",
        "frame = pd.read_csv(path)",
        "write_payload=lambda temporary: frame.to_csv(temporary, index=False),",
        "write_payload=lambda temporary: temporary.write_bytes(raw),",
    },
    "projects/spatial-pharma-dl/scripts/run_pipeline.py": {
        'report.to_csv(index=False).encode("utf-8")'
    },
    "projects/spatial-pharma-dl/scripts/build_notebooks.py": {"json.dump(nb(cells), f, indent=1)"},
    "projects/spatial-pharma-dl/scripts/build_foundation_notebook.py": {
        'if not (ROOT / "projects" / "spatial-pharma-dl").exists():'
    },
    "projects/spatial-pharma-dl/notebooks/07_foundation_model_comparison.ipynb": {
        'if not (ROOT / "projects" / "spatial-pharma-dl").exists():'
    },
    "00_overview_spatial_transcriptomics.ipynb": {"if not (ROOT / 'utils').exists():"},
    "01_environment_setup.ipynb": {"if not (ROOT / 'utils').exists():"},
    "02_fetch_public_visium_data.ipynb": {
        "if not (ROOT / 'utils').exists():",
        "entries = sorted(base.iterdir(), key=lambda p: (p.is_file(), p.name))",
        "size = f'  ({entry.stat().st_size/1e6:.1f} MB)' if entry.is_file() else ''",
        "assert (st.processed_dir() / 'adata_raw.h5ad').exists(), 'Raw cache missing!'",
    },
    "03_load_expression_and_spatial_metadata.ipynb": {"if not (ROOT / 'utils').exists():"},
    "04_qc_and_preprocessing.ipynb": {"if not (ROOT / 'utils').exists():"},
    "05_histology_image_loading_and_preprocessing.ipynb": {"if not (ROOT / 'utils').exists():"},
    "06_spatial_visualization.ipynb": {"if not (ROOT / 'utils').exists():"},
    "07_clustering_and_spatial_domains.ipynb": {"if not (ROOT / 'utils').exists():"},
    "08_spatially_variable_genes.ipynb": {"if not (ROOT / 'utils').exists():"},
    "09_image_feature_extraction_from_histology.ipynb": {"if not (ROOT / 'utils').exists():"},
    "10_integrating_histology_features_with_gene_expression.ipynb": {"if not (ROOT / 'utils').exists():"},
    "11_cell_type_annotation_and_deconvolution_optional.ipynb": {"if not (ROOT / 'utils').exists():"},
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
        for path in sorted((root / "projects/spatial-pharma-dl/notebooks").glob("*.ipynb")):
            notebook = json.loads(path.read_text(encoding="utf-8"))
            relative = str(path.relative_to(root))
            sources[relative] = "\n".join(
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
            if line.strip() not in allowed:
                violations.append((relative, line_number, line.strip()))
    return violations


def test_static_raw_io_inventory_is_narrow_and_detects_new_bypass():
    root = Path(__file__).resolve().parents[3]
    assert _raw_io_inventory(root) == []
    synthetic = {
        "projects/spatial-pharma-dl/src/new_bypass.py": "pd.read_csv(path)\n",
        "projects/spatial-pharma-dl/src/models.py": "if path.is_file():\n    reuse(path)\n",
    }
    assert _raw_io_inventory(root, synthetic) == [
        ("projects/spatial-pharma-dl/src/new_bypass.py", 1, "pd.read_csv(path)"),
        ("projects/spatial-pharma-dl/src/models.py", 1, "if path.is_file():"),
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
