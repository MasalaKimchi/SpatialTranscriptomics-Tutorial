"""Label engineering: harmonized TME classes, module scores, and task columns."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import bootstrap  # noqa: F401
from utils import st_helpers as st

from .data import (
    _processed_fingerprint,
    load_config,
    load_slide,
    tumor_type_for_slide,
)
from .identity import align_labels_with_metadata, validate_anndata_spot_identity
from .validation import require_non_empty, resolve_config
from utils.artifacts import (
    ARTIFACT_CONTRACT_VERSIONS,
    ArtifactValidationError,
    admit_artifact,
    build_fingerprint,
    publish_artifact,
)


DOMAIN_KEYWORDS = {
    "immune_enriched": ["CD3", "CD8", "CD4", "LCK", "PTPRC", "CCL", "CXCL"],
    "stromal": ["COL1", "COL3", "FN1", "ACTA2", "DCN", "LUM", "VIM"],
    "tumor_epithelial": ["KRT", "EPCAM", "MUC1", "PAX8", "CDX2", "KRT20"],
    "proliferative": ["MKI67", "TOP2A", "PCNA", "CCNB"],
    "hypoxic": ["HIF1A", "VEGFA", "LDHA", "CA9"],
    "neuronal": ["SNAP", "MBP", "PLP", "SYT", "RBFOX"],
    "glial": ["GFAP", "AQP4", "OLIG", "MOG"],
}

_LABEL_COLUMNS = (
    "slide_id",
    "spot_id",
    "cluster",
    "cluster_id",
    "domain_name",
    "tme_class",
    "tme_class_id",
)
_DOMAIN_COLUMNS = ("slide_id", "cluster", "domain_name", "tme_class")


def _output_path(filename: str) -> Path:
    return st.project_root() / "outputs" / "pharma" / filename


def _label_path(sample_id: str) -> Path:
    return _output_path(f"labels_{sample_id.replace(' ', '_')}.parquet")


def _table_fingerprint(
    kind: str,
    sample_ids: list[str],
    cfg: dict[str, Any],
):
    resolved = resolve_config(cfg).to_dict()
    upstream = {
        sid: _processed_fingerprint(sid, resolved).digest for sid in sample_ids
    }
    return build_fingerprint(
        kind,
        {
            "configuration": resolved,
            "source": {
                "marker_genes": resolved["marker_genes"],
                "gene_modules": resolved["gene_modules"],
                "domain_keywords": DOMAIN_KEYWORDS,
            },
            "upstream": {"processed_slides": upstream},
            "identity": {"slide_ids": sample_ids},
        },
    )


def _table_schema(frame: pd.DataFrame, *, kind: str, sample_ids: list[str]):
    expected = _LABEL_COLUMNS if kind == "label_table" else _DOMAIN_COLUMNS
    columns = frame.columns.tolist()
    if any(type(column) is not str for column in columns) or len(columns) != len(
        set(columns)
    ):
        raise ArtifactValidationError(
            "reader_validation_failed", artifact_kind=kind
        )
    if kind == "label_table":
        if tuple(columns[: len(expected)]) != expected or any(
            not (column.startswith("gene_") or column.startswith("module_"))
            for column in columns[len(expected) :]
        ):
            raise ArtifactValidationError(
                "reader_validation_failed", artifact_kind=kind
            )
    elif tuple(columns) != expected:
        raise ArtifactValidationError("reader_validation_failed", artifact_kind=kind)
    if frame.empty or frame[list(expected)].isnull().any().any():
        raise ArtifactValidationError("reader_validation_failed", artifact_kind=kind)
    for column in ("slide_id", "cluster", "domain_name", "tme_class"):
        if any(type(value) is not str or not value.strip() for value in frame[column].tolist()):
            raise ArtifactValidationError(
                "reader_validation_failed", artifact_kind=kind
            )
    if kind == "label_table":
        if any(
            type(value) is not str or not value.strip()
            for value in frame["spot_id"].tolist()
        ):
            raise ArtifactValidationError(
                "reader_validation_failed", artifact_kind=kind
            )
        if frame.duplicated(["slide_id", "spot_id"]).any():
            raise ArtifactValidationError(
                "reader_validation_failed", artifact_kind=kind
            )
        numeric = ["cluster_id", "tme_class_id", *columns[len(expected) :]]
        if any(not pd.api.types.is_numeric_dtype(frame[column]) for column in numeric):
            raise ArtifactValidationError(
                "reader_validation_failed", artifact_kind=kind
            )
        values = frame[numeric].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ArtifactValidationError(
                "reader_validation_failed", artifact_kind=kind
            )
        key_columns = ["slide_id", "spot_id"]
    else:
        if frame.duplicated(["slide_id", "cluster"]).any():
            raise ArtifactValidationError(
                "reader_validation_failed", artifact_kind=kind
            )
        key_columns = ["slide_id", "cluster"]
    if set(frame["slide_id"].tolist()) != set(sample_ids):
        raise ArtifactValidationError("reader_validation_failed", artifact_kind=kind)
    keys = frame[key_columns].to_dict("split")["data"]
    return {
        "columns": columns,
        "rows": int(len(frame)),
        "identity_sha256": hashlib.sha256(
            json.dumps(keys, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def _read_table(path: Path, *, kind: str, sample_ids: list[str]):
    frame = (
        pd.read_parquet(path)
        if kind == "label_table"
        else pd.read_csv(
            path,
            dtype={
                "slide_id": str,
                "cluster": str,
                "domain_name": str,
                "tme_class": str,
            },
        )
    )
    schema = _table_schema(frame, kind=kind, sample_ids=sample_ids)
    return frame, schema


def save_label_table(
    frame: pd.DataFrame,
    sample_id: str,
    *,
    cfg: dict[str, Any] | None = None,
) -> Path:
    resolved = load_config() if cfg is None else resolve_config(cfg).to_dict()
    load_slide(sample_id, cfg=resolved)
    path = _label_path(sample_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = _table_schema(frame, kind="label_table", sample_ids=[sample_id])
    publish_artifact(
        path,
        artifact_kind="label_table",
        contract_version=ARTIFACT_CONTRACT_VERSIONS["label_table"],
        fingerprint=_table_fingerprint("label_table", [sample_id], resolved),
        payload_format="parquet",
        payload_schema=schema,
        write_payload=lambda temporary: frame.to_parquet(temporary, index=False),
        reader=lambda temporary: _read_table(
            temporary, kind="label_table", sample_ids=[sample_id]
        ),
    )
    return path


def load_label_table(
    sample_id: str,
    *,
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    resolved = load_config() if cfg is None else resolve_config(cfg).to_dict()
    load_slide(sample_id, cfg=resolved)
    path = _label_path(sample_id)
    admission = admit_artifact(
        path,
        expected_kind="label_table",
        expected_contract_version=ARTIFACT_CONTRACT_VERSIONS["label_table"],
        expected_fingerprint=_table_fingerprint("label_table", [sample_id], resolved),
        reader=lambda candidate: _read_table(
            candidate, kind="label_table", sample_ids=[sample_id]
        ),
    )
    frame, schema = admission.value
    if json.loads(admission.manifest.payload_schema_json) != schema:
        raise ArtifactValidationError(
            "payload_schema_mismatch", artifact_kind="label_table", basename=path.name
        )
    return frame


def save_domain_table(
    frame: pd.DataFrame,
    sample_ids: list[str],
    *,
    cfg: dict[str, Any] | None = None,
) -> Path:
    resolved = load_config() if cfg is None else resolve_config(cfg).to_dict()
    for sample_id in sample_ids:
        load_slide(sample_id, cfg=resolved)
    path = _output_path("domain_annotations.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = _table_schema(frame, kind="domain_table", sample_ids=sample_ids)
    publish_artifact(
        path,
        artifact_kind="domain_table",
        contract_version=ARTIFACT_CONTRACT_VERSIONS["domain_table"],
        fingerprint=_table_fingerprint("domain_table", sample_ids, resolved),
        payload_format="csv",
        payload_schema=schema,
        write_payload=lambda temporary: frame.to_csv(temporary, index=False),
        reader=lambda temporary: _read_table(
            temporary, kind="domain_table", sample_ids=sample_ids
        ),
    )
    return path


def load_domain_table(
    sample_ids: list[str], *, cfg: dict[str, Any] | None = None
) -> pd.DataFrame:
    resolved = load_config() if cfg is None else resolve_config(cfg).to_dict()
    for sample_id in sample_ids:
        load_slide(sample_id, cfg=resolved)
    path = _output_path("domain_annotations.csv")
    admission = admit_artifact(
        path,
        expected_kind="domain_table",
        expected_contract_version=ARTIFACT_CONTRACT_VERSIONS["domain_table"],
        expected_fingerprint=_table_fingerprint("domain_table", sample_ids, resolved),
        reader=lambda candidate: _read_table(
            candidate, kind="domain_table", sample_ids=sample_ids
        ),
    )
    frame, schema = admission.value
    if json.loads(admission.manifest.payload_schema_json) != schema:
        raise ArtifactValidationError(
            "payload_schema_mismatch", artifact_kind="domain_table", basename=path.name
        )
    return frame


def tme_class_names(cfg: dict[str, Any] | None = None) -> list[str]:
    if cfg is None:
        cfg = load_config()
    return list(cfg["labels"]["tme_classes"])


def harmonize_tme_class(domain_name: str, cfg: dict[str, Any] | None = None) -> str:
    """Map slide-local domain_name to a global cross-slide TME class."""
    if cfg is None:
        cfg = load_config()
    allowed = set(tme_class_names(cfg))
    if domain_name in allowed:
        return domain_name
    return "other"


def tme_class_to_id(cfg: dict[str, Any] | None = None) -> dict[str, int]:
    return {name: i for i, name in enumerate(tme_class_names(cfg))}


def marker_genes_for_slide(
    sample_id: str, cfg: dict[str, Any] | None = None
) -> list[str]:
    if cfg is None:
        cfg = load_config()
    ttype = tumor_type_for_slide(sample_id)
    return cfg["marker_genes"].get(ttype, cfg["marker_genes"]["breast"])


def compute_module_scores(adata, cfg: dict[str, Any] | None = None) -> list[str]:
    import scanpy as sc

    if cfg is None:
        cfg = load_config()
    created = []
    for name, genes in cfg["gene_modules"].items():
        present = st.genes_present(adata, genes, verbose=False)
        if len(present) < 2:
            continue
        col = f"module_{name}"
        sc.tl.score_genes(adata, present, score_name=col)
        created.append(col)
    return created


def annotate_domain(cluster_markers: pd.DataFrame) -> dict[str, str]:
    annotations = {}
    for cluster, grp in cluster_markers.groupby("group", observed=True):
        top_genes = grp.nlargest(5, "scores")["names"].tolist()
        scores = {label: 0 for label in DOMAIN_KEYWORDS}
        for gene in top_genes:
            for label, keywords in DOMAIN_KEYWORDS.items():
                if any(kw in gene.upper() for kw in keywords):
                    scores[label] += 1
        best = max(scores, key=scores.get)
        annotations[str(cluster)] = best if scores[best] > 0 else f"domain_{cluster}"
    return annotations


def gene_columns(labels: pd.DataFrame) -> list[str]:
    return [c for c in labels.columns if c.startswith("gene_")]


def module_columns(labels: pd.DataFrame) -> list[str]:
    return [c for c in labels.columns if c.startswith("module_")]


def regression_columns(
    labels: pd.DataFrame, cfg: dict[str, Any] | None = None
) -> list[str]:
    """Return regression target columns per config (modules, genes, or both)."""
    if cfg is None:
        cfg = load_config()
    mode = cfg["labels"].get("regression_targets", "modules")
    mods = module_columns(labels)
    genes = gene_columns(labels)
    if mode == "modules":
        selected = mods
    elif mode == "genes":
        selected = genes
    elif mode == "both":
        selected = mods + genes
    else:
        raise ValueError(f"Unknown regression_targets: {mode!r}")
    require_non_empty(
        selected,
        stage="regression_target_selection",
        subject=f"{mode} regression target columns",
        guidance=(
            "Generate the configured module or gene target columns before "
            "selecting regression targets."
        ),
    )
    return selected


def classification_column(cfg: dict[str, Any] | None = None) -> str:
    if cfg is None:
        cfg = load_config()
    return cfg["labels"]["classification_col"]


def build_labels_for_slide(
    sample_id: str,
    cfg: dict[str, Any] | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    import scanpy as sc

    if cfg is None:
        cfg = load_config()
    seed = seed if seed is not None else cfg.get("seed", st.SEED)
    adata = load_slide(sample_id)
    validate_anndata_spot_identity(
        adata,
        sample_id,
        stage="slide_label_generation",
        require_slide_id=True,
    )
    class_map = tme_class_to_id(cfg)

    markers = st.genes_present(
        adata, marker_genes_for_slide(sample_id, cfg), verbose=False
    )
    module_cols = compute_module_scores(adata, cfg)

    sc.tl.rank_genes_groups(adata, "clusters", method="wilcoxon", random_state=seed)
    cluster_markers = sc.get.rank_genes_groups_df(adata, group=None)
    domain_map = annotate_domain(cluster_markers)

    gene_expr = (
        adata[:, markers].to_df() if markers else pd.DataFrame(index=adata.obs_names)
    )

    rows = []
    cluster_to_id = {
        c: i for i, c in enumerate(sorted(adata.obs["clusters"].unique(), key=str))
    }

    for spot_id in adata.obs_names:
        cluster = str(adata.obs.loc[spot_id, "clusters"])
        domain_name = domain_map.get(cluster, f"domain_{cluster}")
        tme_class = harmonize_tme_class(domain_name, cfg)
        row = {
            "slide_id": sample_id,
            "spot_id": spot_id,
            "cluster": cluster,
            "cluster_id": cluster_to_id[cluster],
            "domain_name": domain_name,
            "tme_class": tme_class,
            "tme_class_id": class_map[tme_class],
        }
        for gene in markers:
            row[f"gene_{gene}"] = float(gene_expr.loc[spot_id, gene])
        for col in module_cols:
            row[col] = float(adata.obs.loc[spot_id, col])
        rows.append(row)

    labels = pd.DataFrame(rows)
    require_non_empty(
        labels,
        stage="slide_label_generation",
        subject=f"labels for slide {sample_id}",
        guidance="Retain at least one usable spot before generating slide labels.",
    )
    return labels


def build_labels_cohort(
    sample_ids: list[str], cfg: dict[str, Any] | None = None
) -> pd.DataFrame:
    if cfg is None:
        cfg = load_config()
    require_non_empty(
        sample_ids,
        stage="cohort_label_generation",
        subject="admitted slide sequence",
        guidance="Admit at least one slide before generating cohort labels.",
    )
    frames = []
    domain_rows = []

    for sid in sample_ids:
        labels = build_labels_for_slide(sid, cfg)
        require_non_empty(
            labels,
            stage="cohort_label_generation",
            subject=f"label rows for slide {sid}",
            guidance="Retain at least one usable labeled spot for every admitted slide.",
        )
        frames.append(labels)
        for cluster in labels["cluster"].unique():
            domain_rows.append(
                {
                    "slide_id": sid,
                    "cluster": cluster,
                    "domain_name": labels.loc[
                        labels["cluster"] == cluster, "domain_name"
                    ].iloc[0],
                    "tme_class": labels.loc[
                        labels["cluster"] == cluster, "tme_class"
                    ].iloc[0],
                }
            )
    require_non_empty(
        frames,
        stage="cohort_label_generation",
        subject="per-slide label frames",
        guidance="Generate at least one non-empty per-slide label frame.",
    )
    total_rows = sum(len(frame) for frame in frames)
    require_non_empty(
        range(total_rows),
        stage="cohort_label_generation",
        subject="combined cohort label rows",
        guidance="Retain at least one usable labeled spot in the admitted cohort.",
    )
    for sid, frame in zip(sample_ids, frames, strict=True):
        path = save_label_table(frame, sid, cfg=cfg)
        print(f"Wrote {path} ({len(frame)} spots)")
    if domain_rows:
        save_domain_table(pd.DataFrame(domain_rows), sample_ids, cfg=cfg)
    return pd.concat(frames, ignore_index=True)


def align_labels_with_patches(labels: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Align complete label and patch metadata tables in patch order."""
    return align_labels_with_metadata(
        labels,
        meta,
        stage="patch_label_alignment",
    )
