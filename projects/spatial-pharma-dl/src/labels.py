"""Label engineering: harmonized TME classes, module scores, and task columns."""

from __future__ import annotations

from typing import Any

import pandas as pd
import scanpy as sc

from . import bootstrap  # noqa: F401
from utils import st_helpers as st

from .data import load_config, load_slide, pharma_outputs_dir, tumor_type_for_slide


DOMAIN_KEYWORDS = {
    "immune_enriched": ["CD3", "CD8", "CD4", "LCK", "PTPRC", "CCL", "CXCL"],
    "stromal": ["COL1", "COL3", "FN1", "ACTA2", "DCN", "LUM", "VIM"],
    "tumor_epithelial": ["KRT", "EPCAM", "MUC1", "PAX8", "CDX2", "KRT20"],
    "proliferative": ["MKI67", "TOP2A", "PCNA", "CCNB"],
    "hypoxic": ["HIF1A", "VEGFA", "LDHA", "CA9"],
    "neuronal": ["SNAP", "MBP", "PLP", "SYT", "RBFOX"],
    "glial": ["GFAP", "AQP4", "OLIG", "MOG"],
}


def tme_class_names(cfg: dict[str, Any] | None = None) -> list[str]:
    cfg = cfg or load_config()
    return list(cfg["labels"]["tme_classes"])


def harmonize_tme_class(domain_name: str, cfg: dict[str, Any] | None = None) -> str:
    """Map slide-local domain_name to a global cross-slide TME class."""
    cfg = cfg or load_config()
    allowed = set(tme_class_names(cfg))
    if domain_name in allowed:
        return domain_name
    return "other"


def tme_class_to_id(cfg: dict[str, Any] | None = None) -> dict[str, int]:
    return {name: i for i, name in enumerate(tme_class_names(cfg))}


def marker_genes_for_slide(sample_id: str, cfg: dict[str, Any] | None = None) -> list[str]:
    cfg = cfg or load_config()
    ttype = tumor_type_for_slide(sample_id)
    return cfg["marker_genes"].get(ttype, cfg["marker_genes"]["breast"])


def compute_module_scores(adata, cfg: dict[str, Any] | None = None) -> list[str]:
    cfg = cfg or load_config()
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


def regression_columns(labels: pd.DataFrame, cfg: dict[str, Any] | None = None) -> list[str]:
    """Return regression target columns per config (modules, genes, or both)."""
    cfg = cfg or load_config()
    mode = cfg["labels"].get("regression_targets", "modules")
    mods = module_columns(labels)
    genes = gene_columns(labels)
    if mode == "modules":
        return mods
    if mode == "genes":
        return genes
    if mode == "both":
        return mods + genes
    raise ValueError(f"Unknown regression_targets: {mode!r}")


def classification_column(cfg: dict[str, Any] | None = None) -> str:
    cfg = cfg or load_config()
    return cfg["labels"]["classification_col"]


def build_labels_for_slide(
    sample_id: str,
    cfg: dict[str, Any] | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    cfg = cfg or load_config()
    seed = seed if seed is not None else cfg.get("seed", st.SEED)
    adata = load_slide(sample_id)
    class_map = tme_class_to_id(cfg)

    markers = st.genes_present(
        adata, marker_genes_for_slide(sample_id, cfg), verbose=False
    )
    module_cols = compute_module_scores(adata, cfg)

    sc.tl.rank_genes_groups(adata, "clusters", method="wilcoxon", random_state=seed)
    cluster_markers = sc.get.rank_genes_groups_df(adata, group=None)
    domain_map = annotate_domain(cluster_markers)

    gene_expr = adata[:, markers].to_df() if markers else pd.DataFrame(index=adata.obs_names)

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

    return pd.DataFrame(rows)


def build_labels_cohort(
    sample_ids: list[str], cfg: dict[str, Any] | None = None
) -> pd.DataFrame:
    cfg = cfg or load_config()
    out_dir = pharma_outputs_dir()
    frames = []
    domain_rows = []

    for sid in sample_ids:
        try:
            labels = build_labels_for_slide(sid, cfg)
        except FileNotFoundError:
            print(f"Skipping labels for {sid} (slide not preprocessed)")
            continue
        path = out_dir / f"labels_{sid.replace(' ', '_')}.parquet"
        labels.to_parquet(path, index=False)
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
        print(f"Wrote {path} ({len(labels)} spots)")

    if domain_rows:
        pd.DataFrame(domain_rows).to_csv(
            out_dir / "domain_annotations.csv", index=False
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def align_labels_with_patches(
    labels: pd.DataFrame, meta: pd.DataFrame
) -> pd.DataFrame:
    return labels.merge(meta, on=["slide_id", "spot_id"], how="inner")
