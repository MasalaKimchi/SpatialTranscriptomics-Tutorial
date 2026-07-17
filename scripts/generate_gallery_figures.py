"""Generate canonical figure gallery PNGs into outputs/figures/.

Requires processed AnnData caches (run notebooks 02-10 first, or this script
will fetch/process minimally).

Run:  conda activate spatial-tx && python scripts/generate_gallery_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import squidpy as sq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from utils import st_helpers as st  # noqa: E402

st.set_seeds()
sc.settings.set_figure_params(dpi=100, facecolor="white")


def _need(path: str) -> None:
    if not (st.processed_dir() / path).exists():
        raise FileNotFoundError(
            f"Missing {path}. Run the tutorial notebooks through the step that creates it."
        )


def fig_qc_histograms(adata) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist(adata.obs["total_counts"], bins=60)
    axes[0].set(title="Total UMI counts per spot", xlabel="total_counts")
    axes[1].hist(adata.obs["n_genes_by_counts"], bins=60, color="tab:green")
    axes[1].set(title="Genes detected per spot", xlabel="n_genes_by_counts")
    axes[2].hist(adata.obs["pct_counts_mt"], bins=60, color="tab:red")
    axes[2].set(title="Mitochondrial % per spot", xlabel="pct_counts_mt")
    plt.tight_layout()
    st.save_fig(fig, "04_qc_histograms.png")
    plt.close(fig)


def fig_spatial_qc(adata) -> None:
    sq.pl.spatial_scatter(
        adata, color=["total_counts", "pct_counts_mt"], ncols=2, size=1.3
    )
    st.save_fig(plt.gcf(), "04_spatial_qc.png")
    plt.close("all")


def fig_he_overview(adata) -> None:
    hires = st.get_image(adata, "hires")
    lowres = st.get_image(adata, "lowres")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(hires)
    axes[0].set_title("H&E hires")
    axes[0].axis("off")
    axes[1].imshow(lowres)
    axes[1].set_title("H&E lowres")
    axes[1].axis("off")
    plt.tight_layout()
    st.save_fig(fig, "05_he_overview.png")
    plt.close(fig)


def fig_markers_spatial(adata) -> None:
    markers = st.genes_present(adata, ["Mbp", "Snap25", "Gfap", "Plp1"], verbose=False)[
        :4
    ]
    if markers:
        sq.pl.spatial_scatter(adata, color=markers, ncols=2, size=1.3)
        st.save_fig(plt.gcf(), "06_marker_spatial.png")
        plt.close("all")


def fig_umap_clusters(adata) -> None:
    sc.pl.umap(adata, color="clusters", legend_loc="on data")
    st.save_fig(plt.gcf(), "07_umap_clusters.png")
    plt.close("all")


def fig_spatial_clusters(adata) -> None:
    sq.pl.spatial_scatter(adata, color="clusters", size=1.4)
    st.save_fig(plt.gcf(), "07_spatial_clusters.png")
    plt.close("all")


def fig_cluster_dotplot(adata) -> None:
    if "rank_genes_groups" not in adata.uns:
        sc.tl.rank_genes_groups(adata, "clusters", method="wilcoxon")
    markers = sc.get.rank_genes_groups_df(adata, group=None)
    top3 = (
        markers.sort_values(["group", "scores"], ascending=[True, False])
        .groupby("group")
        .head(3)["names"]
        .tolist()
    )
    top3 = st.genes_present(adata, dict.fromkeys(top3), verbose=False)[:18]
    if top3:
        sc.pl.dotplot(adata, var_names=top3, groupby="clusters")
        st.save_fig(plt.gcf(), "07_cluster_markers_dotplot.png")
        plt.close("all")


def fig_integration(adata) -> None:
    if "img_features" not in adata.obsm:
        return
    feat_names = list(adata.uns.get("img_feature_names", []))
    if not feat_names:
        return
    Ximg = pd.DataFrame(
        adata.obsm["img_features"], columns=feat_names, index=adata.obs_names
    )
    neuronal = st.genes_present(adata, ["Snap25", "Mbp", "Gfap"], verbose=False)
    if neuronal:
        sc.tl.score_genes(
            adata, neuronal, score_name="sig_neuronal", random_state=st.SEED
        )
        corr = Ximg.apply(lambda col: col.corr(adata.obs["sig_neuronal"]))
        fig, ax = plt.subplots(figsize=(6, 5))
        corr.sort_values().plot(kind="barh", ax=ax, color="steelblue")
        ax.set_title("Image feature correlation with neuronal signature")
        ax.set_xlabel("Pearson r")
        plt.tight_layout()
        st.save_fig(fig, "10_feature_correlation.png")
        plt.close(fig)


def main() -> None:
    print("Generating gallery figures ->", st.figures_dir())

    _need("adata_qc.h5ad")
    adata_qc = st.load_adata("adata_qc.h5ad")
    fig_qc_histograms(adata_qc)
    fig_spatial_qc(adata_qc)
    fig_he_overview(adata_qc)
    fig_markers_spatial(adata_qc)
    print("  04-06 figures OK")

    _need("adata_clustered.h5ad")
    adata_cl = st.load_adata("adata_clustered.h5ad")
    fig_umap_clusters(adata_cl)
    fig_spatial_clusters(adata_cl)
    fig_cluster_dotplot(adata_cl)
    print("  07 figures OK")

    if (st.processed_dir() / "adata_features.h5ad").exists():
        adata_feat = st.load_adata("adata_features.h5ad")
        fig_integration(adata_feat)
        print("  10 figures OK")

    saved = sorted(st.figures_dir().glob("*.png"))
    print(f"Gallery complete: {len(saved)} PNGs in outputs/figures/")
    for p in saved:
        print(" ", p.name)


if __name__ == "__main__":
    main()
