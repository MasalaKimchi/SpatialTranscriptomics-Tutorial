"""Patch tutorial notebooks: pin kernel + insert enhanced visuals and exercises.

Run from repo root:  python scripts/patch_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text,
    }


def insert(nb: dict, idx: int, cell: dict) -> None:
    """Insert a cell once, preserving rerun idempotence."""
    if any(
        existing.get("cell_type") == cell.get("cell_type")
        and existing.get("source") == cell.get("source")
        for existing in nb["cells"]
    ):
        return
    nb["cells"].insert(idx, cell)


def pin_kernel(nb: dict) -> None:
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python (spatial-tx)",
        "language": "python",
        "name": "spatial-tx",
    }
    nb["metadata"]["language_info"] = {
        "name": "python",
        "version": "3.11",
    }


def patch_00(nb: dict) -> None:
    insert(nb, 7, md(
        "### Conceptual diagrams (saved to the figure gallery)\n"
        "These static figures summarize the Visium workflow and the AnnData container.\n"
        "They are also written to `outputs/figures/` so you can browse them from the README\n"
        "without opening this notebook."
    ))
    insert(nb, 8, code(
        "import sys\n"
        "from pathlib import Path\n"
        "import matplotlib.pyplot as plt\n"
        "import matplotlib.patches as mpatches\n"
        "from matplotlib.patches import FancyBboxPatch, FancyArrowPatch\n"
        "\n"
        "ROOT = Path.cwd()\n"
        "if not (ROOT / 'utils').exists():\n"
        "    ROOT = ROOT.parent\n"
        "sys.path.insert(0, str(ROOT))\n"
        "from utils import st_helpers as st\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(12, 3.5))\n"
        "ax.set_xlim(0, 12); ax.set_ylim(0, 3.5); ax.axis('off')\n"
        "ax.set_title('10x Visium workflow (one tissue section)', fontsize=13, fontweight='bold')\n"
        "steps = [\n"
        "    ('Tissue on\\nVisium slide', 0.3),\n"
        "    ('H&E stain\\n+ image', 2.3),\n"
        "    ('mRNA capture\\nat each spot', 4.3),\n"
        "    ('Spot barcode\\n+ UMI', 6.3),\n"
        "    ('Sequencing', 8.3),\n"
        "    ('Spots x genes\\ncount matrix', 10.3),\n"
        "]\n"
        "for label, x in steps:\n"
        "    box = FancyBboxPatch((x, 1.0), 1.6, 1.2, boxstyle='round,pad=0.05',\n"
        "                         facecolor='#e8f4fc', edgecolor='#2b6cb0', linewidth=1.5)\n"
        "    ax.add_patch(box)\n"
        "    ax.text(x + 0.8, 1.6, label, ha='center', va='center', fontsize=9)\n"
        "for i in range(len(steps) - 1):\n"
        "    ax.annotate('', xy=(steps[i+1][1], 1.6), xytext=(steps[i][1] + 1.6, 1.6),\n"
        "                arrowprops=dict(arrowstyle='->', color='#4a5568', lw=1.5))\n"
        "ax.text(6, 0.3, 'Registration: spot pixel coords + scale factors link matrix rows to the H&E image',\n"
        "        ha='center', fontsize=9, style='italic', color='#4a5568')\n"
        "path = st.save_fig(fig, '00_visium_assay.png')\n"
        "plt.show()\n"
        "print('Saved', path)\n"
    ))
    insert(nb, 9, code(
        "import sys\n"
        "from pathlib import Path\n"
        "import matplotlib.pyplot as plt\n"
        "from matplotlib.patches import FancyBboxPatch\n"
        "\n"
        "ROOT = Path.cwd()\n"
        "if not (ROOT / 'utils').exists():\n"
        "    ROOT = ROOT.parent\n"
        "sys.path.insert(0, str(ROOT))\n"
        "from utils import st_helpers as st\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(10, 5))\n"
        "ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis('off')\n"
        "ax.set_title('AnnData container (Visium study object)', fontsize=13, fontweight='bold')\n"
        "slots = [\n"
        "    ('.X', 'spots x genes\\nUMI count matrix', '#fef3c7', 0.5, 3.2),\n"
        "    ('.obs', 'per-spot metadata\\n(QC, clusters)', '#d1fae5', 3.0, 3.2),\n"
        "    ('.var', 'per-gene metadata\\n(HVG flags)', '#fde68a', 5.5, 3.2),\n"
        "    ('.obsm', 'embeddings & coords\\n(spatial, PCA, UMAP)', '#dbeafe', 0.5, 0.8),\n"
        "    ('.uns', 'H&E image + scale factors', '#fce7f3', 3.0, 0.8),\n"
        "    ('.layers', 'raw counts, etc.', '#e5e7eb', 5.5, 0.8),\n"
        "]\n"
        "for name, desc, color, x, y in slots:\n"
        "    box = FancyBboxPatch((x, y), 2.2, 1.6, boxstyle='round,pad=0.05',\n"
        "                         facecolor=color, edgecolor='#374151', linewidth=1.2)\n"
        "    ax.add_patch(box)\n"
        "    ax.text(x + 1.1, y + 1.15, name, ha='center', fontweight='bold', fontsize=11)\n"
        "    ax.text(x + 1.1, y + 0.55, desc, ha='center', fontsize=8)\n"
        "path = st.save_fig(fig, '00_anndata_schema.png')\n"
        "plt.show()\n"
        "print('Saved', path)\n"
    ))
    insert(nb, 10, md(
        "<details>\n"
        "<summary><b>Your turn (exercise)</b></summary>\n"
        "\n"
        "**Prompt:** In your own words, name the four linked data objects in a Visium dataset and\n"
        "explain which AnnData slot holds each one.\n"
        "\n"
        "**Expected answer:** (1) **Count matrix** → `adata.X`; (2) **spot metadata** → `adata.obs`;\n"
        "(3) **spatial coordinates** → `adata.obsm['spatial']`; (4) **H&E image + scale factors** →\n"
        "`adata.uns['spatial']`. Registration ties spot rows to pixel locations via the scale factors.\n"
        "</details>\n"
    ))


def patch_03(nb: dict) -> None:
    insert(nb, 17, md(
        "### Registration check: misaligned vs aligned overlay\n"
        "The most common registration bug is plotting **full-resolution** spot coordinates on the\n"
        "**downscaled** `hires` image *without* multiplying by `tissue_hires_scalef`. The left panel\n"
        "below shows that failure; the right panel shows the correct multiply."
    ))
    insert(nb, 18, code(
        "fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n"
        "for ax, xy, title in [\n"
        "    (axes[0], coords, 'WRONG: full-res coords on hires image'),\n"
        "    (axes[1], xy_hires, 'CORRECT: coords x tissue_hires_scalef'),\n"
        "]:\n"
        "    ax.imshow(hires)\n"
        "    ax.scatter(xy[:, 0], xy[:, 1], s=4, c='cyan', alpha=0.5)\n"
        "    ax.set_title(title, fontsize=10)\n"
        "    ax.axis('off')\n"
        "plt.tight_layout()\n"
        "st.save_fig(fig, '03_registration_misaligned_vs_aligned.png')\n"
        "plt.show()\n"
        "print('Misaligned dots sit in a corner or off-tissue; aligned dots tile the tissue.')\n"
    ))
    insert(nb, 19, md(
        "<details>\n"
        "<summary><b>Your turn (exercise)</b></summary>\n"
        "\n"
        "**Prompt:** A colleague overlays spots on the hires image and they land in the top-left\n"
        "corner. What did they forget?\n"
        "\n"
        "**Expected answer:** They used `adata.obsm['spatial']` directly without multiplying by\n"
        "`adata.uns['spatial'][lib]['scalefactors']['tissue_hires_scalef']`. Full-res pixel\n"
        "coordinates must be scaled down to match the hires image resolution.\n"
        "</details>\n"
    ))


def patch_06(nb: dict) -> None:
    insert(nb, 10, md("### Dotplot: marker expression across spots (grouped view)"))
    insert(nb, 11, code(
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "# Dotplot needs a grouping variable; we bin spots by capture depth (total UMI counts).\n"
        "adata.obs['_depth_bin'] = pd.qcut(\n"
        "    adata.obs['total_counts'], q=4,\n"
        "    labels=['low', 'mid-low', 'mid-high', 'high'], duplicates='drop',\n"
        ")\n"
        "if present:\n"
        "    sc.pl.dotplot(adata, var_names=present, groupby='_depth_bin',\n"
        "                  standard_scale='var', show=False)\n"
        "    fig = plt.gcf()\n"
        "    st.save_fig(fig, '06_marker_dotplot.png')\n"
        "    plt.show()\n"
    ))
    insert(nb, 12, md(
        "<details>\n"
        "<summary><b>Your turn (exercise)</b></summary>\n"
        "\n"
        "**Prompt:** Pick one marker from the spatial plots. Does its high-expression region match\n"
        "what you would expect from brain anatomy (e.g. myelin in white matter)?\n"
        "\n"
        "**Expected answer:** `Mbp`/`Plp1` should be high in white-matter tracts; `Snap25` broadly\n"
        "in gray matter; region-specific markers like `Hpca` highlight substructures. If a marker\n"
        "looks random, check gene-name casing or whether the gene is truly present in the dataset.\n"
        "</details>\n"
    ))


def patch_07(nb: dict) -> None:
    insert(nb, 14, md(
        "### Top marker genes per cluster — spatial panels\n"
        "For the three largest clusters we plot the #1 ranked marker spatially. This links each\n"
        "transcriptomic domain to a concrete molecular signature on the tissue."
    ))
    insert(nb, 15, code(
        "import matplotlib.pyplot as plt\n"
        "\n"
        "top1 = (top_markers.sort_values(['group', 'scores'], ascending=[True, False])\n"
        "                   .groupby('group').head(1))\n"
        "top_clusters = adata.obs['clusters'].value_counts().head(3).index.astype(str).tolist()\n"
        "spatial_genes = []\n"
        "for cl in top_clusters:\n"
        "    row = top1[top1['group'].astype(str) == cl]\n"
        "    if len(row):\n"
        "        g = row.iloc[0]['names']\n"
        "        if g in adata.var_names:\n"
        "            spatial_genes.append(g)\n"
        "spatial_genes = list(dict.fromkeys(spatial_genes))  # dedupe\n"
        "print('Top markers for largest clusters:', spatial_genes)\n"
        "if spatial_genes:\n"
        "    sq.pl.spatial_scatter(adata, color=spatial_genes, ncols=3, size=1.3)\n"
        "    fig = plt.gcf()\n"
        "    st.save_fig(fig, '07_top_cluster_markers_spatial.png')\n"
    ))
    insert(nb, 16, md(
        "<details>\n"
        "<summary><b>Your turn (exercise)</b></summary>\n"
        "\n"
        "**Prompt:** Cluster 0's top marker gene — does its spatial pattern match the cluster's\n"
        "spatial domain on the tissue?\n"
        "\n"
        "**Expected answer:** Yes, for a well-separated cluster the top marker should be enriched\n"
        "exactly where that cluster sits on the histology overlay. Mismatches suggest mixed spots,\n"
        "over-clustering, or a marker that is high in multiple regions.\n"
        "</details>\n"
    ))


def patch_08(nb: dict) -> None:
    insert(nb, 9, md("### Moran's I rank plot (top spatially structured genes)"))
    insert(nb, 10, code(
        "import matplotlib.pyplot as plt\n"
        "\n"
        "top20 = moran.head(20)\n"
        "fig, ax = plt.subplots(figsize=(8, 5))\n"
        "ax.barh(top20.index[::-1], top20['I'].values[::-1], color='steelblue')\n"
        "ax.set_xlabel(\"Moran's I\")\n"
        "ax.set_title('Top 20 spatially variable genes')\n"
        "plt.tight_layout()\n"
        "st.save_fig(fig, '08_moran_rank.png')\n"
        "plt.show()\n"
    ))
    insert(nb, 11, md(
        "### Counterexample: high variance but weak spatial structure\n"
        "Find a gene that is highly variable (HVG) but **not** in the top SVGs — its expression\n"
        "varies across spots but without coherent spatial patches."
    ))
    insert(nb, 12, code(
        "import numpy as np\n"
        "\n"
        "hvg_set = set(adata.var_names[adata.var['highly_variable']]) if 'highly_variable' in adata.var else set()\n"
        "top_svg_set = set(moran.head(50).index)\n"
        "candidates = [g for g in adata.var_names if g in hvg_set and g not in top_svg_set]\n"
        "counter = None\n"
        "if candidates and 'dispersions_norm' in adata.var:\n"
        "    disp = adata.var.loc[candidates, 'dispersions_norm']\n"
        "    counter = disp.idxmax()\n"
        "    I_val = moran.loc[counter, 'I'] if counter in moran.index else float('nan')\n"
        "    print(f'Counterexample gene: {counter}  (Moran I = {I_val:.3f})')\n"
        "    sq.pl.spatial_scatter(adata, color=counter, size=1.3, cmap='viridis')\n"
        "    fig = plt.gcf()\n"
        "    st.save_fig(fig, '08_hvg_not_svg_counterexample.png')\n"
        "else:\n"
        "    print('Could not find a counterexample; inspect moran and HVG tables manually.')\n"
    ))
    insert(nb, 13, md(
        "<details>\n"
        "<summary><b>Your turn (exercise)</b></summary>\n"
        "\n"
        "**Prompt:** Why might a gene be highly variable across spots but have low Moran's I?\n"
        "\n"
        "**Expected answer:** Its expression changes a lot between spots but **without spatial\n"
        "autocorrelation** — e.g. random salt-and-pepper noise, spot-level technical variation, or\n"
        "a program active in scattered cell types rather than coherent tissue regions.\n"
        "</details>\n"
    ))


def patch_09(nb: dict) -> None:
    insert(nb, 3, md(
        "### Inspect per-spot image patches (before feature extraction)\n"
        "Each Visium spot covers a ~55 µm circle on the tissue. Below we crop a square patch\n"
        "centered on 12 example spots so you can see exactly what the 'radiomics' features will\n"
        "summarize. Patch size is derived from `spot_diameter_fullres * tissue_hires_scalef`."
    ))
    insert(nb, 4, code(
        "import matplotlib.pyplot as plt\n"
        "\n"
        "rng = np.random.default_rng(st.SEED)\n"
        "n_show = 12\n"
        "idx = rng.choice(adata.n_obs, size=min(n_show, adata.n_obs), replace=False)\n"
        "fig, axes = plt.subplots(3, 4, figsize=(10, 7))\n"
        "for ax, i in zip(axes.ravel(), idx):\n"
        "    x, y = coords_hires[i]\n"
        "    xi, yi = int(round(x)), int(round(y))\n"
        "    x0, x1 = max(0, xi - half), min(img.shape[1], xi + half)\n"
        "    y0, y1 = max(0, yi - half), min(img.shape[0], yi + half)\n"
        "    patch_img = img[y0:y1, x0:x1]\n"
        "    ax.imshow(patch_img)\n"
        "    ax.set_title(adata.obs_names[i][:12], fontsize=7)\n"
        "    ax.axis('off')\n"
        "fig.suptitle(f'Example spot patches ({patch} px, hires space)', fontsize=11)\n"
        "plt.tight_layout()\n"
        "st.save_fig(fig, '09_spot_patch_montage.png')\n"
        "plt.show()\n"
    ))
    insert(nb, 5, md(
        "<details>\n"
        "<summary><b>Your turn (exercise)</b></summary>\n"
        "\n"
        "**Prompt:** Look at three patches. Can you guess whether each spot sits in white matter,\n"
        "gray matter, or a mixed boundary from morphology alone?\n"
        "\n"
        "**Expected answer:** White matter looks more homogeneous and pink (myelin-rich); gray matter\n"
        "shows denser, darker nuclei (hematoxylin); boundaries look mixed. In notebook 10 we test\n"
        "whether these visual cues predict gene expression.\n"
        "</details>\n"
    ))


def patch_11(nb: dict) -> None:
    insert(nb, 7, md("### Module score distributions by cluster (violin plots)"))
    insert(nb, 8, code(
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "\n"
        "if scored:\n"
        "    plot_df = adata.obs[['clusters'] + scored].copy()\n"
        "    plot_df = plot_df.melt(id_vars='clusters', var_name='module', value_name='score')\n"
        "    fig, ax = plt.subplots(figsize=(12, 5))\n"
        "    sns.violinplot(data=plot_df, x='clusters', y='score', hue='module',\n"
        "                   split=False, inner='box', ax=ax)\n"
        "    ax.set_title('Module scores by Leiden cluster')\n"
        "    ax.set_xlabel('cluster')\n"
        "    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)\n"
        "    plt.tight_layout()\n"
        "    st.save_fig(fig, '11_module_scores_violin.png')\n"
        "    plt.show()\n"
    ))
    insert(nb, 9, md(
        "<details>\n"
        "<summary><b>Your turn (exercise)</b></summary>\n"
        "\n"
        "**Prompt:** Which cluster has the highest oligodendrocyte (`score_oligodendrocyte`) module\n"
        "score? Does that match where you expect myelin on the tissue?\n"
        "\n"
        "**Expected answer:** The cluster with the highest `score_oligodendrocyte` should correspond\n"
        "to white-matter domains on the spatial map (high `Mbp`/`Plp1` region). Module scores are\n"
        "relative program strengths, not cell-type fractions.\n"
        "</details>\n"
    ))


def main() -> None:
    patches = {
        "00_overview_spatial_transcriptomics.ipynb": patch_00,
        "03_load_expression_and_spatial_metadata.ipynb": patch_03,
        "06_spatial_visualization.ipynb": patch_06,
        "07_clustering_and_spatial_domains.ipynb": patch_07,
        "08_spatially_variable_genes.ipynb": patch_08,
        "09_image_feature_extraction_from_histology.ipynb": patch_09,
        "11_cell_type_annotation_and_deconvolution_optional.ipynb": patch_11,
    }

    for name in sorted(ROOT.glob("*.ipynb")):
        nb = json.loads(name.read_text())
        pin_kernel(nb)
        if name.name in patches:
            patches[name.name](nb)
            print(f"patched content: {name.name}")
        else:
            print(f"pinned kernel:   {name.name}")
        name.write_text(json.dumps(nb, indent=1))

    print("Done.")


if __name__ == "__main__":
    main()
