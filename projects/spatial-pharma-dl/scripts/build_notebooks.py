#!/usr/bin/env python3
"""Generate Spatial Pharma DL notebooks."""

from __future__ import annotations

import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent.parent / "notebooks"
KERNEL = {"display_name": "spatial-tx", "language": "python", "name": "spatial-tx"}

SETUP = """from utils import st_helpers as st

ROOT, PHARMA = st.setup_pharma_paths()
st.set_seeds()
print('ROOT:', ROOT)
print('PHARMA:', PHARMA)
"""


def nb(cells: list) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": KERNEL,
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


NOTEBOOKS = {
    "01_data_curation.ipynb": [
        md("# 01 — Data Curation and Cohort QC\n\nDownload public Visium oncology slides and run tutorial-equivalent QC + Leiden clustering per slide."),
        code(SETUP),
        code("""from src.data import load_config, cohort_slide_ids, preprocess_cohort, cohort_summary, pharma_outputs_dir

cfg = load_config()
all_slides = cohort_slide_ids(cfg)
print('Slides to process:', len(all_slides))
for s in all_slides:
    print(' ', s)
"""),
        code("""paths = preprocess_cohort(all_slides, cfg=cfg)
list(paths.keys())
"""),
        code("""summary = cohort_summary(all_slides)
display(summary)
out = pharma_outputs_dir() / 'cohort_summary.csv'
summary.to_csv(out, index=False)
print('Wrote', out)
"""),
        code("""import matplotlib.pyplot as plt
import squidpy as sq
from src.data import load_slide

slide = cfg['cohorts']['oncology'][0]
adata = load_slide(slide)
sq.pl.spatial_scatter(adata, color=['total_counts', 'pct_counts_mt'], ncols=2, size=1.3)
plt.suptitle(slide)
fig_path = pharma_outputs_dir() / 'qc_spatial_breast.png'
plt.savefig(fig_path, dpi=120, bbox_inches='tight')
plt.show()
"""),
        md("**Next:** `02_label_engineering.ipynb`"),
    ],
    "02_label_engineering.ipynb": [
        md("# 02 — Label Engineering\n\nBuild per-spot labels: Leiden domains, marker genes, and module scores."),
        code(SETUP),
        code("""from src.data import load_config
from src.labels import build_labels_cohort, gene_columns, module_columns

cfg = load_config()
all_slides = cfg['cohorts']['oncology'] + cfg['cohorts']['external'] + cfg['cohorts']['benchmark']
labels = build_labels_cohort(all_slides, cfg=cfg)
print('Total spots:', len(labels))
print('Gene columns:', gene_columns(labels))
print('Module columns:', module_columns(labels))
labels.head()
"""),
        code("""domain_summary = labels.groupby(['slide_id', 'domain_name']).size().reset_index(name='n_spots')
display(domain_summary)
"""),
        md("**Next:** `03_patch_dataset.ipynb`"),
    ],
    "03_patch_dataset.ipynb": [
        md("# 03 — Patch Dataset Construction\n\nExtract H&E patches with Macenko stain normalization."),
        code(SETUP),
        code("""from src.data import load_config, pharma_outputs_dir
from src.labels import build_labels_cohort
from src.patches import build_patch_cohort, fit_reference_stain, save_patch_index

cfg = load_config()
oncology = cfg['cohorts']['oncology']
all_slides = oncology + cfg['cohorts']['external'] + cfg['cohorts']['benchmark']
labels = build_labels_cohort(all_slides, cfg=cfg)
ref_stain = fit_reference_stain(oncology, cfg)
ref_stain
"""),
        code("build_patch_cohort(all_slides, ref_stain=ref_stain, cfg=cfg)"),
        code("""idx_path = save_patch_index(labels)
print('Wrote', idx_path)
labels.groupby('slide_id').size()
"""),
        code("""import matplotlib.pyplot as plt
import numpy as np
from src.data import load_slide
from src.patches import extract_patch, coords_hires, patch_size_px, macenko_normalize, resize_patch
from utils import st_helpers as st

slide = oncology[0]
adata = load_slide(slide)
img = st.get_image(adata, 'hires')
_, half = patch_size_px(
    adata,
    min_patch=cfg['patches']['min_patch_px'],
    context_scale=cfg['patches'].get('context_scale', 1.0),
)
coords = coords_hires(adata)
fig, axes = plt.subplots(2, 6, figsize=(12, 4))
for ax, i in zip(axes.flat, np.linspace(0, len(coords)-1, 12, dtype=int)):
    x, y = coords[i]
    norm = macenko_normalize(extract_patch(img, x, y, half), ref_stain)
    ax.imshow(resize_patch(norm, 112)); ax.axis('off')
fig.savefig(pharma_outputs_dir() / 'stain_norm_montage.png', dpi=120, bbox_inches='tight')
plt.show()
"""),
        md("**Next:** `04_train_cnn.ipynb`"),
    ],
    "04_train_cnn.ipynb": [
        md("# 04 — CNN Training (LOSO)\n\nTrain ResNet18 with leave-one-slide-out validation on the breast cohort."),
        code(SETUP),
        code("""from src.data import load_config, pharma_outputs_dir
from src.labels import build_labels_cohort
from src.train import train_loso
import pandas as pd

cfg = load_config()
oncology = cfg['cohorts']['oncology']
labels = build_labels_cohort(oncology + cfg['cohorts']['external'] + cfg['cohorts']['benchmark'], cfg=cfg)
breast_labels = labels[labels['slide_id'].isin(oncology)]
breast_labels.groupby('slide_id').size()
"""),
        code("""results = train_loso(oncology, breast_labels, cfg=cfg)
for r in results:
    print(r['fold'], r['val_slide'], r['model_path'])
"""),
        code("""rows = []
for r in results:
    for h in r['history']:
        rows.append({'fold': r['fold'], 'val_slide': r['val_slide'], **h})
pd.DataFrame(rows).to_csv(pharma_outputs_dir() / 'training_history.csv', index=False)
"""),
        md("**Next:** `05_evaluation.ipynb`"),
    ],
    "05_evaluation.ipynb": [
        md("# 05 — Evaluation and Benchmark\n\nCNN vs radiomics Random Forest, with an optional frozen pathology-foundation-model linear probe. All arms use slide-level holdout."),
        code(SETUP),
        code("""import pandas as pd
from src.data import load_config, cohort_slide_ids
from src.labels import build_labels_cohort
from src.benchmark import run_and_save_benchmark

cfg = load_config()
# Optional research-only arm (Kaiko non-commercial weights):
# cfg['foundation']['enabled'] = True
oncology = cfg['cohorts']['oncology']
labels = build_labels_cohort(cohort_slide_ids(cfg), cfg=cfg)
breast_labels = labels[labels['slide_id'].isin(oncology)]
report_path, results = run_and_save_benchmark(oncology, breast_labels, cfg=cfg)
pd.read_csv(report_path)
"""),
        md("""### Frozen foundation-model arm

When `cfg['foundation']['enabled']` is true, the benchmark downloads the compact pathology encoder once, runs it strictly in inference mode, caches one embedding array per slide, and fits only logistic/ridge probes within each LOSO fold. The encoder receives no gradient updates.

**License:** the tutorial-sized `kaiko_vits16` checkpoint is non-commercial. Do not use its output for commercial pharma decisions."""),
        code("""from src.eval import predict_cnn
from src.patches import load_patch_arrays

for sid in cfg['cohorts']['external']:
    try:
        patches, _ = load_patch_arrays(sid, cfg=cfg)
        predict_cnn(results[-1]['model'], patches[:50], device=results[-1]['device'])
        print(sid, 'external OK, spots sampled: 50')
    except FileNotFoundError as e:
        print(sid, e)
"""),
        md("**Next:** `06_interpretability.ipynb`"),
    ],
    "06_interpretability.ipynb": [
        md("# 06 — Interpretability and Pharma Deliverables\n\nGrad-CAM and spatial prediction maps."),
        code(SETUP),
        code("""import matplotlib.pyplot as plt
import numpy as np
import squidpy as sq
from src.data import load_config, cohort_slide_ids, pharma_outputs_dir, load_slide
from src.labels import build_labels_cohort
from src.train import train_loso
from src.eval import grad_cam_for_patch, evaluate_fold

cfg = load_config()
oncology = cfg['cohorts']['oncology']
labels = build_labels_cohort(cohort_slide_ids(cfg), cfg=cfg)
breast_labels = labels[labels['slide_id'].isin(oncology)]
results = train_loso(oncology, breast_labels, cfg=cfg)
ev = evaluate_fold(results[0])
"""),
        code("""model, device = results[0]['model'], results[0]['device']
X_val, y_pred = results[0]['X_val'], ev['y_cls_pred']
y_true = ev['y_cls']
fig, axes = plt.subplots(2, 3, figsize=(9, 6))
for ax, idx in zip(axes[0], np.where(y_true == y_pred)[0][:3]):
    cam = grad_cam_for_patch(model, X_val[idx], int(y_pred[idx]), device=device)
    ax.imshow((X_val[idx].transpose(1,2,0)*255).astype(np.uint8))
    ax.imshow(cam, cmap='jet', alpha=0.45); ax.set_title('correct'); ax.axis('off')
for ax, idx in zip(axes[1], np.where(y_true != y_pred)[0][:3]):
    cam = grad_cam_for_patch(model, X_val[idx], int(y_pred[idx]), device=device)
    ax.imshow((X_val[idx].transpose(1,2,0)*255).astype(np.uint8))
    ax.imshow(cam, cmap='jet', alpha=0.45); ax.set_title('incorrect'); ax.axis('off')
fig.savefig(pharma_outputs_dir() / 'gradcam_montage.png', dpi=120, bbox_inches='tight')
plt.show()
"""),
        code("""val_slide = results[0]['val_slide']
adata = load_slide(val_slide)
adata.obs['pred_cluster'] = 'NA'
adata.obs.loc[results[0]['lab_val']['spot_id'], 'pred_cluster'] = y_pred.astype(str)
sq.pl.spatial_scatter(adata, color='pred_cluster', size=1.3)
plt.savefig(pharma_outputs_dir() / 'spatial_predictions.png', dpi=120, bbox_inches='tight')
plt.show()
"""),
        md("Compare the run with the frozen-model findings in [`PROJECT_REPORT.md`](../PROJECT_REPORT.md).\n\n**End of pipeline.**"),
    ],
}


def main():
    NB_DIR.mkdir(parents=True, exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        with open(NB_DIR / name, "w") as f:
            json.dump(nb(cells), f, indent=1)
        print("Wrote", NB_DIR / name)


if __name__ == "__main__":
    main()
