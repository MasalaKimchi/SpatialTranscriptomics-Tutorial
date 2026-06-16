# Analysis Strategy — Spatial Pharma DL

Phase-by-phase protocol for histology-to-TME molecular profiling.

---

## Phase 1 — Data curation and cohort QC

**Notebook:** `notebooks/01_data_curation.ipynb`

### Steps

1. Download Tier 1 (breast) + Tier 2 (external) slides via `squidpy.datasets.visium`
2. Per slide, run tutorial-equivalent pipeline:
   - `sc.pp.calculate_qc_metrics` (mito genes via `var['mt']`)
   - Filter: min_counts=500, min_cells=3, pct_mito < 30
   - `normalize_total` → `log1p` → HVG (2000) → scale → PCA (50)
   - Neighbors (15, 30 PCs) → UMAP → Leiden (resolution=1.0)
3. Save: `data/processed/pharma/<sample_id>_clustered.h5ad`
4. Export: `outputs/pharma/cohort_summary.csv`

### QC checks

- Spot counts before/after filtering
- Median genes and mito% per slide
- Spatial QC montage (counts, mito on tissue)

---

## Phase 2 — Label engineering

**Notebook:** `notebooks/02_label_engineering.ipynb`

### Task families

| Task | Labels | Source |
|------|--------|--------|
| Domain classification | Leiden cluster ID | `adata.obs['clusters']` |
| Marker regression | log-normalized expression | `adata[:, gene].X` for present markers |
| Module scores | Immune, ECM, proliferation, hypoxia | `sc.tl.score_genes` |

### Domain annotation

Post-hoc: rank marker genes per cluster → assign human-readable names
(`immune_enriched`, `stromal`, `tumor_epithelial`, etc.).

### Outputs

- `outputs/pharma/labels_<sample_id>.parquet`
- `outputs/pharma/domain_annotations.csv`

---

## Phase 3 — Patch dataset construction

**Notebook:** `notebooks/03_patch_dataset.ipynb`

### Patch extraction

- Center on `obsm['spatial'] * tissue_hires_scalef`
- Size from `spot_diameter_fullres * scalef` (min 8 px)
- Resize to 224×224 for ResNet18

### Stain normalization

Macenko method fit on reference slide (first breast section), applied to all slides.

### Index file

`data/processed/pharma/patch_index.parquet` columns:

- `slide_id`, `spot_id`, `x`, `y`, `cluster`, `gene_*`, `module_*`

**Critical:** `slide_id` is the split unit — never random spot splits.

---

## Phase 4 — Model training

**Notebook:** `notebooks/04_train_cnn.ipynb`

### Architecture

- ResNet18 (ImageNet pretrained)
- Shared backbone → classification head (n_clusters) + regression head (n_genes)
- Loss = cls_weight × CE + reg_weight × MSE

### Training protocol

- **3-fold LOSO** on Tier 1 breast slides (hold out 1 slide per fold)
- AdamW, lr=1e-4, early stopping (patience=3)
- Batch size 32, max 15 epochs

### Baseline

Re-run notebook-10-style RF on same patches (15 radiomics features) for comparison.

### Outputs

- `outputs/pharma/models/resnet18_fold{N}.pt`
- `outputs/pharma/metrics_per_fold.csv`
- `outputs/pharma/training_curves.png`

---

## Phase 5 — Evaluation and external validation

**Notebook:** `notebooks/05_evaluation.ipynb`

### Metrics

| Task | Primary | Secondary |
|------|---------|-----------|
| Classification | Balanced accuracy, macro-F1 | Confusion matrix per slide |
| Regression | Pearson r, R² per gene | MAE, calibration plot |

### Comparisons

1. RF vs CNN — same labels, same LOSO splits
2. In-distribution (breast) vs external (CRC, ovarian, GBM) zero-shot

### Reporting rule

All metrics reported **per slide** and **per fold**. No pooled random spot metrics.

### Outputs

- `outputs/pharma/benchmark_report.csv`
- Spatial prediction maps on held-out slides
- Failure case gallery

---

## Phase 6 — Interpretability and deliverables

**Notebook:** `notebooks/06_interpretability.ipynb`

### Analyses

- Grad-CAM on correct vs incorrect predictions
- Spatial overlay: predicted vs actual gene expression
- ROC per domain, gene correlation heatmaps

### Documents

- Fill `PROJECT_REPORT.md` with results after running all phases

---

## Decision log

| Decision | Rationale |
|----------|-----------|
| ResNet18 over custom CNN | Pharma-standard, pretrained, simple |
| LOSO over random split | Spatial autocorrelation causes leakage |
| Macenko stain norm | Cross-slide scanner/stain variation |
| Multi-task learning | Domains + genes jointly improve representation |
| Breast primary cohort | Most clinically documented public Visium oncology data |
