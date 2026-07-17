# Spatial Pharma DL

This mini-project asks whether H&E morphology can act as a low-cost surrogate for
spatial transcriptomics when mapping tumor epithelial, immune, and stromal regions.
It compares handcrafted radiomics, an ImageNet-pretrained ResNet18, and frozen
pathology foundation models on public 10x Visium oncology slides.

The central validation rule is simple: **the held-out unit is a complete slide,
never an individual spot**.

## Current findings

The initial pipeline underperformed because slide-local Leiden cluster integers were
treated as if they represented the same biology across slides. It also used very small
native patches, single-gene targets, and no training augmentation. The current v2
configuration replaces those choices with harmonized TME classes, module-score
regression, 3× spot context, per-slide Macenko normalization, and lightweight image
augmentation.

The strongest completed experiment uses frozen foundation-model embeddings and nested
leave-one-slide-out model selection:

| Task | Kaiko ViT-S/16 | Phikon ViT-B | Majority baseline | Coverage |
|---|---:|---:|---:|---:|
| All four labels | 0.294 | **0.320** | 0.187 | 100% |
| Confident three labels | 0.326 | **0.366** | 0.211 | 39.0% |

Values are mean outer-fold macro-F1. Phikon transfers better, but its weakest fold is
0.227. Removing the ambiguous `other` label improves F1 by answering an easier question
and discarding 61% of spots; it is not a like-for-like replacement metric.

Slide z-scoring plus L2 normalization was selected in all all-class outer folds. It
reduces linear PCA slide silhouette from 0.357 to -0.001, although UMAP still separates
slides nonlinearly. Residual domain shift and label quality—not a lack of model size—are
now the main constraints.

See the executed
[`07_foundation_model_comparison.ipynb`](notebooks/07_foundation_model_comparison.ipynb)
and the consolidated [`PROJECT_REPORT.md`](PROJECT_REPORT.md).

## Workflow

1. Curate four breast slides plus external CRC, ovarian, and GBM slides.
2. Perform QC, normalization, PCA, neighborhood construction, and Leiden clustering.
3. Map slide-local domains into globally consistent TME labels and compute molecular
   module scores.
4. Extract 224×224 H&E patches using 3× spot context and per-slide stain normalization.
5. Benchmark radiomics RF, ResNet18, and optional frozen foundation-model probes.
6. Evaluate with slide-level holdout and inspect confusion, spatial, and domain-shift
   diagnostics.

The six core notebooks implement the general pipeline. Notebook 07 is the consolidated,
executed foundation-model experiment.

## Quickstart

From the repository root:

```bash
conda activate spatial-tx
pip install -r requirements.txt -r projects/spatial-pharma-dl/requirements-pharma.txt
python projects/spatial-pharma-dl/scripts/run_pipeline.py
```

Useful modes:

```bash
# Reuse processed data and cached patches.
PHARMA_TRAIN_ONLY=1 python projects/spatial-pharma-dl/scripts/run_pipeline.py

# Small smoke run.
PHARMA_QUICK=1 PHARMA_TRAIN_ONLY=1 \
  python projects/spatial-pharma-dl/scripts/run_pipeline.py

# Include the configured frozen encoder in the standard benchmark.
PHARMA_FOUNDATION=1 PHARMA_TRAIN_ONLY=1 \
  python projects/spatial-pharma-dl/scripts/run_pipeline.py
```

Open the notebook workflow with:

```bash
jupyter lab projects/spatial-pharma-dl/notebooks/
```

## Frozen foundation models

The encoder is always frozen. Patches are embedded once, cached under
`data/processed/pharma/foundation_embeddings/<model>/`, and reused by fold-local
logistic/ridge probes.

Configure the standard benchmark in `configs/default.yaml`:

```yaml
foundation:
  enabled: false
  model: kaiko_vits16  # or phikon
```

The dedicated comparison notebook evaluates Kaiko and Phikon with inner LOSO selection
of preprocessing, regularization, and class weighting inside every outer held-out-slide
fold. Rebuild its source notebook with:

```bash
python projects/spatial-pharma-dl/scripts/build_foundation_notebook.py
```

Model sources: [Kaiko ViT-S/16](https://huggingface.co/1aurent/vit_small_patch16_224.kaiko_ai_towards_large_pathology_fms)
and [Phikon](https://huggingface.co/owkin/phikon). Both tested checkpoints have
non-commercial research terms and are unsuitable for a commercial pharma workflow
without legal and model-governance review.

## Repository map

| Path | Purpose |
|---|---|
| [`configs/default.yaml`](configs/default.yaml) | Cohorts, labels, patches, training, and encoder settings |
| [`src/`](src/) | Data, label, patch, training, evaluation, and foundation-model modules |
| [`notebooks/`](notebooks/) | Six core workflow notebooks plus one consolidated FM comparison |
| [`scripts/build_notebooks.py`](scripts/build_notebooks.py) | Rebuild notebooks 01–06 |
| [`scripts/build_foundation_notebook.py`](scripts/build_foundation_notebook.py) | Rebuild notebook 07 |
| [`scripts/run_pipeline.py`](scripts/run_pipeline.py) | End-to-end command-line runner |
| [`tests/test_foundation.py`](tests/test_foundation.py) | Frozen-encoder and nested-LOSO checks |
| [`PROJECT_REPORT.md`](PROJECT_REPORT.md) | Consolidated methods, results, caveats, and recommendations |

Generated outputs are written to `outputs/pharma/` and are intentionally ignored by Git.

## Limitations

- Four slides are not four independent patients; two are adjacent sections.
- TME labels are heuristic molecular annotations, not pathologist ground truth.
- Visium spots are multicellular and spatially autocorrelated.
- Whole-slide z-scoring is label-free but transductive and cannot support isolated-patch
  inference.
- Public data contain no treatment arms, so pharmacodynamic claims require proprietary
  cohorts.
- The tested foundation-model weights are research/non-commercial.

The next credible route to higher F1 is better annotation and more independent patients,
followed by multiscale tiles, spatial-neighbor aggregation, and an explicit abstention
curve. Fine-tuning should wait until the cohort can distinguish transferable biology from
slide identity.
