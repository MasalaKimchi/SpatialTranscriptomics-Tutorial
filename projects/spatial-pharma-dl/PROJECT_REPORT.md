# Spatial Pharma DL — Consolidated Report

## Executive summary

This project evaluates whether routinely available H&E morphology can predict spatial
tumor-microenvironment state measured by 10x Visium. The current evidence supports a
research prototype, not a decision-grade biomarker.

The original CNN/RF experiment produced weak cross-slide classification because Leiden
cluster identifiers were slide-local and therefore contradictory when pooled. The v2
pipeline harmonizes labels, increases histological context, predicts smoother molecular
programs, and adds stain normalization and augmentation.

The strongest completed classification result comes from frozen pathology foundation
models. Under nested slide-level validation, Phikon reaches 0.320 mean macro-F1 on all
four TME labels, compared with 0.294 for Kaiko and 0.187 for a fold-matched majority
baseline. A selective three-class task reaches 0.366 but retains only 39.0% of spots.

**Recommendation:** share the workflow and findings with caveats. Improve labels and add
independent patients before fine-tuning or making biological-performance claims.

## Project question

> Can cheap H&E serve as a surrogate for expensive spatial transcriptomics when
> localizing tumor epithelium, immune infiltration, stromal remodeling, and molecular
> programs in oncology tissue?

The intended deliverables are slide-held-out performance estimates, spatial diagnostics,
and a reproducible comparison of radiomics, CNN, and frozen foundation representations.

## Cohort

| Slide | Spots | Role |
|---|---:|---|
| V1 Breast Cancer Block A Section 1 | 3,798 | Primary breast LOSO |
| V1 Breast Cancer Block A Section 2 | 3,985 | Adjacent-section breast LOSO |
| Visium FFPE Human Breast Cancer | 2,516 | FFPE breast LOSO |
| Parent Visium Human Breast Cancer | 3,988 | Breast LOSO |
| Parent Visium Human Colorectal Cancer | 3,130 | External validation |
| Parent Visium Human Ovarian Cancer | 3,484 | External validation |
| Targeted Visium Human Glioblastoma | 2,548 | External validation |
| `visium_hne` mouse brain | 2,688 | Tutorial benchmark |

The frozen-model comparison uses the 14,287 breast spots. Sections 1 and 2 come from the
same specimen, so effective patient-level diversity is smaller than the slide count.

## Methods

### Labels and patches

- Slide-local molecular domains are mapped to harmonized TME classes.
- The primary regression targets are immune, ECM, proliferation, and related module
  scores; single genes remain optional secondary targets.
- H&E patches use 3× the Visium spot diameter, are resized to 224×224, and use per-slide
  Macenko normalization.
- CNN training applies flips, 90-degree rotations, and mild color jitter.

### Models

- Radiomics Random Forest on handcrafted patch features.
- ImageNet-pretrained ResNet18 with classification and regression heads.
- Frozen Kaiko ViT-S/16 embeddings (384 dimensions).
- Frozen Phikon ViT-B embeddings (768 dimensions).

Only downstream linear probes are trained for the foundation-model experiments. The
encoders receive no gradient updates.

### Validation

- Every outer fold holds out one complete slide.
- Foundation-model preprocessing, logistic regularization, and class weighting are
  selected using inner LOSO folds among the three training slides.
- Macro-F1 uses a fixed task label set.
- Every outer fold includes a majority-class baseline fitted on its training slides.
- The confident three-class task reports retained coverage explicitly.
- Slide-wise z-scoring is label-free but transductive: it uses the complete inference
  slide's feature distribution.

## Why the first version underperformed

| Problem | Consequence | Current correction |
|---|---|---|
| Slide-local cluster IDs pooled as global labels | Contradictory supervision | Harmonized TME taxonomy |
| Roughly 15-pixel native patches enlarged to 224 | Blurred, context-poor morphology | 3× spot context |
| Single-gene regression as the primary target | Noisy morphology–expression relationship | Module-score targets |
| Equal classification and regression loss | Noisy class task dominated gradients | Classification weight 0.25 |
| One global stain reference | FFPE/fresh-frozen shift | Per-slide Macenko normalization |
| No augmentation | Easy slide- and stain-specific fitting | Geometric and color augmentation |

## Executed foundation-model results

| Task | Encoder | Mean macro-F1 | Worst fold | Balanced accuracy | Majority F1 | Coverage |
|---|---|---:|---:|---:|---:|---:|
| Four labels | Kaiko | 0.294 | 0.249 | 0.346 | 0.187 | 100% |
| Four labels | Phikon | **0.320** | 0.247 | **0.375** | 0.187 | 100% |
| Confident three labels | Kaiko | 0.326 | 0.203 | 0.411 | 0.211 | 39.0% |
| Confident three labels | Phikon | **0.366** | 0.227 | **0.433** | 0.211 | 39.0% |

### Phikon fold stability

| Held-out slide | Four-label F1 | Confident three-label F1 | Three-label coverage |
|---|---:|---:|---:|
| Breast section 1 | 0.450 | 0.518 | 25.8% |
| Breast section 2 | 0.307 | 0.394 | 23.1% |
| FFPE breast | 0.276 | 0.324 | 53.6% |
| Parent breast | 0.247 | 0.227 | 53.4% |

Phikon is the stronger tested encoder, but the fold range is wide and the weakest fold is
close to the baseline regime. The result demonstrates improvement, not robustness.

Slide z-score plus L2 normalization was chosen in all all-class encoder/fold comparisons.
For Phikon, PCA-space slide silhouette decreases from 0.357 to -0.001. The accompanying
UMAP continues to separate slides, indicating that nonlinear domain structure remains.

The earlier raw Kaiko probe reached macro-F1 0.141. Its regression arm achieved mean
Pearson r 0.119 but negative mean R² in every fold, and the prespecified FFPE ECM map
failed to transfer. That experiment is superseded by the nested comparison notebook and
is retained here only as diagnostic context.

## Earlier CNN and RF result

The pre-remediation full run produced mean balanced accuracy of 0.060 for ResNet18 and
0.063 for radiomics RF. Mean gene Pearson r was -0.013 and 0.068 respectively. Those
classification numbers used obsolete slide-local cluster IDs and must not be compared
directly with the harmonized foundation-model task.

## What should improve F1 next

1. Replace heuristic cluster annotations with pathologist-reviewed labels or
   reproducible marker rules.
2. Add independent patients, institutions, scanners, and preparation protocols.
3. Combine spot-centered and wider-context tiles.
4. Aggregate neighboring spot embeddings with a spatial graph.
5. Report an abstention/F1–coverage curve for ambiguous spots instead of silently
   removing `other`.
6. Consider parameter-efficient fine-tuning only after the first five items reduce the
   risk of learning slide identity.

## Reproducibility

- Configuration: `configs/default.yaml`
- Seed: 0
- Environment: `spatial-tx` plus `requirements-pharma.txt`
- Pipeline: `python projects/spatial-pharma-dl/scripts/run_pipeline.py`
- Notebook builder: `python projects/spatial-pharma-dl/scripts/build_foundation_notebook.py`
- Executed notebook: `notebooks/07_foundation_model_comparison.ipynb`
- Result tables: `outputs/pharma/foundation_v2/`

## Evidence grade and limitations

**Share with caveats.** The workflow uses reproducible slide-level validation, but the
cohort is small, two slides are adjacent sections, labels are heuristic, Visium spots are
multicellular, and normalization is whole-slide transductive. Both foundation-model
checkpoints have non-commercial research terms. Public Visium data also lack treatment
arms, so pharmacodynamic claims require separate proprietary cohorts.
