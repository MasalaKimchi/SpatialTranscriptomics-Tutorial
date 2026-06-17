# Spatial Pharma DL — Project Report (Template)

> Fill this document after running notebooks 01–06.

## Executive summary

- **Hypothesis tested:** H&E patches can predict TME domain and marker-gene expression with slide-level validation.
- **Primary result (CNN vs RF on breast LOSO, smoke run):** RF achieves higher mean gene Pearson r (~0.06–0.09 vs ~0.00–0.03 CNN) on 500-spot subsample with 2 epochs; both models show low domain balanced accuracy (~3–8%) — expected with minimal training. Full 15-epoch run recommended for production metrics.
- **External validation result:** Patches generated for CRC, ovarian, GBM slides; zero-shot inference supported via notebook 05.
- **Recommendation for pharma stakeholders:** Pipeline is operational; scale to full epochs and full spot counts for decision-grade benchmarks.

## Cohort summary

| Slide ID | Spots | Clusters | Median genes | Notes |
|----------|-------|----------|--------------|-------|
| V1_Breast_Cancer_Block_A_Section_1 | 3798 | 14 | 6027 | Primary breast IDC |
| V1_Breast_Cancer_Block_A_Section_2 | 3985 | 17 | 5584 | Adjacent section |
| Visium_FFPE_Human_Breast_Cancer | 2516 | 15 | 4841 | FFPE clinical-like |
| Parent_Visium_Human_BreastCancer | 3988 | 14 | 3883 | Whole-transcriptome |
| Parent_Visium_Human_ColorectalCancer | 3130 | 14 | 3543 | External validation |
| Parent_Visium_Human_OvarianCancer | 3484 | 10 | 3469 | External validation |
| Targeted_Visium_Human_Glioblastoma_Pan_Cancer | 2548 | 11 | 357 | External validation |
| visium_hne | 2688 | 17 | 5805 | Tutorial benchmark |

## Model performance

See `outputs/pharma/benchmark_report.csv` after running `scripts/run_pipeline.py`.

### Leave-one-slide-out (breast cohort) — smoke run (PHARMA_QUICK=1)

| Model | Mean balanced acc | Mean Pearson r |
|-------|-------------------|----------------|
| CNN (ResNet18) | ~0.05 | ~0.00 |
| RF (radiomics) | ~0.06 | ~0.07 |

## Reproducibility

- Config: `configs/default.yaml`
- Seed: 0
- Environment: `spatial-tx` + `requirements-pharma.txt`
- Run command: `PHARMA_QUICK=1 PHARMA_TRAIN_ONLY=1 python projects/spatial-pharma-dl/scripts/run_pipeline.py`
