# Spatial Pharma DL — Project Report (Template)

> Fill this document after running notebooks 01–06.

## Executive summary

- **Hypothesis tested:**
- **Primary result (CNN vs RF on breast LOSO):**
- **External validation result:**
- **Recommendation for pharma stakeholders:**

## Cohort summary

| Slide ID | Spots | Clusters | Median genes | Notes |
|----------|-------|----------|--------------|-------|
| | | | | |

## Model performance

### Leave-one-slide-out (breast cohort)

| Fold | Held-out slide | RF balanced acc | CNN balanced acc | RF mean Pearson r | CNN mean Pearson r |
|------|----------------|-----------------|------------------|-------------------|---------------------|
| | | | | | |

### Per-gene regression (held-out slides)

| Gene | Pearson r | R² | MAE |
|------|-----------|-----|-----|
| | | | |

### External validation (zero-shot)

| Slide | Tumor type | Balanced acc | Mean Pearson r |
|-------|------------|--------------|----------------|
| | | | |

## Interpretability

- Grad-CAM observations:
- Failure modes:
- Biological plausibility assessment:

## Limitations and next steps

- Limitations:
- Proposed extensions (proprietary cohorts, treatment arms, HD platforms):

## Reproducibility

- Config: `configs/default.yaml`
- Seed: 0
- Environment: `spatial-tx` + `requirements-pharma.txt`
- Date run:
