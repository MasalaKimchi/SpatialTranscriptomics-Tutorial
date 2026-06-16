# Spatial Pharma DL — Project Brief

## Hypothesis

H&E morphology encodes enough tumor microenvironment (TME) information to predict spatial
molecular state (transcriptomic domains and key marker genes) with accuracy that improves
over handcrafted radiomics — using only public Visium data and slide-level validation.

## Pharma question

> Can cheap H&E serve as a surrogate for expensive spatial transcriptomics when localizing
> target expression, immune infiltration, and stromal remodeling in oncology tissue?

## Stakeholder mapping

| Drug-discovery decision | Deliverable in this project |
|-------------------------|----------------------------|
| Where is my target expressed? | Spatial gene prediction (ESR1, ERBB2, immune markers) |
| Can morphology stratify patients? | TME domain classification from patches |
| Is H&E a viable ST surrogate? | RF vs CNN benchmark with LOSO validation |
| Can we trust predictions biologically? | Grad-CAM + spatial error maps |

## Datasets

| Tier | Slides | Role |
|------|--------|------|
| 1 — Breast TME | 4 Visium breast slides | Primary LOSO training |
| 2 — External | CRC, ovarian, GBM | Zero-shot generalization |
| 3 — Benchmark | Mouse brain (tutorial) | Method parity with notebook 10 |

## Methods summary

1. Tutorial-equivalent QC + Leiden clustering per slide
2. Label engineering: domains, marker genes, module scores
3. Per-spot H&E patches with Macenko stain normalization
4. ResNet18 multi-task CNN (classification + regression)
5. Leave-one-slide-out CV on breast cohort
6. Grad-CAM interpretability

## Success criteria

| Criterion | Target |
|-----------|--------|
| CNN vs RF (breast LOSO) | Higher balanced accuracy AND mean gene Pearson r |
| Validation design | Slide-level holdout only (no random spot splits) |
| External evaluation | Metrics on ≥2 non-breast tumors |
| Interpretability | Grad-CAM highlights tissue, not artifacts |
| Reproducibility | Fixed seeds, YAML config, cached h5ad per slide |

## Known limitations

- No treatment arms in public Visium — pharmacodynamic comparison requires proprietary data
- Visium spots are multi-cell (~55 µm) — screening tool, not single-cell resolution
- Breast sections 1+2 are same patient — external tumors provide cross-cohort story
- Clinical ER/HER2 status is sample-level metadata, not per-spot labels

## References

- Tutorial baseline: notebook 10 (RF on 15 radiomics features, mouse brain)
- 10x Visium public datasets via Squidpy
- Macenko stain normalization for cross-slide comparability
