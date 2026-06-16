# Spatial Pharma DL — Histology-to-TME Molecular Profiling

Pharma-facing subproject that predicts tumor microenvironment (TME) molecular state from
H&E patches using a simple ResNet18 CNN, with leave-one-slide-out validation on public
10x Visium oncology data.

## Quickstart

From the repository root:

```bash
conda activate spatial-tx
pip install -r requirements.txt -r projects/spatial-pharma-dl/requirements-pharma.txt

# Run notebooks in order (01 → 06)
jupyter lab projects/spatial-pharma-dl/notebooks/
```

Or use the Python modules directly:

```bash
python -c "
from projects.spatial_pharma_dl.src.data import load_config, preprocess_cohort
cfg = load_config()
# preprocess_cohort downloads + QC slides (requires network on first run)
"
```

## Project structure

| Path | Purpose |
|------|---------|
| [`PROJECT.md`](PROJECT.md) | Pharma hypothesis and success criteria |
| [`ANALYSIS_STRATEGY.md`](ANALYSIS_STRATEGY.md) | Full phase-by-phase protocol |
| [`configs/default.yaml`](configs/default.yaml) | Cohorts, genes, hyperparameters |
| [`src/`](src/) | Data, patches, labels, model, train, eval |
| [`notebooks/`](notebooks/) | Six-phase analysis workflow |
| [`PROJECT_REPORT.md`](PROJECT_REPORT.md) | Results template (fill after running) |

## Datasets

**Primary (breast TME, LOSO training):** 4 human breast Visium slides via `squidpy.datasets.visium`.

**External validation:** CRC, ovarian, glioblastoma slides (held out from training).

**Benchmark:** Tutorial mouse brain (`visium_hne`) for RF vs CNN comparison.

## Outputs

All generated artifacts go to `outputs/pharma/` at the repo root (git-ignored).
