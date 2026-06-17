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

Or run the full pipeline from the command line:

```bash
# Full run (downloads ~8 Visium slides on first use; CNN training 30+ min on GPU)
python projects/spatial-pharma-dl/scripts/run_pipeline.py

# Resume training only (after phases 1–3 are cached)
PHARMA_TRAIN_ONLY=1 python projects/spatial-pharma-dl/scripts/run_pipeline.py

# Fast smoke test (500 spots/fold, 2 epochs)
PHARMA_QUICK=1 PHARMA_TRAIN_ONLY=1 python projects/spatial-pharma-dl/scripts/run_pipeline.py
```

`KMP_DUPLICATE_LIB_OK=TRUE` is set automatically via `src/bootstrap.py` on macOS.

Or inspect modules directly:

```bash
cd projects/spatial-pharma-dl
python -c "
from src.data import load_config, cohort_slide_ids
cfg = load_config()
print(cohort_slide_ids(cfg))
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
