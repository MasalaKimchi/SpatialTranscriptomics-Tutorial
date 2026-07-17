---
title: Technology Stack
mapped_at: 2026-07-17
last_mapped_commit: 1c2d0739bbb2b724a4eaef1cdbb16d865bff7580
---

# Technology Stack

## Runtime and Packaging

- The repository is Python-only and declares Python `>=3.11` in `pyproject.toml`; `README.md` recommends Python 3.10 or 3.11 for scientific-package compatibility.
- The reproducible Conda environment is named `spatial-tx` in `environment.yml` and pins Python 3.11 while sourcing packages from conda-forge and bioconda.
- The pip installation path is split between the tutorial dependencies in `requirements.txt` and optional deep-learning dependencies in `projects/spatial-pharma-dl/requirements-pharma.txt`.
- `pyproject.toml` defines package metadata for `spatial-transcriptomics-tutorial` version 0.1.0 and an optional `pharma` dependency group.
- Only `utils*` is discovered as an installable package by `pyproject.toml`; `projects/spatial-pharma-dl/src/` is reached through runtime `sys.path` setup rather than normal package installation.
- Shared filesystem and import bootstrapping lives in `utils/st_helpers.py` and `projects/spatial-pharma-dl/src/bootstrap.py`.

## Notebook and Tutorial Layer

- Thirteen root-level Jupyter notebooks, `00_overview_spatial_transcriptomics.ipynb` through `12_summary_research_extensions.ipynb`, form the primary tutorial interface.
- Jupyter and IPython are runtime dependencies through `requirements.txt`; notebooks target the `Python (spatial-tx)` kernel described in `README.md`.
- The core spatial object model is AnnData, supplied by `anndata>=0.9`; HDF5-backed `.h5ad` files are the principal intermediate format.
- Scanpy `>=1.9` provides QC, normalization, PCA, neighborhood graphs, UMAP, marker analysis, and Leiden orchestration, including the compatibility wrapper in `utils/st_helpers.py`.
- Squidpy `>=1.3` provides public Visium datasets, spatial plotting, and spatial statistics; dataset access is centralized in `utils/st_helpers.py`.
- Leiden clustering depends on `leidenalg>=0.9` and `igraph>=0.10`, declared in both `requirements.txt` and `environment.yml`.

## Scientific and Visualization Stack

- NumPy, pandas, and SciPy are the general numerical and tabular foundation across `utils/`, `scripts/`, and `projects/spatial-pharma-dl/src/`.
- scikit-learn supplies regressors, classifiers, preprocessing, metrics, and linear probes in `projects/spatial-pharma-dl/src/eval.py`, `projects/spatial-pharma-dl/src/foundation.py`, and `projects/spatial-pharma-dl/src/foundation_eval.py`.
- Matplotlib and seaborn produce notebook plots and committed gallery assets; headless generation is configured with the `Agg` backend in `scripts/generate_gallery_figures.py` and `projects/spatial-pharma-dl/scripts/run_pipeline.py`.
- scikit-image implements patch resizing, stain decomposition, GLCM texture, entropy, and edge features in `projects/spatial-pharma-dl/src/patches.py`.
- OpenCV is installed through `opencv-python>=4.7` for image-processing exercises, although the reusable pharma modules primarily use scikit-image.
- `requests`, `pooch`, and `tqdm` are declared utility dependencies; direct dataset transfer is delegated to Squidpy rather than implemented as repository-specific HTTP code.

## Deep Learning and Foundation Models

- PyTorch `>=2.0` is the training and inference runtime for the optional pharma project under `projects/spatial-pharma-dl/`.
- torchvision `>=0.15` supplies ImageNet-pretrained ResNet, EfficientNet, ConvNeXt, and ViT backbones in `projects/spatial-pharma-dl/src/models.py`.
- The main trainable architecture is a shared image encoder with classification and regression heads implemented by `MultiTaskImageModel` in `projects/spatial-pharma-dl/src/models.py`.
- `projects/spatial-pharma-dl/src/train.py` uses PyTorch datasets, data loaders, Adam-style optimization, early stopping, and leave-one-slide-out evaluation orchestration.
- timm `>=1.0` loads the Kaiko pathology ViT through an `hf_hub:` model name in `projects/spatial-pharma-dl/src/foundation.py`.
- transformers `>=4.40` loads Phikon using `AutoModel.from_pretrained` in `projects/spatial-pharma-dl/src/foundation.py`.
- Foundation encoders are frozen; scikit-learn logistic and ridge probes are fit on cached embeddings rather than fine-tuning the downloaded weights.
- Device selection for CPU, CUDA, or Apple MPS is isolated in `projects/spatial-pharma-dl/src/device.py`.

## Configuration, Storage, and Automation

- YAML configuration is parsed with PyYAML from `projects/spatial-pharma-dl/configs/default.yaml`; it controls cohorts, QC, labels, patches, training, evaluation, and foundation models.
- pandas and PyArrow write label and patch-index Parquet files from `projects/spatial-pharma-dl/src/labels.py` and `projects/spatial-pharma-dl/src/patches.py`.
- NumPy compressed archives store patch tensors and frozen embeddings in `projects/spatial-pharma-dl/src/patches.py` and `projects/spatial-pharma-dl/src/foundation.py`.
- JSON is used for notebook source manipulation and experiment summaries in `scripts/patch_notebooks.py`, `projects/spatial-pharma-dl/scripts/build_notebooks.py`, and `projects/spatial-pharma-dl/scripts/run_pipeline.py`.
- `nbformat` generates the foundation-model comparison notebook in `projects/spatial-pharma-dl/scripts/build_foundation_notebook.py`.
- The end-to-end noninteractive entry point is `projects/spatial-pharma-dl/scripts/run_pipeline.py`; environment flags select train-only, quick, forced-patch, and foundation-model modes.
- Unit and regression tests use pytest conventions under `projects/spatial-pharma-dl/tests/`, with NumPy and PyTorch fixtures and no separate application server or database.
