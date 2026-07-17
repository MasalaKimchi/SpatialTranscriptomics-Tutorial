---
title: External Integrations
mapped_at: 2026-07-17
last_mapped_commit: 1c2d0739bbb2b724a4eaef1cdbb16d865bff7580
---

# External Integrations

## Public Spatial Data

- Squidpy is the primary external data gateway; `utils/st_helpers.py` calls `squidpy.datasets.visium_hne_adata()` for the built-in mouse-brain H&E tutorial sample.
- The same helper calls `squidpy.datasets.visium(sample_id)` for arbitrary public 10x Genomics Visium samples used by the pharma cohort.
- Cohort sample identifiers are configured in `projects/spatial-pharma-dl/configs/default.yaml`, including breast, colorectal, ovarian, glioblastoma, and benchmark datasets.
- `projects/spatial-pharma-dl/src/data.py` invokes the helper for each configured slide, then normalizes and caches it locally.
- First access requires outbound network connectivity; Squidpy owns its download cache, while processed project copies are written beneath `data/processed/`.
- The data integration is anonymous and public: no API key, OAuth flow, or private credential is referenced in repository code.
- Raw 10x Genomics download is explained as an alternative in `02_fetch_public_visium_data.ipynb`, but reusable Python code uses Squidpy rather than a bespoke 10x API client.

## Hugging Face Model Hub

- The optional pathology foundation-model integration is implemented in `projects/spatial-pharma-dl/src/foundation.py`.
- Kaiko uses repository ID `1aurent/vit_small_patch16_224.kaiko_ai_towards_large_pathology_fms` and is loaded by timm with the `hf_hub:` URI convention.
- Phikon uses repository ID `owkin/phikon` and is loaded with `transformers.AutoModel.from_pretrained`.
- Downloads occur only when foundation functionality is enabled or called; the default `foundation.enabled` setting is `false` in `projects/spatial-pharma-dl/configs/default.yaml`.
- `huggingface-hub`, `timm`, and `transformers` are declared in `projects/spatial-pharma-dl/requirements-pharma.txt` and the `pharma` optional group in `pyproject.toml`.
- Repository code does not supply a Hugging Face token. Ungated/public access is expected for the selected checkpoints, subject to upstream availability and cache state.
- Downloaded encoder weights use the normal timm/Transformers cache, while derived per-slide embeddings are cached under `data/processed/pharma/foundation_embeddings/<model>/`.
- Embedding cache identity includes model name, slide ID, and patch version through `_embedding_cache_path` in `projects/spatial-pharma-dl/src/foundation.py`.
- `projects/spatial-pharma-dl/README.md` and `projects/spatial-pharma-dl/PROJECT_REPORT.md` flag the tested model terms as research/non-commercial, creating a legal-governance boundary for pharma use.

## ImageNet Weights and Torchvision

- Trainable CNN/ViT baselines request torchvision ImageNet weights through typed weight enums in `projects/spatial-pharma-dl/src/models.py`.
- Instantiating a model with `pretrained: true` can trigger a network download into PyTorch's standard cache on first use.
- Supported external weight families are ResNet18, ResNet50, EfficientNet-B0, ConvNeXt-Tiny, and ViT-B/16.
- Offline execution is possible after caches exist or by setting `training.pretrained: false` in `projects/spatial-pharma-dl/configs/default.yaml`.

## Local Persistence Contracts

- Processed AnnData is persisted as `.h5ad` through `utils/st_helpers.py` and `projects/spatial-pharma-dl/src/data.py`; this is the boundary between notebook stages and pipeline phases.
- Label tables and the combined patch index are persisted as Parquet by `projects/spatial-pharma-dl/src/labels.py` and `projects/spatial-pharma-dl/src/patches.py`, requiring a PyArrow-compatible engine.
- H&E patch arrays are stored as compressed NumPy `.npz` files in `projects/spatial-pharma-dl/src/patches.py`; metadata currently uses an object payload read with `allow_pickle=True`.
- Frozen embeddings use compressed `.npz` files with non-object arrays and `allow_pickle=False` in `projects/spatial-pharma-dl/src/foundation.py`.
- Metrics and run summaries are written as CSV and JSON beneath `outputs/pharma/` by `projects/spatial-pharma-dl/src/eval.py`, `projects/spatial-pharma-dl/src/benchmark.py`, and `projects/spatial-pharma-dl/scripts/run_pipeline.py`.
- Committed preview PNGs under `outputs/figures/` are generated locally by `scripts/generate_gallery_figures.py`; there is no remote artifact store or experiment-tracking service.

## Runtime Environment Interfaces

- `projects/spatial-pharma-dl/scripts/run_pipeline.py` consumes `PHARMA_TRAIN_ONLY`, `PHARMA_QUICK`, `PHARMA_FOUNDATION`, and `PHARMA_FORCE_PATCHES` as its operational interface.
- `KMP_DUPLICATE_LIB_OK` is set in `projects/spatial-pharma-dl/scripts/run_pipeline.py` to tolerate duplicate OpenMP runtimes in some local scientific environments.
- `projects/spatial-pharma-dl/src/bootstrap.py` and `utils/st_helpers.py` manipulate `sys.path` so the nested project can import the root `utils` package and its own generic `src` package.
- Jupyter is the human-facing runtime integration; notebook generation and patching are implemented by `projects/spatial-pharma-dl/scripts/build_notebooks.py`, `projects/spatial-pharma-dl/scripts/build_foundation_notebook.py`, and `scripts/patch_notebooks.py`.

## Absent Service Integrations

- There is no database, message queue, web framework, REST/GraphQL server, cloud storage SDK, telemetry agent, or secrets-management client in the tracked Python code.
- There is no CI workflow, container definition, deployment manifest, or hosted notebook configuration in the currently tracked repository.
- There are no application credentials, `.env` readers, or private endpoint URLs in the mapped source tree.
- External availability risks are therefore concentrated in first-run dataset downloads, pretrained weight downloads, and third-party package installation.
