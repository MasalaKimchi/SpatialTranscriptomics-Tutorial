<!-- GSD:project-start source:PROJECT.md -->

## Project

**Spatial Transcriptomics Tutorial Reliability Upgrade**

This repository is a notebook-first spatial transcriptomics tutorial with an optional pharma-facing deep-learning extension for predicting tumor-microenvironment molecular state from H&E patches. This milestone strengthens the existing implementation through exactly 20 high-priority correctness, security, reproducibility, validation, and test-infrastructure improvements while preserving its educational workflow and public outputs.

**Core Value:** Reported spatial and machine-learning results must be scientifically trustworthy, reproducible, and produced from validated artifacts without hidden data leakage.

### Constraints

- **Behavioral compatibility**: Keep notebook order, documented CLI entry points, config keys, output names, and public Python exports stable unless a security-safe artifact migration is explicitly documented.
- **Scientific validity**: All learned preprocessing, model selection, imputation, and scaling must use training data only within each outer fold.
- **Offline tests**: Default automated tests must not download datasets or model weights.
- **Resource budget**: Fast CI must remain CPU-compatible; network and full-cohort tests are opt-in tiers.
- **Security**: Untrusted cache and checkpoint reads must not execute Python objects.
- **Traceability**: Every improvement maps to one requirement, targeted tests, and a reviewable GSD phase/commit.

<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->

## Technology Stack

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

<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->

## Conventions

## Repository Style

- Python is the implementation language for shared helpers, notebook maintenance, and the pharma extension; the primary teaching surface is Jupyter notebooks.
- Source files use module docstrings and, almost universally, `from __future__ import annotations`; representative modules are `utils/st_helpers.py` and `projects/spatial-pharma-dl/src/data.py`.
- Public functions and classes use `snake_case` and `PascalCase`; private implementation helpers use a leading underscore, such as `_mito_prefix` in `projects/spatial-pharma-dl/src/data.py` and `_normalize_od` in `projects/spatial-pharma-dl/src/patches.py`.
- Constants are uppercase at module scope, for example `SEED` in `utils/st_helpers.py`, `CONFIG_PATH` in `projects/spatial-pharma-dl/src/data.py`, and `FOUNDATION_MODELS` in `projects/spatial-pharma-dl/src/foundation.py`.
- Files are organized by responsibility: data loading, labels, patches, models, training, evaluation, and foundation-model evaluation each have separate modules under `projects/spatial-pharma-dl/src/`.
- Formatting broadly follows standard Python conventions: four-space indentation, blank lines between top-level definitions, double-quoted strings, and trailing commas in multiline calls.
- Ruff currently reports no violations across `utils/`, `scripts/`, `projects/spatial-pharma-dl/src/`, `projects/spatial-pharma-dl/scripts/`, and `projects/spatial-pharma-dl/tests/`, but no Ruff configuration or pinned rule set is declared in `pyproject.toml`.

## Imports and Packaging

- Standard-library, third-party, and local imports are grouped, although import bootstrapping requires explicit exceptions such as `# noqa: E402` in tests and scripts.
- Heavy optional libraries are often imported inside functions, as in `preprocess_slide()` in `projects/spatial-pharma-dl/src/data.py`; this keeps lightweight imports usable without Scanpy or model-download dependencies.
- `projects/spatial-pharma-dl/src/__init__.py` exposes a small lazy public API via `__getattr__`, avoiding eager imports but making export discovery less direct than ordinary imports.
- The pharma extension relies on runtime path mutation through `projects/spatial-pharma-dl/src/bootstrap.py` and `utils/st_helpers.py`; notebooks repeat related `sys.path` setup because `src` is not installed as a distinct package.
- The root package metadata in `pyproject.toml` installs only `utils*`, so the `projects/spatial-pharma-dl/src/` namespace is intentionally outside normal package discovery.

## Types, Data, and Configuration

- Type hints cover most public Python functions and use modern syntax such as `Path | None`, `dict[str, Any]`, and tuple return types.
- Scientific objects such as AnnData and several dataset-return values remain untyped or use `Any`, reflecting optional dependencies and dynamic schemas.
- Configuration is loaded from `projects/spatial-pharma-dl/configs/default.yaml` as nested `dict[str, Any]`; callers directly index required keys and use `.get()` only for optional values.
- There is no schema validation or typed configuration object, so misspelled keys and invalid types fail at the eventual access or library call rather than at configuration load time.
- Tabular conventions are explicit in code: slide and spot identity use `slide_id` and `spot_id`, class targets use `tme_class`/`tme_class_id`, gene targets start with `gene_`, and module targets start with `module_`.
- File paths use `pathlib.Path` and are derived relative to the repository; helper functions create data and output directories on demand in `utils/st_helpers.py` and `projects/spatial-pharma-dl/src/data.py`.
- Reproducibility is centralized around `st.set_seeds()` and `st.SEED`, with stochastic Scanpy and scikit-learn calls generally receiving explicit seeds.

## Documentation and Readability

- Public helpers generally have concise docstrings describing purpose, parameters, return shapes, or operational assumptions.
- `utils/st_helpers.py` uses section comments to make a long shared utility module navigable; smaller pharma modules depend more on descriptive names and module boundaries.
- User-facing failures usually include remediation, for example missing cache errors in `utils/st_helpers.py`, `projects/spatial-pharma-dl/src/data.py`, and `projects/spatial-pharma-dl/src/patches.py` tell the user which earlier workflow step to run.
- Notebook-generation scripts store substantial code as multiline string literals in `scripts/patch_notebooks.py`, `projects/spatial-pharma-dl/scripts/build_notebooks.py`, and `projects/spatial-pharma-dl/scripts/build_foundation_notebook.py`; this is difficult to lint, navigate, and refactor compared with reusable imported functions.
- Generated notebooks intentionally repeat environment setup and narrative scaffolding, favoring self-contained teaching artifacts over strict deduplication.

## Error Handling and Validation

- Expected invalid arguments use specific built-ins: `ValueError` for unsupported modes or backbones, `TypeError` for incompatible model objects, `KeyError` for missing AnnData content, and `FileNotFoundError` for absent caches.
- Compatibility fallback is narrowly scoped in `utils/st_helpers.py`, where `run_leiden()` catches only `TypeError` before retrying the older Scanpy signature.
- Numerical fallback is also narrow in `projects/spatial-pharma-dl/src/patches.py`, where Macenko normalization catches `np.linalg.LinAlgError` and returns the original patch.
- Batch workflows skip only known missing-input conditions, such as absent processed slides in `projects/spatial-pharma-dl/src/data.py`, `projects/spatial-pharma-dl/src/labels.py`, and `projects/spatial-pharma-dl/src/patches.py`.
- Several low-data conditions return empty structures or fallback values rather than raising: tiny patches yield empty features, missing marker panels are skipped, and unavailable cohort members are omitted.
- Assertions are used mainly as notebook execution guards, while reusable modules prefer explicit exceptions.
- Validation is uneven at data boundaries: model and task modes receive explicit checks, but shapes, dtypes, required DataFrame columns, uniqueness, YAML schema, and alignment losses are often assumed.
- Logging uses `print()` in scripts and cohort loops rather than the standard `logging` package, appropriate for tutorials but limiting structured diagnostics in automated runs.

<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->

## Architecture

## System Shape

## Architectural Layers

## Primary Data Flow

## Entry Points

- Human-guided tutorial: open `00_overview_spatial_transcriptomics.ipynb` and proceed numerically through `12_summary_research_extensions.ipynb`.
- Pharma end-to-end CLI: run `projects/spatial-pharma-dl/scripts/run_pipeline.py` from the repository root.
- Pharma interactive workflow: run notebooks in `projects/spatial-pharma-dl/notebooks/` in numeric order.
- Gallery regeneration: run `scripts/generate_gallery_figures.py` after prerequisite root caches exist.
- Root notebook maintenance: run `scripts/patch_notebooks.py` to apply idempotent content patches to selected tutorial notebooks.
- Pharma notebook generation: run `projects/spatial-pharma-dl/scripts/build_notebooks.py` or `projects/spatial-pharma-dl/scripts/build_foundation_notebook.py`.
- Import entry point: `projects/spatial-pharma-dl/src/__init__.py` exposes a lazy package-level API so optional heavy dependencies are not imported until needed.

## Boundaries and Dependency Direction

## Architectural Conventions

## Cross-Cutting Concerns

<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->

## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->

## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:

- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->

## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
