---
last_mapped_commit: 1c2d0739bbb2b724a4eaef1cdbb16d865bff7580
mapping_focus: quality
---

# Coding Conventions

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
