"""Spatial Pharma DL — histology-to-TME molecular profiling.

The package-level API is resolved lazily so importing :mod:`src` does not load
the complete spatial-analysis and training stack.  This keeps focused utilities
usable when optional dependencies such as Scanpy are not installed.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "cohort_slide_ids",
    "load_config",
    "load_slide_patches",
    "pharma_outputs_dir",
    "preprocess_cohort",
    "run_and_save_benchmark",
    "run_loso_benchmark",
    "train_loso",
]

_EXPORTS = {
    "cohort_slide_ids": (".data", "cohort_slide_ids"),
    "load_config": (".data", "load_config"),
    "load_slide_patches": (".train", "load_slide_patches"),
    "pharma_outputs_dir": (".data", "pharma_outputs_dir"),
    "preprocess_cohort": (".data", "preprocess_cohort"),
    "run_and_save_benchmark": (".benchmark", "run_and_save_benchmark"),
    "run_loso_benchmark": (".benchmark", "run_loso_benchmark"),
    "train_loso": (".train", "train_loso"),
}


def __getattr__(name: str) -> Any:
    """Resolve public exports on first access and cache the result."""
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
