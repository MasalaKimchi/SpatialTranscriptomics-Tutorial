"""Spatial Pharma DL — histology-to-TME molecular profiling."""

__version__ = "0.1.0"

from .benchmark import run_and_save_benchmark, run_loso_benchmark
from .data import cohort_slide_ids, load_config, pharma_outputs_dir, preprocess_cohort
from .train import load_slide_patches, train_loso

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
