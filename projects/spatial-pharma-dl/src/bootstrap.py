"""One-time import path setup for Spatial Pharma DL."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# PyTorch + numpy/scipy on macOS often link duplicate OpenMP runtimes.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parents[3]
PHARMA_ROOT = Path(__file__).resolve().parents[1]


def ensure_import_paths() -> tuple[Path, Path]:
    """Add repo root and pharma project dir to ``sys.path`` if needed."""
    for path in (REPO_ROOT, PHARMA_ROOT):
        entry = str(path)
        if entry not in sys.path:
            sys.path.insert(0, entry)
    return REPO_ROOT, PHARMA_ROOT


ensure_import_paths()
