"""Offline structural validation for every committed teaching notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat
import pytest

pytestmark = pytest.mark.offline

ROOT = Path(__file__).resolve().parents[3]
PHARMA_NOTEBOOKS = ROOT / "projects" / "spatial-pharma-dl" / "notebooks"
ALLOWED_CELL_TYPES = {"markdown", "code", "raw"}


def _numbered_notebooks(directory: Path) -> list[Path]:
    return sorted(directory.glob("[0-9][0-9]_*.ipynb"))


def _prefixes(paths: list[Path]) -> list[int]:
    return [int(path.stem[:2]) for path in paths]


def test_notebook_discovery_preserves_public_sequences() -> None:
    root_notebooks = _numbered_notebooks(ROOT)
    pharma_notebooks = _numbered_notebooks(PHARMA_NOTEBOOKS)

    assert len(root_notebooks) == 13
    assert _prefixes(root_notebooks) == list(range(13))
    assert len(pharma_notebooks) == 7
    assert _prefixes(pharma_notebooks) == list(range(1, 8))
    assert len({path.name for path in root_notebooks}) == len(root_notebooks)
    assert len({path.name for path in pharma_notebooks}) == len(pharma_notebooks)


@pytest.mark.parametrize(
    "path",
    _numbered_notebooks(ROOT) + _numbered_notebooks(PHARMA_NOTEBOOKS),
    ids=lambda path: str(path.relative_to(ROOT)),
)
def test_committed_notebook_is_valid_and_nonempty(path: Path) -> None:
    notebook = nbformat.read(path, as_version=4)

    nbformat.validate(notebook)
    assert notebook.cells
    assert {cell.cell_type for cell in notebook.cells} <= ALLOWED_CELL_TYPES
    assert all(
        isinstance(cell.source, str)
        for cell in notebook.cells
        if cell.cell_type == "code"
    )


def test_notebook_kernel_families_remain_distinct() -> None:
    for path in _numbered_notebooks(ROOT):
        kernel = nbformat.read(path, as_version=4).metadata.kernelspec
        assert kernel.name == "python3"
        assert "spatial-tx" in kernel.display_name.lower()

    for path in _numbered_notebooks(PHARMA_NOTEBOOKS):
        kernel = nbformat.read(path, as_version=4).metadata.kernelspec
        assert kernel.name == "spatial-tx"
        assert "spatial-tx" in kernel.display_name.lower()
