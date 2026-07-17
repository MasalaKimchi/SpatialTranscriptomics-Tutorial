"""Offline orchestration evidence for typed artifact reuse decisions."""

from __future__ import annotations

import pytest

from src import data
from utils import st_helpers as st

pytestmark = pytest.mark.offline


def test_available_processed_status_is_pure_for_missing_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "project_root", lambda: tmp_path)
    assert data.available_processed_slide_ids(["slide_a"], cfg=data.load_config()) == set()
    assert not (tmp_path / "data").exists()
