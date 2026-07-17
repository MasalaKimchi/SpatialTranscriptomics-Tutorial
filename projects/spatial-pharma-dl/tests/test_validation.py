"""Offline evidence for deterministic aggregate experiment validation."""

from __future__ import annotations

import copy
import importlib
import math
import sys
from pathlib import Path

import pytest
import yaml

from src.validation import ConfigValidationError, ResolvedConfig, resolve_config

pytestmark = pytest.mark.offline

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def _valid_config() -> dict[str, object]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_malformed_config_reports_ordered_aggregate_issues() -> None:
    cfg = _valid_config()
    cfg["unknown_root"] = "remove-me"
    cfg["seed"] = True
    cfg["cohorts"]["oncology"] = ["slide_a", "slide_a"]
    cfg["cohorts"]["external"] = ["slide_a"]
    cfg["preprocessing"]["n_pcs"] = 10
    cfg["preprocessing"]["n_pcs_neighbors"] = 20
    cfg["labels"]["tme_classes"] = ["tumor_epithelial"]
    cfg["training"]["cls_weight"] = 0
    cfg["training"]["reg_weight"] = 0

    with pytest.raises(ConfigValidationError) as caught:
        resolve_config(cfg)

    paths = [issue.path for issue in caught.value.issues]
    expected = [
        "config.unknown_root",
        "seed",
        "cohorts.oncology[1]",
        "cohorts.external[0]",
        "preprocessing.n_pcs_neighbors",
        "labels.tme_classes",
        "training.cls_weight",
    ]
    assert paths == expected
    message = str(caught.value)
    for issue in caught.value.issues:
        assert issue.path in message
        assert repr(issue.received) in message
        assert issue.expected in message
        assert issue.guidance in message


def test_numeric_bool_unknown_and_dynamic_mapping_rules() -> None:
    cfg = _valid_config()
    cfg["training"]["batch_size"] = False
    cfg["training"]["surprise"] = 3
    cfg["marker_genes"]["custom_panel"] = ["GENE_A"]
    cfg["gene_modules"]["custom_module"] = ["GENE_A", "GENE_B"]

    with pytest.raises(ConfigValidationError) as caught:
        resolve_config(cfg)

    paths = [issue.path for issue in caught.value.issues]
    assert "training.batch_size" in paths
    assert "training.surprise" in paths
    assert "marker_genes.custom_panel" not in paths
    assert "gene_modules.custom_module" not in paths


@pytest.mark.parametrize(
    ("value", "expected_path"),
    [
        (math.nan, "training.lr"),
        (math.inf, "training.lr"),
        ({"not", "json"}, "config.training.lr"),
        (object(), "config.training.lr"),
    ],
)
def test_non_finite_and_unsupported_values_fail_closed(
    value: object, expected_path: str
) -> None:
    cfg = _valid_config()
    cfg["training"]["lr"] = value
    with pytest.raises(ConfigValidationError) as caught:
        resolve_config(cfg)
    assert expected_path in {issue.path for issue in caught.value.issues}


def test_canonical_json_sorts_mappings_but_preserves_cohort_lists() -> None:
    first = _valid_config()
    second = {key: first[key] for key in reversed(first)}
    second["training"] = {
        key: first["training"][key] for key in reversed(first["training"])
    }
    assert resolve_config(first).canonical_json == resolve_config(second).canonical_json

    reordered = copy.deepcopy(first)
    reordered["cohorts"]["oncology"] = list(
        reversed(reordered["cohorts"]["oncology"])
    )
    assert resolve_config(first).canonical_json != resolve_config(reordered).canonical_json


def test_resolved_config_returns_fresh_mutable_plain_trees() -> None:
    resolved = resolve_config(_valid_config())
    assert isinstance(resolved, ResolvedConfig)
    first = resolved.to_dict()
    second = resolved.to_dict()
    assert first == second
    first["cohorts"]["oncology"].append("mutated")
    first["training"]["epochs"] = 999
    assert first != second
    assert "mutated" not in second["cohorts"]["oncology"]


def test_validation_import_stays_lightweight(monkeypatch: pytest.MonkeyPatch) -> None:
    heavy = {"torch", "scanpy", "squidpy", "torchvision", "timm", "transformers"}
    before = set(sys.modules)
    monkeypatch.delitem(sys.modules, "src.validation", raising=False)
    importlib.import_module("src.validation")
    loaded = {name.split(".", 1)[0] for name in set(sys.modules) - before}
    assert heavy.isdisjoint(loaded)
