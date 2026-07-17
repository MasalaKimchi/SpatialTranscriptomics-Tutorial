"""Offline evidence for deterministic aggregate experiment validation."""

from __future__ import annotations

import copy
import importlib
import math
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest
import yaml

from src import data
from src.data import cohort_slide_ids, load_config
from src.validation import ConfigValidationError, ResolvedConfig, resolve_config

pytestmark = pytest.mark.offline

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
ORIGINAL_TOP_LEVEL_KEYS = {
    "seed",
    "experiment",
    "cohorts",
    "preprocessing",
    "labels",
    "marker_genes",
    "gene_modules",
    "patches",
    "training",
    "foundation",
    "evaluation",
}


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


def test_oversized_integer_stays_inside_aggregate_failure_boundary() -> None:
    cfg = _valid_config()
    cfg["training"]["lr"] = 10**10000
    cfg["training"]["batch_size"] = False

    with pytest.raises(ConfigValidationError) as caught:
        resolve_config(cfg)

    assert [issue.path for issue in caught.value.issues] == [
        "training.batch_size",
        "training.lr",
        "config.training.lr",
    ]
    assert "<oversized integer>" in str(caught.value)


def test_hostile_repr_is_never_executed_while_reporting_invalid_values() -> None:
    class Hostile:
        def __repr__(self) -> str:
            raise AssertionError("validation executed hostile repr")

    cfg = _valid_config()
    cfg["training"]["lr"] = Hostile()
    cfg[Hostile()] = "invalid key"

    with pytest.raises(ConfigValidationError) as caught:
        resolve_config(cfg)

    assert "<Hostile>" in str(caught.value)
    assert {issue.path for issue in caught.value.issues} >= {
        "config.<key>",
        "training.lr",
        "config.training.lr",
    }
    assert [issue.path for issue in caught.value.issues].count("config.<key>") == 1


def test_invalid_primitive_keys_have_deterministic_deduplicated_diagnostics() -> None:
    first = _valid_config()
    first[2] = "second"
    first[1] = "first"
    second = _valid_config()
    second[1] = "first"
    second[2] = "second"

    with pytest.raises(ConfigValidationError) as caught_first:
        resolve_config(first)
    with pytest.raises(ConfigValidationError) as caught_second:
        resolve_config(second)

    first_keys = [
        issue.received
        for issue in caught_first.value.issues
        if issue.path == "config.<key>"
    ]
    second_keys = [
        issue.received
        for issue in caught_second.value.issues
        if issue.path == "config.<key>"
    ]
    assert first_keys == second_keys == [1, 2]
    assert str(caught_first.value) == str(caught_second.value)


def test_invalid_key_sort_never_calls_user_repr_or_comparison() -> None:
    class HostileKey:
        def __init__(self, stable_hash: int):
            self.stable_hash = stable_hash

        def __hash__(self) -> int:
            return self.stable_hash

        def __repr__(self) -> str:
            raise AssertionError("invalid-key sorting executed repr")

        def __lt__(self, _other: object) -> bool:
            raise AssertionError("invalid-key sorting executed comparison")

    cfg = _valid_config()
    cfg[HostileKey(2)] = "second"
    cfg[HostileKey(1)] = "first"

    with pytest.raises(ConfigValidationError) as caught:
        resolve_config(cfg)

    assert [issue.path for issue in caught.value.issues].count("config.<key>") == 2
    assert str(caught.value).count("received <HostileKey>") == 2


def test_hostile_root_mapping_subclasses_are_rejected_without_execution() -> None:
    calls: dict[str, int] = {}

    def hostile(name: str) -> None:
        calls[name] = calls.get(name, 0) + 1
        raise AssertionError(f"executed hostile root operation: {name}")

    class HostileDict(dict[object, object]):
        def __len__(self) -> int:
            hostile("dict_len")

        def __iter__(self):
            hostile("dict_iter")

        def __getitem__(self, key: object) -> object:
            hostile("dict_getitem")

        def __contains__(self, key: object) -> bool:
            hostile("dict_contains")

        def __repr__(self) -> str:
            hostile("dict_repr")

    class HostileMapping(Mapping[object, object]):
        def __len__(self) -> int:
            hostile("mapping_len")

        def __iter__(self):
            hostile("mapping_iter")

        def __getitem__(self, key: object) -> object:
            hostile("mapping_getitem")

        def __repr__(self) -> str:
            hostile("mapping_repr")

    for value in (HostileDict(), HostileMapping()):
        calls.clear()
        with pytest.raises(ConfigValidationError) as caught:
            resolve_config(value)
        assert [issue.path for issue in caught.value.issues] == ["config"]
        assert "ATTACKER" not in str(caught.value)
        assert calls == {}


def test_hostile_allowed_type_subclasses_aggregate_without_execution() -> None:
    calls: dict[str, int] = {}

    def hostile(name: str):
        calls[name] = calls.get(name, 0) + 1
        raise AssertionError(f"ATTACKER operation executed: {name}")

    class HostileInt(int):
        def bit_length(self) -> int:
            hostile("int_bit_length")

        def __repr__(self) -> str:
            hostile("int_repr")

        def __hash__(self) -> int:
            hostile("int_hash")

        def __lt__(self, other: object) -> bool:
            hostile("int_compare")

    class HostileFloat(float):
        def __repr__(self) -> str:
            hostile("float_repr")

        def __hash__(self) -> int:
            hostile("float_hash")

        def __lt__(self, other: object) -> bool:
            hostile("float_compare")

        def __ge__(self, other: object) -> bool:
            hostile("float_compare")

    class HostileStr(str):
        def strip(self, chars: str | None = None) -> str:
            hostile("str_strip")

        def __repr__(self) -> str:
            hostile("str_repr")

        def __hash__(self) -> int:
            hostile("str_hash")

        def __lt__(self, other: object) -> bool:
            hostile("str_compare")

    class HostileList(list[object]):
        def __len__(self) -> int:
            hostile("list_len")

        def __iter__(self):
            hostile("list_iter")

        def __getitem__(self, key: object) -> object:
            hostile("list_getitem")

        def __repr__(self) -> str:
            hostile("list_repr")

    class HostileTuple(tuple[object, ...]):
        def __len__(self) -> int:
            hostile("tuple_len")

        def __iter__(self):
            hostile("tuple_iter")

        def __getitem__(self, key: object) -> object:
            hostile("tuple_getitem")

        def __repr__(self) -> str:
            hostile("tuple_repr")

    class HostileDict(dict[object, object]):
        def __len__(self) -> int:
            hostile("nested_dict_len")

        def __iter__(self):
            hostile("nested_dict_iter")

        def __getitem__(self, key: object) -> object:
            hostile("nested_dict_getitem")

        def __contains__(self, key: object) -> bool:
            hostile("nested_dict_contains")

        def __repr__(self) -> str:
            hostile("nested_dict_repr")

    class HostileMapping(Mapping[object, object]):
        def __len__(self) -> int:
            hostile("nested_mapping_len")

        def __iter__(self):
            hostile("nested_mapping_iter")

        def __getitem__(self, key: object) -> object:
            hostile("nested_mapping_getitem")

        def __repr__(self) -> str:
            hostile("nested_mapping_repr")

    concrete_path_type = type(Path())

    class HostilePath(concrete_path_type):
        def as_posix(self) -> str:
            hostile("path_as_posix")

        def __str__(self) -> str:
            hostile("path_str")

        def __repr__(self) -> str:
            hostile("path_repr")

        def __hash__(self) -> int:
            hostile("path_hash")

    cfg = _valid_config()
    cfg["seed"] = HostileInt(1)
    cfg["experiment"] = HostileStr("attacker-experiment")
    cfg["cohorts"]["external"] = HostileList(["attacker-slide"])
    cfg["marker_genes"] = HostileDict({"attacker": ["GENE"]})
    cfg["gene_modules"] = HostileMapping()
    cfg["patches"]["version"] = HostilePath("attacker-path")
    cfg["training"]["lr"] = HostileFloat(0.1)
    cfg["evaluation"]["primary_metrics"]["classification"] = HostileTuple(
        ("accuracy",)
    )
    calls.clear()

    reversed_cfg = {key: cfg[key] for key in reversed(cfg)}
    errors: list[ConfigValidationError] = []
    for candidate in (cfg, reversed_cfg):
        with pytest.raises(ConfigValidationError) as caught:
            resolve_config(candidate)
        errors.append(caught.value)

    schema_paths = [issue.path for issue in errors[0].issues[:8]]
    assert schema_paths == [
        "seed",
        "experiment",
        "cohorts.external",
        "marker_genes",
        "gene_modules",
        "patches.version",
        "training.lr",
        "evaluation.primary_metrics.classification",
    ]
    message = str(errors[0])
    for label in (
        "HostileInt",
        "HostileFloat",
        "HostileStr",
        "HostileList",
        "HostileTuple",
        "HostileDict",
        "HostileMapping",
        "HostilePath",
    ):
        assert f"<{label}>" in message
    assert "ATTACKER" not in message
    assert str(errors[0]) == str(errors[1])
    assert calls == {}


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


def test_default_yaml_adds_only_fail_closed_cohort_policy() -> None:
    raw = _valid_config()
    assert set(raw) == ORIGINAL_TOP_LEVEL_KEYS | {"cohort_policy"}
    assert raw["cohort_policy"] == {"allow_partial": False}

    loaded = load_config()
    assert loaded["cohort_policy"]["allow_partial"] is False
    assert set(loaded) == set(raw)


def test_load_config_accepts_custom_path_and_returns_fresh_dicts(tmp_path: Path) -> None:
    custom_path = tmp_path / "experiment.yaml"
    custom_path.write_text(yaml.safe_dump(_valid_config()), encoding="utf-8")

    first = load_config(custom_path)
    second = load_config(custom_path)
    assert first == second
    assert type(first) is dict
    assert type(first["cohorts"]) is dict
    assert type(first["cohorts"]["oncology"]) is list
    first["training"]["epochs"] = 999
    assert second["training"]["epochs"] != 999


@pytest.mark.parametrize(
    "content",
    [
        "",
        "{}\n",
        "cohorts: [not, a, mapping]\n",
        "cohorts: [unterminated\n",
    ],
)
def test_load_config_routes_empty_and_malformed_yaml_to_domain_error(
    tmp_path: Path, content: str
) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigValidationError):
        load_config(path)


def test_explicit_empty_config_is_validated_without_default_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel_called = False

    def unexpected_default_load(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal sentinel_called
        sentinel_called = True
        raise AssertionError("explicit config must not reload defaults")

    monkeypatch.setattr(data, "load_config", unexpected_default_load)
    with pytest.raises(ConfigValidationError):
        cohort_slide_ids({})
    assert sentinel_called is False


def test_invalid_startup_config_fails_before_side_effect_seam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    invalid_path = tmp_path / "invalid.yaml"
    invalid_path.write_text("{}\n", encoding="utf-8")
    side_effect_called = False

    def forbidden_side_effect() -> Path:
        nonlocal side_effect_called
        side_effect_called = True
        raise AssertionError("side effect reached before config resolution")

    monkeypatch.setattr(data, "pharma_processed_dir", forbidden_side_effect)
    with pytest.raises(ConfigValidationError):
        load_config(invalid_path)
        data.pharma_processed_dir()
    assert side_effect_called is False
