"""Pure validation contracts for Spatial Pharma DL experiment startup.

This module intentionally depends only on the Python standard library so an
invalid experiment can be rejected before scientific or model libraries load.
"""

from __future__ import annotations

import json
import math
from collections.abc import Collection, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MODEL_NAMES = (
    "resnet18",
    "resnet50",
    "efficientnet_b0",
    "convnext_tiny",
    "vit_b_16",
)
FOUNDATION_MODEL_NAMES = ("kaiko_vits16", "phikon")
DEVICE_NAMES = ("auto", "cpu", "cuda", "mps")
CLASSIFICATION_METRICS = ("balanced_accuracy", "macro_f1", "accuracy")
REGRESSION_METRICS = ("pearson_r", "r2", "mae")

_MISSING = object()
_ROOT_KEYS = (
    "seed",
    "experiment",
    "cohorts",
    "cohort_policy",
    "preprocessing",
    "labels",
    "marker_genes",
    "gene_modules",
    "patches",
    "training",
    "foundation",
    "evaluation",
)
_OPTIONAL_DEFAULTS: dict[str, Any] = {
    "seed": 0,
    "experiment": "v2_remediation",
    "cohort_policy": {"allow_partial": False},
    "patches": {
        "version": "v1",
        "context_scale": 3.0,
        "per_slide_stain_norm": True,
    },
    "training": {
        "model": "resnet18",
        "pretrained": True,
        "device": "auto",
        "augment": True,
    },
    "foundation": {
        "enabled": False,
        "model": "kaiko_vits16",
        "device": "auto",
        "batch_size": 64,
        "cache": True,
    },
}


class PharmaValidationError(ValueError):
    """Base class for actionable pharma workflow validation failures."""


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One deterministic configuration defect."""

    path: str
    received: object
    expected: str
    guidance: str


class ConfigValidationError(PharmaValidationError):
    """Raised once with all configuration issues discovered in one pass."""

    def __init__(self, issues: Sequence[ValidationIssue]):
        self.issues = tuple(issues)
        lines = [f"Experiment configuration has {len(self.issues)} issue(s):"]
        lines.extend(
            f"- {issue.path}: received {issue.received!r}; expected "
            f"{issue.expected}. {issue.guidance}"
            for issue in self.issues
        )
        super().__init__("\n".join(lines))


class CohortAdmissionError(PharmaValidationError):
    """Raised when a configured cohort cannot be admitted safely."""

    def __init__(self, manifest: CohortManifest, message: str | None = None):
        self.manifest = manifest
        unavailable = tuple(
            record.slide_id for record in (*manifest.skipped, *manifest.failed)
        )
        self.unavailable_slide_ids = tuple(dict.fromkeys(unavailable))
        if message is None:
            rendered = ", ".join(self.unavailable_slide_ids) or "none"
            message = (
                "Cohort admission failed; unavailable configured slides: "
                f"{rendered}. Correct the listed inputs or explicitly enable "
                "cohort_policy.allow_partial."
            )
        super().__init__(message)


class StageValidationError(PharmaValidationError):
    """Raised when an empty or undersized stage input is rejected."""


@dataclass(frozen=True, slots=True)
class ResolvedConfig:
    """Immutable canonical configuration with a mutable compatibility view."""

    canonical_json: str

    def to_dict(self) -> dict[str, Any]:
        """Return a fresh mutable plain JSON tree on every call."""
        value = json.loads(self.canonical_json)
        if not isinstance(value, dict):  # pragma: no cover - constructor invariant
            raise TypeError("Resolved configuration root must be a JSON object.")
        return value


@dataclass(frozen=True, slots=True)
class SlideAdmission:
    """One configured slide's deterministic admission outcome."""

    slide_id: str
    cohort: str
    status: str
    reason_code: str | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class CohortManifest:
    """Deterministic evidence for one cohort admission decision."""

    schema_version: str
    allow_partial: bool
    configured: tuple[SlideAdmission, ...]
    included: tuple[SlideAdmission, ...]
    skipped: tuple[SlideAdmission, ...]
    failed: tuple[SlideAdmission, ...]

    @property
    def canonical_json(self) -> str:
        """Return stable JSON containing only manifest primitives."""
        return json.dumps(
            {
                "schema_version": self.schema_version,
                "allow_partial": self.allow_partial,
                "configured": [_admission_dict(item) for item in self.configured],
                "included": [_admission_dict(item) for item in self.included],
                "skipped": [_admission_dict(item) for item in self.skipped],
                "failed": [_admission_dict(item) for item in self.failed],
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a fresh mutable JSON tree on every call."""
        value = json.loads(self.canonical_json)
        if not isinstance(value, dict):  # pragma: no cover - constructor invariant
            raise TypeError("Cohort manifest root must be a JSON object.")
        return value


@dataclass(frozen=True, slots=True)
class AdmittedRun:
    """Immutable resolved configuration and its admitted cohort."""

    config: ResolvedConfig
    manifest: CohortManifest

    @property
    def slide_ids(self) -> tuple[str, ...]:
        """Return admitted slide IDs in configuration order."""
        return tuple(record.slide_id for record in self.manifest.included)


def _admission_dict(record: SlideAdmission) -> dict[str, str | None]:
    return {
        "slide_id": record.slide_id,
        "cohort": record.cohort,
        "status": record.status,
        "reason_code": record.reason_code,
        "reason": record.reason,
    }


class _Issues:
    def __init__(self) -> None:
        self.values: list[ValidationIssue] = []

    def add(
        self,
        path: str,
        received: object,
        expected: str,
        guidance: str,
    ) -> None:
        self.values.append(ValidationIssue(path, received, expected, guidance))


def _merge_optional_defaults(raw: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(raw)
    for key, default in _OPTIONAL_DEFAULTS.items():
        if key not in merged:
            merged[key] = deepcopy(default)
        elif isinstance(default, Mapping) and isinstance(merged[key], Mapping):
            section = dict(merged[key])
            for nested_key, nested_default in default.items():
                section.setdefault(nested_key, deepcopy(nested_default))
            merged[key] = section
    return merged


def _mapping(
    value: object,
    path: str,
    issues: _Issues,
    *,
    allowed: Sequence[str] | None = None,
) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        issues.add(
            path,
            value,
            "a mapping with string keys",
            f"Set {path} to a YAML mapping with the documented fields.",
        )
        return None
    bad_keys = sorted((key for key in value if not isinstance(key, str)), key=repr)
    for key in bad_keys:
        issues.add(
            f"{path}.<key>",
            key,
            "a string mapping key",
            "Replace the key with a documented string name.",
        )
    if allowed is not None:
        allowed_set = set(allowed)
        for key in sorted(
            (key for key in value if isinstance(key, str) and key not in allowed_set)
        ):
            issues.add(
                f"{path}.{key}",
                value[key],
                f"one of {tuple(allowed)}",
                "Remove the unknown key or correct its spelling.",
            )
    return value  # type: ignore[return-value]


def _required(mapping: Mapping[str, Any] | None, key: str, path: str, issues: _Issues):
    if mapping is None or key not in mapping:
        issues.add(
            path,
            "<missing>",
            "a required value",
            f"Add {path} using the documented configuration contract.",
        )
        return _MISSING
    return mapping[key]


def _string(value: object, path: str, issues: _Issues, *, filename: bool = False):
    valid = isinstance(value, str) and bool(value.strip())
    if valid and filename:
        valid = all(char not in value for char in '/\\\0') and value not in {".", ".."}
    if not valid:
        expected = "a non-empty filename-safe string" if filename else "a non-empty string"
        issues.add(path, value, expected, f"Set {path} to a non-empty string.")
        return None
    return value


def _boolean(value: object, path: str, issues: _Issues) -> bool | None:
    if not isinstance(value, bool):
        issues.add(path, value, "a boolean", f"Set {path} to true or false.")
        return None
    return value


def _number(
    value: object,
    path: str,
    issues: _Issues,
    *,
    integer: bool = False,
    minimum: float | None = None,
    maximum: float | None = None,
    strict_minimum: bool = False,
) -> int | float | None:
    valid_type = isinstance(value, int if integer else (int, float)) and not isinstance(
        value, bool
    )
    finite = valid_type and math.isfinite(float(value))
    in_range = finite
    if in_range and minimum is not None:
        in_range = value > minimum if strict_minimum else value >= minimum
    if in_range and maximum is not None:
        in_range = value <= maximum
    if not in_range:
        kind = "integer" if integer else "finite number"
        bounds = []
        if minimum is not None:
            bounds.append(f"> {minimum}" if strict_minimum else f">= {minimum}")
        if maximum is not None:
            bounds.append(f"<= {maximum}")
        expected = f"a {kind}" + (f" ({', '.join(bounds)})" if bounds else "")
        issues.add(path, value, expected, f"Set {path} within the documented range.")
        return None
    return value  # type: ignore[return-value]


def _choice(value: object, path: str, issues: _Issues, choices: Sequence[str]):
    if not isinstance(value, str) or value not in choices:
        issues.add(path, value, f"one of {tuple(choices)}", f"Choose a supported {path} value.")
        return None
    return value


def _string_list(
    value: object,
    path: str,
    issues: _Issues,
    *,
    minimum: int = 1,
    choices: Sequence[str] | None = None,
) -> list[str] | None:
    if not isinstance(value, list):
        issues.add(path, value, "a list of strings", f"Set {path} to a YAML list.")
        return None
    if len(value) < minimum:
        issues.add(path, value, f"at least {minimum} string value(s)", f"Add values to {path}.")
    seen: set[str] = set()
    result: list[str] = []
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        valid = _string(item, item_path, issues)
        if valid is None:
            continue
        if choices is not None and valid not in choices:
            issues.add(
                item_path,
                valid,
                f"one of {tuple(choices)}",
                f"Choose a supported value for {path}.",
            )
        if valid in seen:
            issues.add(item_path, valid, "a unique value", f"Remove the duplicate from {path}.")
        else:
            seen.add(valid)
        result.append(valid)
    return result


def _validate_cohorts(root: Mapping[str, Any], issues: _Issues) -> None:
    cohorts = _mapping(
        _required(root, "cohorts", "cohorts", issues),
        "cohorts",
        issues,
        allowed=("oncology", "external", "benchmark"),
    )
    all_seen: dict[str, str] = {}
    total = 0
    for name in ("oncology", "external", "benchmark"):
        values = _string_list(
            _required(cohorts, name, f"cohorts.{name}", issues),
            f"cohorts.{name}",
            issues,
            minimum=0,
        )
        if values is None:
            continue
        total += len(values)
        if name == "oncology" and len(values) < 2:
            issues.add(
                "cohorts.oncology",
                values,
                "at least two slide IDs",
                "Configure two oncology slides so LOSO evaluation is possible.",
            )
        for index, slide_id in enumerate(values):
            if slide_id in all_seen and all_seen[slide_id] != name:
                issues.add(
                    f"cohorts.{name}[{index}]",
                    slide_id,
                    "a slide ID unique across all cohort lists",
                    f"Remove the duplicate also configured in cohorts.{all_seen[slide_id]}.",
                )
            elif slide_id not in all_seen:
                all_seen[slide_id] = name
    if total == 0:
        issues.add(
            "cohorts",
            [],
            "at least one configured slide",
            "Add a slide ID to a cohort list.",
        )


def _validate_preprocessing(root: Mapping[str, Any], issues: _Issues) -> None:
    keys = (
        "min_counts",
        "min_cells",
        "max_pct_mito",
        "n_top_genes_hvg",
        "n_pcs",
        "n_neighbors",
        "n_pcs_neighbors",
        "leiden_resolution",
    )
    section = _mapping(
        _required(root, "preprocessing", "preprocessing", issues),
        "preprocessing",
        issues,
        allowed=keys,
    )
    integers: dict[str, int | float | None] = {}
    for key in ("min_counts", "min_cells", "n_top_genes_hvg", "n_pcs", "n_neighbors", "n_pcs_neighbors"):
        integers[key] = _number(
            _required(section, key, f"preprocessing.{key}", issues),
            f"preprocessing.{key}",
            issues,
            integer=True,
            minimum=1,
        )
    _number(
        _required(section, "max_pct_mito", "preprocessing.max_pct_mito", issues),
        "preprocessing.max_pct_mito",
        issues,
        minimum=0,
        maximum=100,
    )
    _number(
        _required(section, "leiden_resolution", "preprocessing.leiden_resolution", issues),
        "preprocessing.leiden_resolution",
        issues,
        minimum=0,
        strict_minimum=True,
    )
    if integers["n_pcs"] is not None and integers["n_pcs_neighbors"] is not None:
        if integers["n_pcs_neighbors"] > integers["n_pcs"]:
            issues.add(
                "preprocessing.n_pcs_neighbors",
                integers["n_pcs_neighbors"],
                "less than or equal to preprocessing.n_pcs",
                "Reduce n_pcs_neighbors or increase n_pcs.",
            )


def _validate_labels(root: Mapping[str, Any], issues: _Issues) -> None:
    section = _mapping(
        _required(root, "labels", "labels", issues),
        "labels",
        issues,
        allowed=("classification_col", "regression_targets", "tme_classes"),
    )
    _string(
        _required(section, "classification_col", "labels.classification_col", issues),
        "labels.classification_col",
        issues,
    )
    _choice(
        _required(section, "regression_targets", "labels.regression_targets", issues),
        "labels.regression_targets",
        issues,
        ("modules", "genes", "both"),
    )
    classes = _string_list(
        _required(section, "tme_classes", "labels.tme_classes", issues),
        "labels.tme_classes",
        issues,
    )
    if classes is not None and "other" not in classes:
        issues.add(
            "labels.tme_classes",
            classes,
            "a unique class list containing 'other'",
            "Add the fallback 'other' class.",
        )


def _validate_dynamic_gene_map(
    root: Mapping[str, Any], key: str, issues: _Issues, *, minimum_genes: int
) -> None:
    section = _mapping(_required(root, key, key, issues), key, issues)
    if section is None:
        return
    string_keys = [name for name in section if isinstance(name, str)]
    if not string_keys:
        issues.add(key, section, "a non-empty mapping", f"Add at least one named {key} entry.")
    for name in sorted(string_keys):
        if not name.strip():
            issues.add(f"{key}.<key>", name, "a non-empty name", f"Name each {key} entry.")
        _string_list(section[name], f"{key}.{name}", issues, minimum=minimum_genes)
    if key == "marker_genes" and "breast" not in section:
        issues.add(
            "marker_genes.breast",
            "<missing>",
            "a non-empty fallback gene panel",
            "Add the breast marker panel used by the runtime fallback.",
        )


def _validate_patches(root: Mapping[str, Any], issues: _Issues) -> None:
    keys = ("version", "context_scale", "min_patch_px", "output_size", "per_slide_stain_norm")
    section = _mapping(
        _required(root, "patches", "patches", issues), "patches", issues, allowed=keys
    )
    _string(_required(section, "version", "patches.version", issues), "patches.version", issues)
    _number(
        _required(section, "context_scale", "patches.context_scale", issues),
        "patches.context_scale",
        issues,
        minimum=0,
        strict_minimum=True,
    )
    for key in ("min_patch_px", "output_size"):
        _number(
            _required(section, key, f"patches.{key}", issues),
            f"patches.{key}",
            issues,
            integer=True,
            minimum=2,
        )
    _boolean(
        _required(section, "per_slide_stain_norm", "patches.per_slide_stain_norm", issues),
        "patches.per_slide_stain_norm",
        issues,
    )


def _validate_training(root: Mapping[str, Any], issues: _Issues) -> None:
    keys = (
        "model", "pretrained", "device", "batch_size", "epochs", "lr",
        "weight_decay", "cls_weight", "reg_weight", "patience", "num_workers", "augment",
    )
    section = _mapping(
        _required(root, "training", "training", issues), "training", issues, allowed=keys
    )
    _choice(_required(section, "model", "training.model", issues), "training.model", issues, MODEL_NAMES)
    _choice(_required(section, "device", "training.device", issues), "training.device", issues, DEVICE_NAMES)
    for key in ("pretrained", "augment"):
        _boolean(_required(section, key, f"training.{key}", issues), f"training.{key}", issues)
    for key in ("batch_size", "epochs", "patience"):
        _number(
            _required(section, key, f"training.{key}", issues), f"training.{key}", issues,
            integer=True, minimum=1,
        )
    _number(
        _required(section, "num_workers", "training.num_workers", issues),
        "training.num_workers", issues, integer=True, minimum=0,
    )
    _number(
        _required(section, "lr", "training.lr", issues), "training.lr", issues,
        minimum=0, strict_minimum=True,
    )
    weights: dict[str, int | float | None] = {}
    for key in ("weight_decay", "cls_weight", "reg_weight"):
        weights[key] = _number(
            _required(section, key, f"training.{key}", issues), f"training.{key}", issues,
            minimum=0,
        )
    if weights["cls_weight"] == 0 and weights["reg_weight"] == 0:
        issues.add(
            "training.cls_weight",
            0,
            "at least one nonzero task weight",
            "Set cls_weight or reg_weight above zero.",
        )


def _validate_foundation(root: Mapping[str, Any], issues: _Issues) -> None:
    keys = ("enabled", "model", "device", "batch_size", "cache")
    section = _mapping(
        _required(root, "foundation", "foundation", issues), "foundation", issues, allowed=keys
    )
    for key in ("enabled", "cache"):
        _boolean(_required(section, key, f"foundation.{key}", issues), f"foundation.{key}", issues)
    _choice(
        _required(section, "model", "foundation.model", issues),
        "foundation.model", issues, FOUNDATION_MODEL_NAMES,
    )
    _choice(
        _required(section, "device", "foundation.device", issues),
        "foundation.device", issues, DEVICE_NAMES,
    )
    _number(
        _required(section, "batch_size", "foundation.batch_size", issues),
        "foundation.batch_size", issues, integer=True, minimum=1,
    )


def _validate_evaluation(root: Mapping[str, Any], issues: _Issues) -> None:
    section = _mapping(
        _required(root, "evaluation", "evaluation", issues),
        "evaluation", issues, allowed=("primary_metrics",),
    )
    metrics = _mapping(
        _required(section, "primary_metrics", "evaluation.primary_metrics", issues),
        "evaluation.primary_metrics", issues, allowed=("classification", "regression"),
    )
    _string_list(
        _required(metrics, "classification", "evaluation.primary_metrics.classification", issues),
        "evaluation.primary_metrics.classification", issues, choices=CLASSIFICATION_METRICS,
    )
    _string_list(
        _required(metrics, "regression", "evaluation.primary_metrics.regression", issues),
        "evaluation.primary_metrics.regression", issues, choices=REGRESSION_METRICS,
    )


def _validate_json_tree(value: object, path: str, issues: _Issues) -> object:
    if isinstance(value, Path):
        return value.as_posix()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            issues.add(path, value, "a finite JSON number", f"Replace the non-finite value at {path}.")
        return value
    if isinstance(value, tuple):
        return [_validate_json_tree(item, f"{path}[{index}]", issues) for index, item in enumerate(value)]
    if isinstance(value, list):
        return [_validate_json_tree(item, f"{path}[{index}]", issues) for index, item in enumerate(value)]
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key in sorted(value, key=repr):
            if not isinstance(key, str):
                issues.add(f"{path}.<key>", key, "a string JSON object key", "Replace the non-string key.")
                continue
            result[key] = _validate_json_tree(value[key], f"{path}.{key}", issues)
        return result
    issues.add(
        path,
        value,
        "a JSON-safe primitive, list, tuple, mapping, or pathlib.Path",
        "Remove arbitrary Python objects and sets from the configuration.",
    )
    return None


def resolve_config(raw: Mapping[str, Any] | None) -> ResolvedConfig:
    """Resolve documented defaults and aggregate all schema defects.

    The returned JSON sorts mapping keys while preserving every list's order.
    No strings or numbers are coerced.
    """
    issues = _Issues()
    if not isinstance(raw, Mapping):
        issues.add(
            "config",
            raw,
            "a non-empty YAML mapping",
            "Provide a YAML document whose root is a mapping.",
        )
        raise ConfigValidationError(issues.values)
    root = _merge_optional_defaults(raw)
    _mapping(root, "config", issues, allowed=_ROOT_KEYS)

    _number(_required(root, "seed", "seed", issues), "seed", issues, integer=True, minimum=0)
    _string(_required(root, "experiment", "experiment", issues), "experiment", issues, filename=True)
    _validate_cohorts(root, issues)
    policy = _mapping(
        _required(root, "cohort_policy", "cohort_policy", issues),
        "cohort_policy", issues, allowed=("allow_partial",),
    )
    _boolean(
        _required(policy, "allow_partial", "cohort_policy.allow_partial", issues),
        "cohort_policy.allow_partial", issues,
    )
    _validate_preprocessing(root, issues)
    _validate_labels(root, issues)
    _validate_dynamic_gene_map(root, "marker_genes", issues, minimum_genes=1)
    _validate_dynamic_gene_map(root, "gene_modules", issues, minimum_genes=2)
    _validate_patches(root, issues)
    _validate_training(root, issues)
    _validate_foundation(root, issues)
    _validate_evaluation(root, issues)

    canonical_value = _validate_json_tree(root, "config", issues)
    if issues.values:
        raise ConfigValidationError(issues.values)
    canonical_json = json.dumps(
        canonical_value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return ResolvedConfig(canonical_json)


def admit_run(
    cfg: Mapping[str, Any],
    *,
    available_slide_ids: Collection[str] | None = None,
    failures: Mapping[str, str] | None = None,
) -> AdmittedRun:
    """Resolve and admit one ordered cohort without filesystem side effects.

    ``None`` availability means remote availability is not known yet. A supplied
    collection is treated as a complete local preflight result.
    """
    resolved = resolve_config(cfg)
    normalized = resolved.to_dict()
    allow_partial = normalized["cohort_policy"]["allow_partial"]
    available = None if available_slide_ids is None else set(available_slide_ids)
    failure_reasons = dict(failures or {})

    configured: list[SlideAdmission] = []
    included: list[SlideAdmission] = []
    skipped: list[SlideAdmission] = []
    failed: list[SlideAdmission] = []

    for cohort in ("oncology", "external", "benchmark"):
        for slide_id in normalized["cohorts"][cohort]:
            configured.append(SlideAdmission(slide_id, cohort, "configured"))
            if slide_id in failure_reasons:
                reason = failure_reasons[slide_id]
                failed.append(
                    SlideAdmission(
                        slide_id,
                        cohort,
                        "failed",
                        "source_load_failed",
                        reason,
                    )
                )
                skipped.append(
                    SlideAdmission(
                        slide_id,
                        cohort,
                        "skipped",
                        "source_load_failed",
                        "Source loading failed; correct the source or rerun with a usable slide.",
                    )
                )
            elif available is not None and slide_id not in available:
                skipped.append(
                    SlideAdmission(
                        slide_id,
                        cohort,
                        "skipped",
                        "missing_processed_slide",
                        "Create the processed slide cache before running cache-backed stages.",
                    )
                )
            else:
                included.append(SlideAdmission(slide_id, cohort, "included"))

    manifest = CohortManifest(
        schema_version="cohort-manifest-v1",
        allow_partial=allow_partial,
        configured=tuple(configured),
        included=tuple(included),
        skipped=tuple(skipped),
        failed=tuple(failed),
    )
    if (skipped or failed) and not allow_partial:
        raise CohortAdmissionError(manifest)
    if not included:
        raise CohortAdmissionError(
            manifest,
            "Cohort admission failed: no usable configured slides remain.",
        )
    admitted_oncology = sum(item.cohort == "oncology" for item in included)
    if normalized["cohorts"]["oncology"] and admitted_oncology < 2:
        raise CohortAdmissionError(
            manifest,
            "Cohort admission failed: the oncology LOSO benchmark requires at "
            "least two admitted oncology slides.",
        )
    return AdmittedRun(resolved, manifest)
