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
_CONCRETE_PATH_TYPE = type(Path())
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
            f"- {issue.path}: received {_safe_received(issue.received)}; expected "
            f"{issue.expected}. {issue.guidance}"
            for issue in self.issues
        )
        super().__init__("\n".join(lines))


class CohortAdmissionInputError(PharmaValidationError):
    """Raised when injected admission evidence is malformed or non-canonical."""


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

    def __init__(
        self,
        *,
        stage: str,
        subject: str,
        observed: int,
        minimum: int,
        guidance: str,
        shape: tuple[int, ...] | None = None,
        message: str | None = None,
    ) -> None:
        self.stage = stage
        self.subject = subject
        self.observed = int(observed)
        self.minimum = int(minimum)
        self.shape = None if shape is None else tuple(int(size) for size in shape)
        details = f"observed count={self.observed}"
        if self.shape is not None:
            details += f", observed shape={self.shape}"
        if message is None:
            message = (
                f"{self.stage}: {self.subject} is empty ({details}, expected "
                f">={self.minimum}). {guidance}"
            )
        super().__init__(message)


class PreprocessingValidationError(PharmaValidationError):
    """Structured scientific nonviability raised before graph construction."""

    def __init__(
        self,
        *,
        stage: str,
        reason_code: str,
        counts: dict[str, int],
        requested: dict[str, int],
        guidance: str,
    ) -> None:
        self.stage = stage
        self.reason_code = reason_code
        self.counts = json.loads(_canonical_json(counts))
        self.requested = json.loads(_canonical_json(requested))
        self.guidance = guidance
        super().__init__(
            f"{stage}: preprocessing is scientifically nonviable "
            f"({reason_code}). {guidance}"
        )


def require_non_empty(
    value: object,
    *,
    stage: str,
    subject: str,
    minimum: int = 1,
    shape: tuple[int, ...] | None = None,
    guidance: str,
) -> None:
    """Reject empty or undersized values with stable structured diagnostics."""
    resolved_shape = shape
    if resolved_shape is None:
        candidate = getattr(value, "shape", None)
        if candidate is not None:
            resolved_shape = tuple(int(size) for size in candidate)
    if resolved_shape is not None and resolved_shape:
        observed = resolved_shape[0]
    else:
        try:
            observed = len(value)  # type: ignore[arg-type]
        except TypeError as exc:
            raise TypeError("require_non_empty value must be sized") from exc
    if observed < minimum:
        raise StageValidationError(
            stage=stage,
            subject=subject,
            observed=observed,
            minimum=minimum,
            shape=resolved_shape,
            guidance=guidance,
        )


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
class PreprocessingResolution:
    """Immutable canonical preprocessing resolution and provenance record."""

    canonical_json: str

    def to_dict(self) -> dict[str, Any]:
        """Return a fresh exact-built-in primitive tree."""
        value = json.loads(self.canonical_json)
        if type(value) is not dict:  # pragma: no cover - constructor invariant
            raise TypeError("Preprocessing resolution root must be a JSON object.")
        return value


def _manifest_error(guidance: str) -> PreprocessingValidationError:
    return PreprocessingValidationError(
        stage="run_provenance",
        reason_code="malformed_preprocessing_manifest",
        counts={},
        requested={},
        guidance=guidance,
    )


def _admit_manifest_json(value: object, path: str) -> object:
    """Copy an untrusted candidate only after exact JSON-primitive admission."""
    value_type = type(value)
    if value is None or value_type in (str, bool):
        return value
    if value_type is int:
        if value.bit_length() > 4096:
            raise _manifest_error(f"Replace the oversized integer at {path}.")
        return value
    if value_type is float:
        if not math.isfinite(value):
            raise _manifest_error(f"Replace the non-finite number at {path}.")
        return value
    if value_type is list:
        return [
            _admit_manifest_json(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if value_type is dict:
        keys = list(value.keys())
        if any(type(key) is not str for key in keys):
            raise _manifest_error(f"Use exact string mapping keys at {path}.")
        return {
            key: _admit_manifest_json(value[key], f"{path}.{key}")
            for key in sorted(keys)
        }
    raise _manifest_error(f"Use only exact JSON primitives at {path}.")


@dataclass(frozen=True, slots=True, init=False)
class PreprocessingManifest:
    """Canonical admitted-order preprocessing provenance for one run."""

    canonical_json: str

    def __init__(self, *, slide_ids: object, records: object) -> None:
        admitted_ids = _admit_manifest_json(slide_ids, "slide_ids")
        admitted_records = _admit_manifest_json(records, "records")
        if type(admitted_ids) is not list or any(
            type(slide_id) is not str or not slide_id.strip()
            for slide_id in admitted_ids
        ):
            raise _manifest_error("Provide slide_ids as a list of non-empty strings.")
        if len(set(admitted_ids)) != len(admitted_ids):
            raise _manifest_error("Remove duplicate admitted slide IDs.")
        if type(admitted_records) is not list or len(admitted_records) != len(admitted_ids):
            raise _manifest_error("Provide exactly one preprocessing record per slide.")
        expected_top = {
            "schema_version",
            "slide_id",
            "counts",
            "exclusions",
            "requested",
            "resolved",
            "reasons",
        }
        expected_sections = {
            "counts": {
                "input_spots", "input_genes", "after_filter_cells_spots",
                "after_filter_cells_genes", "after_filter_genes_spots",
                "after_filter_genes_genes", "post_qc_spots", "post_qc_genes",
                "actual_hvgs",
            },
            "exclusions": {
                "cell_filter", "gene_filter", "mitochondrial_filter",
                "hvg_not_selected",
            },
            "requested": {"hvg", "pca", "neighbors", "graph_pcs"},
            "resolved": {"hvg_call", "pca", "neighbors", "graph_pcs"},
            "reasons": {"hvg", "pca", "neighbors", "graph_pcs"},
        }
        for index, (slide_id, record) in enumerate(zip(admitted_ids, admitted_records)):
            if type(record) is not dict or set(record) != expected_top:
                raise _manifest_error(f"Record {index} has an invalid schema.")
            if record["schema_version"] != "spatial-pharma-preprocessing-v1":
                raise _manifest_error(f"Record {index} has an unsupported schema version.")
            if record["slide_id"] != slide_id:
                raise _manifest_error(f"Record {index} does not match admitted slide order.")
            for section, keys in expected_sections.items():
                if type(record[section]) is not dict or set(record[section]) != keys:
                    raise _manifest_error(f"Record {index} has an invalid {section} section.")
            try:
                expected = finalize_preprocessing_resolution(
                    resolve_post_qc_preprocessing(
                        slide_id=slide_id,
                        input_spots=record["counts"]["input_spots"],
                        input_genes=record["counts"]["input_genes"],
                        after_filter_cells_spots=record["counts"][
                            "after_filter_cells_spots"
                        ],
                        after_filter_cells_genes=record["counts"][
                            "after_filter_cells_genes"
                        ],
                        after_filter_genes_spots=record["counts"][
                            "after_filter_genes_spots"
                        ],
                        after_filter_genes_genes=record["counts"][
                            "after_filter_genes_genes"
                        ],
                        post_qc_spots=record["counts"]["post_qc_spots"],
                        post_qc_genes=record["counts"]["post_qc_genes"],
                        requested_hvg=record["requested"]["hvg"],
                        requested_pcs=record["requested"]["pca"],
                        requested_neighbors=record["requested"]["neighbors"],
                        requested_graph_pcs=record["requested"]["graph_pcs"],
                    ),
                    actual_hvgs=record["counts"]["actual_hvgs"],
                ).to_dict()
            except PreprocessingValidationError:
                raise _manifest_error(
                    f"Record {index} contains invalid preprocessing values."
                ) from None
            if record != expected:
                raise _manifest_error(
                    f"Record {index} is inconsistent with deterministic resolution."
                )
        value = {
            "schema_version": "spatial-pharma-preprocessing-manifest-v1",
            "slide_ids": admitted_ids,
            "records": admitted_records,
        }
        object.__setattr__(self, "canonical_json", _canonical_json(value))

    def to_dict(self) -> dict[str, Any]:
        """Return a fresh exact-built-in manifest tree."""
        value = json.loads(self.canonical_json)
        if type(value) is not dict:  # pragma: no cover - constructor invariant
            raise TypeError("Preprocessing manifest root must be a JSON object.")
        return value


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _preprocessing_input_error(field: str) -> PreprocessingValidationError:
    return PreprocessingValidationError(
        stage="input",
        reason_code="invalid_preprocessing_input",
        counts={},
        requested={},
        guidance=f"Provide {field} as an exact nonnegative built-in integer.",
    )


def _admit_preprocessing_int(value: object, field: str, *, positive: bool = False) -> int:
    if type(value) is not int or value.bit_length() > 4096:
        raise _preprocessing_input_error(field)
    if value < (1 if positive else 0):
        raise _preprocessing_input_error(field)
    return value


def resolve_post_qc_preprocessing(
    *,
    slide_id: str,
    input_spots: int,
    input_genes: int,
    after_filter_cells_spots: int,
    after_filter_cells_genes: int,
    after_filter_genes_spots: int,
    after_filter_genes_genes: int,
    post_qc_spots: int,
    post_qc_genes: int,
    requested_hvg: int,
    requested_pcs: int,
    requested_neighbors: int,
    requested_graph_pcs: int,
) -> PreprocessingResolution:
    """Resolve the HVG call from observed post-QC cardinalities."""
    if type(slide_id) is not str or not slide_id.strip():
        raise _preprocessing_input_error("slide_id")
    raw_counts = {
        "input_spots": input_spots,
        "input_genes": input_genes,
        "after_filter_cells_spots": after_filter_cells_spots,
        "after_filter_cells_genes": after_filter_cells_genes,
        "after_filter_genes_spots": after_filter_genes_spots,
        "after_filter_genes_genes": after_filter_genes_genes,
        "post_qc_spots": post_qc_spots,
        "post_qc_genes": post_qc_genes,
    }
    counts = {
        key: _admit_preprocessing_int(value, key)
        for key, value in raw_counts.items()
    }
    raw_requested = {
        "hvg": requested_hvg,
        "pca": requested_pcs,
        "neighbors": requested_neighbors,
        "graph_pcs": requested_graph_pcs,
    }
    requested = {
        key: _admit_preprocessing_int(value, f"requested_{key}", positive=True)
        for key, value in raw_requested.items()
    }
    if counts["post_qc_spots"] < 3:
        raise PreprocessingValidationError(
            stage="post_qc",
            reason_code="insufficient_post_qc_spots",
            counts=counts,
            requested=requested,
            guidance="Retain at least three spots after QC or revise the QC thresholds.",
        )
    if counts["post_qc_genes"] < 2:
        raise PreprocessingValidationError(
            stage="post_qc",
            reason_code="insufficient_post_qc_genes",
            counts=counts,
            requested=requested,
            guidance="Retain at least two genes after QC or revise the QC thresholds.",
        )
    transitions = (
        ("input_spots", "after_filter_cells_spots"),
        ("after_filter_cells_spots", "after_filter_genes_spots"),
        ("after_filter_genes_spots", "post_qc_spots"),
        ("input_genes", "after_filter_cells_genes"),
        ("after_filter_cells_genes", "after_filter_genes_genes"),
        ("after_filter_genes_genes", "post_qc_genes"),
    )
    if any(counts[after] > counts[before] for before, after in transitions):
        raise _preprocessing_input_error("monotonic stage counts")
    resolved_hvg = min(requested["hvg"], counts["post_qc_genes"])
    record = {
        "schema_version": "spatial-pharma-preprocessing-v1",
        "slide_id": slide_id,
        "counts": {**counts, "actual_hvgs": None},
        "exclusions": {
            "cell_filter": counts["input_spots"] - counts["after_filter_cells_spots"],
            "gene_filter": counts["after_filter_cells_genes"]
            - counts["after_filter_genes_genes"],
            "mitochondrial_filter": counts["after_filter_genes_spots"]
            - counts["post_qc_spots"],
            "hvg_not_selected": None,
        },
        "requested": requested,
        "resolved": {
            "hvg_call": resolved_hvg,
            "pca": None,
            "neighbors": None,
            "graph_pcs": None,
        },
        "reasons": {
            "hvg": (
                "requested_value_accepted"
                if resolved_hvg == requested["hvg"]
                else "requested_value_capped_to_post_qc_genes"
            ),
            "pca": None,
            "neighbors": None,
            "graph_pcs": None,
        },
    }
    return PreprocessingResolution(_canonical_json(record))


def finalize_preprocessing_resolution(
    resolution: PreprocessingResolution, *, actual_hvgs: int
) -> PreprocessingResolution:
    """Resolve legal PCA and graph dimensions from the actual selected HVGs."""
    if type(resolution) is not PreprocessingResolution:
        raise _preprocessing_input_error("resolution")
    actual = _admit_preprocessing_int(actual_hvgs, "actual_hvgs")
    record = resolution.to_dict()
    counts = record["counts"]
    requested = record["requested"]
    counts["actual_hvgs"] = actual
    if actual < 2:
        raise PreprocessingValidationError(
            stage="post_hvg",
            reason_code="insufficient_actual_hvgs",
            counts=counts,
            requested=requested,
            guidance="Select at least two variable genes before PCA.",
        )
    if actual > counts["post_qc_genes"]:
        raise _preprocessing_input_error("actual_hvgs")
    pca_rank_limit = min(counts["post_qc_spots"], actual) - 1
    resolved_pca = min(requested["pca"], pca_rank_limit)
    resolved_neighbors = min(requested["neighbors"], counts["post_qc_spots"] - 1)
    if resolved_pca < 1:
        raise PreprocessingValidationError(
            stage="post_hvg",
            reason_code="no_legal_pca_components",
            counts=counts,
            requested=requested,
            guidance="Retain enough spots and variable genes for at least one component.",
        )
    if resolved_neighbors < 2:
        raise PreprocessingValidationError(
            stage="post_hvg",
            reason_code="insufficient_legal_neighbors",
            counts=counts,
            requested=requested,
            guidance="Retain at least three spots for a graph with two neighbors.",
        )
    resolved_graph_pcs = min(requested["graph_pcs"], resolved_pca)
    record["resolved"].update(
        pca=resolved_pca,
        neighbors=resolved_neighbors,
        graph_pcs=resolved_graph_pcs,
    )
    record["reasons"].update(
        pca=(
            "requested_value_accepted"
            if resolved_pca == requested["pca"]
            else "requested_value_capped_to_rank_limit"
        ),
        neighbors=(
            "requested_value_accepted"
            if resolved_neighbors == requested["neighbors"]
            else "requested_value_capped_to_spot_limit"
        ),
        graph_pcs=(
            "requested_value_accepted"
            if resolved_graph_pcs == requested["graph_pcs"]
            else "requested_value_capped_to_resolved_pcs"
        ),
    )
    record["exclusions"]["hvg_not_selected"] = counts["post_qc_genes"] - actual
    return PreprocessingResolution(_canonical_json(record))


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


def _safe_received(value: object) -> str:
    """Render bounded validation evidence without invoking user-defined reprs."""
    if value is None or type(value) in (bool, float):
        return repr(value)
    if type(value) is int:
        if value.bit_length() > 4096:
            return "<oversized integer>"
        return repr(value)
    if type(value) is str:
        rendered = repr(value)
        return rendered if len(rendered) <= 160 else f"{rendered[:157]}..."
    if type(value) in (Path, type(Path())):
        return "<path>"
    if type(value) in (list, tuple):
        if len(value) > 12:
            return f"<{type(value).__name__} length={len(value)}>"
        return "[" + ", ".join(_safe_received(item) for item in value) + "]"
    if type(value) is dict:
        return f"<mapping length={len(value)}>"
    return f"<{type(value).__name__}>"


def _invalid_key_sort_token(value: object) -> tuple[object, ...]:
    """Return a deterministic token without user repr, hash, or comparison."""
    value_type = type(value)
    if value is None:
        return (0,)
    if value_type is bool:
        return (1, int(value))
    if value_type is int:
        return (2, value)
    if value_type is float:
        return (3, value.hex())
    if value_type is bytes:
        return (4, len(value), value[:64].hex())
    if value_type is tuple:
        return (5, len(value), tuple(_invalid_key_sort_token(item) for item in value))
    return (99, value_type.__module__, value_type.__qualname__)


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
    # ``raw`` has already passed the exact-dict root gate in ``resolve_config``.
    # Copy only exact built-in dictionaries here: a Mapping/dict subclass may
    # override iteration, lookup, membership, or ``__deepcopy__``.
    if type(raw) is not dict:  # pragma: no cover - resolve_config invariant
        raise TypeError("configuration root must be an exact built-in dict")
    merged = raw.copy()
    for key, default in _OPTIONAL_DEFAULTS.items():
        if key not in merged:
            merged[key] = deepcopy(default)
        elif type(default) is dict and type(merged[key]) is dict:
            section = merged[key].copy()
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
) -> dict[str, Any] | None:
    if type(value) is not dict:
        issues.add(
            path,
            value,
            "a mapping with string keys",
            f"Set {path} to a YAML mapping with the documented fields.",
        )
        return None
    bad_keys = sorted(
        (key for key in value if type(key) is not str),
        key=_invalid_key_sort_token,
    )
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
            (key for key in value if type(key) is str and key not in allowed_set)
        ):
            issues.add(
                f"{path}.{key}",
                value[key],
                f"one of {tuple(allowed)}",
                "Remove the unknown key or correct its spelling.",
            )
    return value


def _required(mapping: dict[str, Any] | None, key: str, path: str, issues: _Issues):
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
    valid = type(value) is str and bool(value.strip())
    if valid and filename:
        valid = all(char not in value for char in '/\\\0') and value not in {".", ".."}
    if not valid:
        expected = "a non-empty filename-safe string" if filename else "a non-empty string"
        issues.add(path, value, expected, f"Set {path} to a non-empty string.")
        return None
    return value


def _boolean(value: object, path: str, issues: _Issues) -> bool | None:
    if type(value) is not bool:
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
    valid_type = type(value) is int if integer else type(value) in (int, float)
    finite = valid_type and (
        (type(value) is not int or value.bit_length() <= 4096)
        and (type(value) is not float or math.isfinite(value))
    )
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
    if type(value) is not str or value not in choices:
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
    if type(value) is not list:
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
    string_keys = [name for name in section if type(name) is str]
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
    value_type = type(value)
    if value_type is _CONCRETE_PATH_TYPE:
        return value.as_posix()
    if value is None or value_type in (str, bool):
        return value
    if value_type is int:
        if value.bit_length() > 4096:
            issues.add(
                path,
                value,
                "a bounded JSON integer",
                f"Replace the oversized integer at {path}.",
            )
            return None
        return value
    if value_type is float:
        if not math.isfinite(value):
            issues.add(path, value, "a finite JSON number", f"Replace the non-finite value at {path}.")
        return value
    if value_type is tuple:
        return [_validate_json_tree(item, f"{path}[{index}]", issues) for index, item in enumerate(value)]
    if value_type is list:
        return [_validate_json_tree(item, f"{path}[{index}]", issues) for index, item in enumerate(value)]
    if value_type is dict:
        result: dict[str, object] = {}
        for key in sorted(key for key in value if type(key) is str):
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
    if type(raw) is not dict:
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
    unattempted_slide_ids: Collection[str] | None = None,
) -> AdmittedRun:
    """Resolve and admit one ordered cohort without filesystem side effects.

    ``None`` availability means remote availability is not known yet. A supplied
    collection is treated as a complete local preflight result.
    """
    resolved = resolve_config(cfg)
    normalized = resolved.to_dict()
    allow_partial = normalized["cohort_policy"]["allow_partial"]
    configured_ids = {
        slide_id
        for cohort in ("oncology", "external", "benchmark")
        for slide_id in normalized["cohorts"][cohort]
    }
    def normalize_ids(
        values: Collection[str] | None, *, field: str
    ) -> set[str] | None:
        if values is None:
            return None
        if isinstance(values, (str, bytes)):
            raise CohortAdmissionInputError(
                f"{field} must be a collection of configured string slide IDs."
            )
        try:
            original_items = list(values)
        except TypeError as exc:
            raise CohortAdmissionInputError(
                f"{field} must be an iterable of configured string slide IDs."
            ) from exc
        if any(type(item) is not str for item in original_items):
            raise CohortAdmissionInputError(
                f"{field} contains a non-string slide ID."
            )
        if any(item not in configured_ids for item in original_items):
            raise CohortAdmissionInputError(
                f"{field} contains an unknown slide ID."
            )
        return set(original_items)

    available = normalize_ids(available_slide_ids, field="available_slide_ids")
    unattempted = normalize_ids(
        unattempted_slide_ids,
        field="unattempted_slide_ids",
    ) or set()
    failure_reasons: dict[str, str] = {}
    if failures is not None:
        if not isinstance(failures, Mapping):
            raise CohortAdmissionInputError(
                "failures must map configured string slide IDs to string details."
            )
        for slide_id, detail in failures.items():
            if not isinstance(slide_id, str) or slide_id not in configured_ids:
                raise CohortAdmissionInputError(
                    "failures contains an unknown or non-string slide ID."
                )
            if not isinstance(detail, str):
                raise CohortAdmissionInputError(
                    "failure details must be strings; arbitrary objects are not accepted."
                )
            failure_reasons[slide_id] = (
                "Source acquisition failed for the configured slide; verify "
                "network access and the public dataset identifier before retrying."
            )

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
            elif slide_id in unattempted:
                skipped.append(
                    SlideAdmission(
                        slide_id,
                        cohort,
                        "skipped",
                        "source_not_attempted",
                        "Source acquisition was not attempted after an earlier strict failure.",
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
