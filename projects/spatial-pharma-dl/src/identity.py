"""Exact compound spot identity and complete metadata-order alignment."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from typing import Any, Sequence

import pandas as pd

from .validation import PharmaValidationError


KEY_COLUMNS = ("slide_id", "spot_id")
RESERVED_COLUMNS = ("_label_source_row", "_patch_source_row")
_SAMPLE_LIMIT = 5
_SAMPLE_COMPONENT_LIMIT = 64


@dataclass(frozen=True, slots=True)
class IdentityIssue:
    """One deterministic compound-identity defect."""

    code: str
    side: str
    count: int
    sample_keys: tuple[tuple[str, str], ...] = ()
    sample_rows: tuple[tuple[int, str, str], ...] = ()


class IdentityValidationError(PharmaValidationError):
    """Raised with all safely discoverable identity issues in schema order."""

    def __init__(self, *, stage: str, issues: Sequence[IdentityIssue]) -> None:
        self.stage = stage
        self.issues = tuple(issues)
        lines = [f"{stage}: compound identity has {len(self.issues)} issue(s):"]
        for issue in self.issues:
            detail = (
                f"- {issue.code} [{issue.side}]: count={issue.count}"
            )
            if issue.sample_keys:
                rendered = ", ".join(
                    f"({_render_key_component(slide_id)}, "
                    f"{_render_key_component(spot_id)})"
                    for slide_id, spot_id in issue.sample_keys
                )
                detail += f"; sample_keys={rendered}"
            if issue.sample_rows:
                rendered = ", ".join(
                    f"row={row}/{column}/{type_name}"
                    for row, column, type_name in issue.sample_rows
                )
                detail += f"; sample_rows={rendered}"
            lines.append(detail)
        super().__init__("\n".join(lines))


def _type_label(value: object) -> str:
    """Return inert evidence without reading caller-controlled class attributes."""
    if value is None:
        return "builtins.NoneType"
    if type(value) is bool:
        return "builtins.bool"
    if type(value) is int:
        return "builtins.int"
    if type(value) is float:
        return "builtins.float"
    if type(value) is str:
        return "builtins.str"
    if type(value) is bytes:
        return "builtins.bytes"
    if type(value) is list:
        return "builtins.list"
    if type(value) is tuple:
        return "builtins.tuple"
    if type(value) is dict:
        return "builtins.dict"
    return "non_builtin_object"


def _render_key_component(value: str) -> str:
    """Render one admitted exact string as bounded, single-line JSON evidence."""
    prefix = value[:_SAMPLE_COMPONENT_LIMIT]
    rendered = json.dumps(prefix, ensure_ascii=True)
    omitted = len(value) - len(prefix)
    if omitted:
        rendered = f'{rendered[:-1]}...<+{omitted} chars>"'
    return rendered


def _parameter_issue(name: str, value: object) -> IdentityIssue | None:
    if type(value) is not str:
        return IdentityIssue(
            code="invalid_type",
            side="parameters",
            count=1,
            sample_rows=((0, name, _type_label(value)),),
        )
    if not value.strip():
        return IdentityIssue(
            code="blank_value",
            side="parameters",
            count=1,
            sample_rows=((0, name, "builtins.str"),),
        )
    return None


def _validate_parameter(
    name: str,
    value: object,
    *,
    stage_for_error: str,
) -> str:
    issue = _parameter_issue(name, value)
    if issue is not None:
        raise IdentityValidationError(stage=stage_for_error, issues=(issue,))
    return value  # type: ignore[return-value]


def _schema_issues(frame: pd.DataFrame, side: str) -> list[IdentityIssue]:
    issues: list[IdentityIssue] = []
    columns = tuple(frame.columns)
    for column in KEY_COLUMNS:
        if column not in columns:
            issues.append(
                IdentityIssue(code="missing_column", side=side, count=1)
            )
    for column in RESERVED_COLUMNS:
        if column in columns:
            issues.append(
                IdentityIssue(code="reserved_column", side=side, count=1)
            )
    return issues


def _admit_key_rows(
    frame: pd.DataFrame,
    side: str,
) -> tuple[tuple[tuple[str, str], ...], list[IdentityIssue]]:
    invalid: list[tuple[int, str, str]] = []
    blank: list[tuple[int, str, str]] = []

    for row in range(len(frame)):
        for column in KEY_COLUMNS:
            value = frame[column].iat[row]
            if type(value) is not str:
                invalid.append((row, column, _type_label(value)))
            elif not value.strip():
                blank.append((row, column, "builtins.str"))

    issues: list[IdentityIssue] = []
    if invalid:
        issues.append(
            IdentityIssue(
                code="invalid_type",
                side=side,
                count=len(invalid),
                sample_rows=tuple(invalid[:_SAMPLE_LIMIT]),
            )
        )
    if blank:
        issues.append(
            IdentityIssue(
                code="blank_value",
                side=side,
                count=len(blank),
                sample_rows=tuple(blank[:_SAMPLE_LIMIT]),
            )
        )
    if issues:
        return (), issues

    rows = tuple(
        (frame["slide_id"].iat[row], frame["spot_id"].iat[row])
        for row in range(len(frame))
    )
    return rows, []


def _sample(keys: Sequence[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(keys)[:_SAMPLE_LIMIT])


def _duplicate_issue(
    keys: tuple[tuple[str, str], ...], side: str
) -> IdentityIssue | None:
    counts = Counter(keys)
    duplicates = [key for key, count in counts.items() if count > 1]
    if not duplicates:
        return None
    return IdentityIssue(
        code="duplicate_key",
        side=side,
        count=sum(counts[key] - 1 for key in duplicates),
        sample_keys=_sample(duplicates),
    )


def _alignment_issues(
    label_keys: tuple[tuple[str, str], ...],
    metadata_keys: tuple[tuple[str, str], ...],
    *,
    expected_slide_id: str | None,
    value_row_count: int | None,
) -> tuple[list[IdentityIssue], tuple[int, ...]]:
    issues: list[IdentityIssue] = []
    label_duplicate = _duplicate_issue(label_keys, "labels")
    metadata_duplicate = _duplicate_issue(metadata_keys, "metadata")
    if label_duplicate is not None:
        issues.append(label_duplicate)
    if metadata_duplicate is not None:
        issues.append(metadata_duplicate)

    wrong_slide: list[tuple[str, str]] = []
    if expected_slide_id is not None:
        wrong_slide = [
            key for key in metadata_keys if key[0] != expected_slide_id
        ]
        if wrong_slide:
            issues.append(
                IdentityIssue(
                    code="wrong_slide",
                    side="metadata",
                    count=len(wrong_slide),
                    sample_keys=_sample(wrong_slide),
                )
            )

    label_row_ordinals = tuple(
        index
        for index, key in enumerate(label_keys)
        if expected_slide_id is None or key[0] == expected_slide_id
    )
    selected_label_keys = tuple(label_keys[index] for index in label_row_ordinals)

    if label_duplicate is None and metadata_duplicate is None and not wrong_slide:
        label_set = set(selected_label_keys)
        metadata_set = set(metadata_keys)
        labels_by_spot: dict[str, set[str]] = {}
        metadata_by_spot: dict[str, set[str]] = {}
        for slide_id, spot_id in label_keys:
            labels_by_spot.setdefault(spot_id, set()).add(slide_id)
        for slide_id, spot_id in metadata_keys:
            metadata_by_spot.setdefault(spot_id, set()).add(slide_id)

        label_only_all = sorted(label_set - metadata_set)
        metadata_only_all = sorted(metadata_set - label_set)
        cross_slide = {
            key
            for key in label_only_all
            if metadata_by_spot.get(key[1], set()) - {key[0]}
        }
        cross_slide.update(
            key
            for key in metadata_only_all
            if labels_by_spot.get(key[1], set()) - {key[0]}
        )
        label_only = [key for key in label_only_all if key not in cross_slide]
        metadata_only = [key for key in metadata_only_all if key not in cross_slide]
        if label_only:
            issues.append(
                IdentityIssue(
                    code="label_only",
                    side="labels",
                    count=len(label_only),
                    sample_keys=_sample(label_only),
                )
            )
        if metadata_only:
            issues.append(
                IdentityIssue(
                    code="metadata_only",
                    side="metadata",
                    count=len(metadata_only),
                    sample_keys=_sample(metadata_only),
                )
            )
        if cross_slide:
            issues.append(
                IdentityIssue(
                    code="cross_slide",
                    side="alignment",
                    count=len(cross_slide),
                    sample_keys=_sample(tuple(cross_slide)),
                )
            )

    if value_row_count is not None and value_row_count != len(metadata_keys):
        issues.append(
            IdentityIssue(
                code="cardinality_mismatch",
                side="values",
                count=abs(value_row_count - len(metadata_keys)),
            )
        )
    return issues, label_row_ordinals


def align_labels_with_metadata(
    labels: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    stage: str,
    expected_slide_id: str | None = None,
    value_row_count: int | None = None,
) -> pd.DataFrame:
    """Return complete one-to-one labels in metadata order after strict admission."""
    stage = _validate_parameter("stage", stage, stage_for_error="identity_parameters")
    if expected_slide_id is not None:
        expected_slide_id = _validate_parameter(
            "expected_slide_id",
            expected_slide_id,
            stage_for_error=stage,
        )
    if value_row_count is not None and (
        type(value_row_count) is not int or value_row_count < 0
    ):
        raise IdentityValidationError(
            stage=stage,
            issues=(
                IdentityIssue(
                    code="invalid_type",
                    side="values",
                    count=1,
                    sample_rows=(
                        (0, "value_row_count", _type_label(value_row_count)),
                    ),
                ),
            ),
        )
    if type(labels) is not pd.DataFrame or type(metadata) is not pd.DataFrame:
        bad = labels if type(labels) is not pd.DataFrame else metadata
        side = "labels" if type(labels) is not pd.DataFrame else "metadata"
        raise IdentityValidationError(
            stage=stage,
            issues=(
                IdentityIssue(
                    code="invalid_type",
                    side=side,
                    count=1,
                    sample_rows=((0, "table", _type_label(bad)),),
                ),
            ),
        )

    schema_issues = [
        *_schema_issues(labels, "labels"),
        *_schema_issues(metadata, "metadata"),
    ]
    if schema_issues:
        raise IdentityValidationError(stage=stage, issues=schema_issues)

    label_keys, label_issues = _admit_key_rows(labels, "labels")
    metadata_keys, metadata_issues = _admit_key_rows(metadata, "metadata")
    if label_issues or metadata_issues:
        raise IdentityValidationError(
            stage=stage,
            issues=(*label_issues, *metadata_issues),
        )

    issues, label_ordinals = _alignment_issues(
        label_keys,
        metadata_keys,
        expected_slide_id=expected_slide_id,
        value_row_count=value_row_count,
    )
    if issues:
        raise IdentityValidationError(stage=stage, issues=issues)

    label_rows = labels.iloc[list(label_ordinals)].copy()
    label_rows["_label_source_row"] = list(label_ordinals)
    metadata_rows = metadata.copy()
    metadata_rows["_patch_source_row"] = range(len(metadata_rows))
    try:
        aligned = metadata_rows.merge(
            label_rows,
            on=list(KEY_COLUMNS),
            how="left",
            validate="one_to_one",
            sort=False,
            indicator=True,
            suffixes=("_metadata", ""),
        )
    except Exception as exc:
        raise RuntimeError(
            f"{stage}: one-to-one merge failed after identity proof"
        ) from exc
    if not aligned["_merge"].eq("both").all():  # pragma: no cover - proof invariant
        raise RuntimeError(f"{stage}: merge lost rows after identity proof")
    return aligned.drop(columns="_merge").reset_index(drop=True)


def validate_anndata_spot_identity(
    adata: Any,
    slide_id: str,
    *,
    stage: str,
    require_slide_id: bool = False,
) -> None:
    """Admit exact AnnData compound identity before row construction."""
    stage = _validate_parameter("stage", stage, stage_for_error="identity_parameters")
    slide_id = _validate_parameter("slide_id", slide_id, stage_for_error=stage)
    if type(require_slide_id) is not bool:
        raise IdentityValidationError(
            stage=stage,
            issues=(
                IdentityIssue(
                    code="invalid_type",
                    side="parameters",
                    count=1,
                    sample_rows=(
                        (0, "require_slide_id", _type_label(require_slide_id)),
                    ),
                ),
            ),
        )
    obs_names = adata.obs_names
    invalid: list[tuple[int, str, str]] = []
    blank: list[tuple[int, str, str]] = []
    admitted: list[tuple[str, str]] = []
    for row in range(len(obs_names)):
        value = obs_names[row]
        if type(value) is not str:
            invalid.append((row, "spot_id", _type_label(value)))
        elif not value.strip():
            blank.append((row, "spot_id", "builtins.str"))
        else:
            admitted.append((slide_id, value))
    issues: list[IdentityIssue] = []
    if invalid:
        issues.append(
            IdentityIssue(
                code="invalid_type",
                side="anndata",
                count=len(invalid),
                sample_rows=tuple(invalid[:_SAMPLE_LIMIT]),
            )
        )
    if blank:
        issues.append(
            IdentityIssue(
                code="blank_value",
                side="anndata",
                count=len(blank),
                sample_rows=tuple(blank[:_SAMPLE_LIMIT]),
            )
        )
    obs = getattr(adata, "obs", None)
    has_slide_id = (
        obs is not None
        and hasattr(obs, "columns")
        and "slide_id" in tuple(obs.columns)
    )
    if require_slide_id and not has_slide_id:
        issues.append(
            IdentityIssue(code="missing_column", side="anndata", count=1)
        )
    if has_slide_id:
        persisted_invalid: list[tuple[int, str, str]] = []
        persisted_blank: list[tuple[int, str, str]] = []
        wrong_slide: list[tuple[str, str]] = []
        wrong_slide_count = 0
        persisted = obs["slide_id"]
        if len(persisted) != len(obs_names):
            issues.append(
                IdentityIssue(
                    code="cardinality_mismatch",
                    side="anndata",
                    count=abs(len(persisted) - len(obs_names)),
                )
            )
        else:
            for row in range(len(persisted)):
                value = persisted.iloc[row]
                if type(value) is not str:
                    persisted_invalid.append(
                        (row, "slide_id", _type_label(value))
                    )
                elif not value.strip():
                    persisted_blank.append((row, "slide_id", "builtins.str"))
                elif value != slide_id:
                    wrong_slide_count += 1
                    spot_id = obs_names[row]
                    if type(spot_id) is str:
                        wrong_slide.append((value, spot_id))
            if persisted_invalid:
                issues.append(
                    IdentityIssue(
                        code="invalid_type",
                        side="anndata",
                        count=len(persisted_invalid),
                        sample_rows=tuple(persisted_invalid[:_SAMPLE_LIMIT]),
                    )
                )
            if persisted_blank:
                issues.append(
                    IdentityIssue(
                        code="blank_value",
                        side="anndata",
                        count=len(persisted_blank),
                        sample_rows=tuple(persisted_blank[:_SAMPLE_LIMIT]),
                    )
                )
            if wrong_slide_count:
                issues.append(
                    IdentityIssue(
                        code="wrong_slide",
                        side="anndata",
                        count=wrong_slide_count,
                        sample_keys=_sample(wrong_slide),
                    )
                )
    if not issues:
        duplicate = _duplicate_issue(tuple(admitted), "anndata")
        if duplicate is not None:
            issues.append(duplicate)
    if issues:
        raise IdentityValidationError(stage=stage, issues=issues)


__all__ = [
    "IdentityIssue",
    "IdentityValidationError",
    "align_labels_with_metadata",
    "validate_anndata_spot_identity",
]
