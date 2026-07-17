"""Bounded artifact identity, admission, and publication primitives.

This module intentionally imports only the Python standard library.  A checksum
establishes integrity and lineage for locally produced bytes; it is not an
authenticity guarantee for pickle-bearing formats.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA_VERSION = "spatial-pharma-artifact-manifest-v1"
FINGERPRINT_ALGORITHM = "sha256"
MANIFEST_SUFFIX = ".manifest.json"
MAX_MANIFEST_BYTES = 65_536
MAX_DEPTH = 12
MAX_NODES = 2_048
MAX_MAP_ITEMS = 128
MAX_LIST_ITEMS = 256
MAX_STRING_LENGTH = 4_096
MAX_INTEGER_BITS = 256
CHECKSUM_CHUNK_SIZE = 1024 * 1024

ARTIFACT_CONTRACT_VERSIONS = {
    "root_h5ad": "root-h5ad-v1",
    "processed_slide": "processed-slide-v1",
    "label_table": "label-table-v1",
    "domain_table": "domain-table-v1",
    "patch": "patch-v1",
    "patch_index": "patch-index-v1",
    "embedding": "embedding-v1",
    "checkpoint": "checkpoint-v1",
    "report": "report-v1",
    "summary": "summary-v1",
    "cohort_manifest": "cohort-manifest-wrapper-v1",
    "preprocessing_manifest": "preprocessing-manifest-wrapper-v1",
}

_CONFIG_PROJECTIONS = {
    "root_h5ad": ("preprocessing",),
    "processed_slide": ("preprocessing",),
    "label_table": ("labels",),
    "domain_table": ("labels",),
    "patch": ("patches",),
    "patch_index": ("patches",),
    "embedding": ("foundation",),
    "checkpoint": ("training",),
    "report": ("evaluation",),
    "summary": ("evaluation",),
    "cohort_manifest": ("preprocessing",),
    "preprocessing_manifest": ("preprocessing",),
}

_REGENERATION_GUIDANCE = "Regenerate this artifact from its trusted source."
_HEX = frozenset("0123456789abcdef")


def _bounded_basename(value: object) -> str:
    if type(value) is not str:
        return "artifact"
    if not value or len(value) > 240 or "\n" in value or "\r" in value:
        return "artifact"
    if value in {".", ".."} or Path(value).name != value:
        return "artifact"
    return value


class ArtifactValidationError(ValueError):
    """Structured, bounded failure at an artifact trust boundary."""

    __slots__ = ("artifact_kind", "basename", "reason_code", "guidance")

    def __init__(
        self,
        reason_code: str,
        *,
        artifact_kind: str = "unknown",
        basename: str = "artifact",
    ) -> None:
        self.artifact_kind = (
            artifact_kind if artifact_kind in ARTIFACT_CONTRACT_VERSIONS else "unknown"
        )
        self.basename = _bounded_basename(basename)
        self.reason_code = reason_code
        self.guidance = _REGENERATION_GUIDANCE
        super().__init__(
            f"Artifact {self.basename} rejected ({reason_code}). {self.guidance}"
        )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _fresh_tree(canonical_json: str) -> dict[str, Any]:
    value = json.loads(canonical_json)
    if type(value) is not dict:
        raise TypeError("canonical artifact state must be an object")
    return value


@dataclass(frozen=True, slots=True)
class ArtifactFingerprint:
    algorithm: str
    digest: str
    canonical_inputs_json: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "digest": self.digest,
            "inputs": _fresh_tree(self.canonical_inputs_json),
        }


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    canonical_json: str
    artifact_kind: str
    contract_version: str
    fingerprint: ArtifactFingerprint
    payload_sha256: str
    payload_byte_count: int
    payload_schema_json: str
    payload_filename: str
    payload_format: str

    def to_dict(self) -> dict[str, Any]:
        return _fresh_tree(self.canonical_json)


@dataclass(frozen=True, slots=True)
class ArtifactAdmission:
    payload_path: Path
    manifest: ArtifactManifest
    value: object


@dataclass(frozen=True, slots=True)
class ArtifactReuseStatus:
    reusable: bool
    reason_code: str
    admission: ArtifactAdmission | None = None


def _error(reason: str, *, kind: str = "unknown", basename: str = "artifact") -> ArtifactValidationError:
    return ArtifactValidationError(reason, artifact_kind=kind, basename=basename)


def _admit_tree(
    value: object,
    *,
    depth: int = 0,
    counter: list[int] | None = None,
    kind: str = "unknown",
    basename: str = "artifact",
) -> object:
    if counter is None:
        counter = [0]
    counter[0] += 1
    if counter[0] > MAX_NODES or depth > MAX_DEPTH:
        raise _error("malformed_manifest", kind=kind, basename=basename)
    value_type = type(value)
    if value is None or value_type is bool:
        return value
    if value_type is str:
        if len(value) > MAX_STRING_LENGTH:
            raise _error("malformed_manifest", kind=kind, basename=basename)
        return value
    if value_type is int:
        if value.bit_length() > MAX_INTEGER_BITS:
            raise _error("malformed_manifest", kind=kind, basename=basename)
        return value
    if value_type is float:
        if not math.isfinite(value):
            raise _error("malformed_manifest", kind=kind, basename=basename)
        return value
    if value_type is list:
        if len(value) > MAX_LIST_ITEMS:
            raise _error("malformed_manifest", kind=kind, basename=basename)
        return [
            _admit_tree(
                item,
                depth=depth + 1,
                counter=counter,
                kind=kind,
                basename=basename,
            )
            for item in value
        ]
    if value_type is dict:
        if len(value) > MAX_MAP_ITEMS:
            raise _error("malformed_manifest", kind=kind, basename=basename)
        keys = list(value.keys())
        if any(type(key) is not str or len(key) > MAX_STRING_LENGTH for key in keys):
            raise _error("malformed_manifest", kind=kind, basename=basename)
        return {
            key: _admit_tree(
                value[key],
                depth=depth + 1,
                counter=counter,
                kind=kind,
                basename=basename,
            )
            for key in sorted(keys)
        }
    raise _error("malformed_manifest", kind=kind, basename=basename)


def _duplicate_rejecting_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def _is_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(char in _HEX for char in value)


def _require_keys(value: object, keys: frozenset[str], *, kind: str, basename: str) -> dict[str, Any]:
    if type(value) is not dict or frozenset(value) != keys:
        raise _error("malformed_manifest", kind=kind, basename=basename)
    return value


def _manifest_from_tree(tree: object, *, expected_basename: str | None = None) -> ArtifactManifest:
    root = _require_keys(
        tree,
        frozenset({"schema_version", "artifact_kind", "contract_version", "complete", "fingerprint", "payload"}),
        kind="unknown",
        basename=expected_basename or "artifact",
    )
    kind = root["artifact_kind"] if type(root["artifact_kind"]) is str else "unknown"
    basename = expected_basename or "artifact"
    if root["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise _error("unsupported_manifest", kind=kind, basename=basename)
    if kind not in ARTIFACT_CONTRACT_VERSIONS:
        raise _error("unsupported_manifest", basename=basename)
    if root["complete"] is not True:
        raise _error("incomplete_manifest", kind=kind, basename=basename)
    if type(root["contract_version"]) is not str:
        raise _error("unsupported_manifest", kind=kind, basename=basename)
    fingerprint = _require_keys(
        root["fingerprint"], frozenset({"algorithm", "digest", "inputs"}), kind=kind, basename=basename
    )
    payload = _require_keys(
        root["payload"], frozenset({"filename", "format", "byte_count", "sha256", "schema"}), kind=kind, basename=basename
    )
    filename = payload["filename"]
    if type(filename) is not str or _bounded_basename(filename) != filename:
        raise _error("malformed_manifest", kind=kind, basename=basename)
    basename = expected_basename or filename
    if expected_basename is not None and filename != expected_basename:
        raise _error("stale_fingerprint", kind=kind, basename=basename)
    if fingerprint["algorithm"] != FINGERPRINT_ALGORITHM or not _is_sha256(fingerprint["digest"]):
        raise _error("unsupported_manifest", kind=kind, basename=basename)
    if type(payload["format"]) is not str or not payload["format"]:
        raise _error("malformed_manifest", kind=kind, basename=basename)
    if type(payload["byte_count"]) is not int or payload["byte_count"] < 0:
        raise _error("malformed_manifest", kind=kind, basename=basename)
    if not _is_sha256(payload["sha256"]):
        raise _error("malformed_manifest", kind=kind, basename=basename)
    inputs_json = _canonical_json(fingerprint["inputs"])
    observed_digest = hashlib.sha256(inputs_json.encode("utf-8")).hexdigest()
    if not hmac.compare_digest(observed_digest, fingerprint["digest"]):
        raise _error("stale_fingerprint", kind=kind, basename=basename)
    fingerprint_record = ArtifactFingerprint(
        algorithm=FINGERPRINT_ALGORITHM,
        digest=fingerprint["digest"],
        canonical_inputs_json=inputs_json,
    )
    canonical = _canonical_json(root)
    return ArtifactManifest(
        canonical_json=canonical,
        artifact_kind=kind,
        contract_version=root["contract_version"],
        fingerprint=fingerprint_record,
        payload_sha256=payload["sha256"],
        payload_byte_count=payload["byte_count"],
        payload_schema_json=_canonical_json(payload["schema"]),
        payload_filename=filename,
        payload_format=payload["format"],
    )


def parse_manifest_bytes(
    raw: bytes,
    *,
    expected_basename: str | None = None,
    parsed_candidate: object | None = None,
) -> ArtifactManifest:
    """Parse and exactly admit bounded sidecar bytes or a test candidate."""
    basename = expected_basename or "artifact"
    try:
        if parsed_candidate is None:
            if type(raw) is not bytes or not raw or len(raw) > MAX_MANIFEST_BYTES:
                raise ValueError("invalid bytes")
            parsed = json.loads(
                raw.decode("utf-8"), object_pairs_hook=_duplicate_rejecting_object
            )
        else:
            parsed = parsed_candidate
        admitted = _admit_tree(parsed, basename=basename)
    except ArtifactValidationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
        raise _error("malformed_manifest", basename=basename) from None
    return _manifest_from_tree(admitted, expected_basename=expected_basename)


def _admit_projection_inputs(value: object, kind: str) -> dict[str, Any]:
    try:
        admitted = _admit_tree(value, kind=kind)
    except ArtifactValidationError as exc:
        raise ArtifactValidationError(
            "invalid_fingerprint_inputs", artifact_kind=kind
        ) from exc
    root = _require_keys(
        admitted,
        frozenset({"configuration", "source", "upstream", "identity"}),
        kind=kind,
        basename="artifact",
    )
    for key in root:
        if type(root[key]) is not dict:
            raise _error("invalid_fingerprint_inputs", kind=kind)
    return root


def build_fingerprint(
    artifact_kind: str,
    inputs: object,
    *,
    contract_version: str | None = None,
) -> ArtifactFingerprint:
    """Build a deterministic fingerprint from one explicit kind projection."""
    if type(artifact_kind) is not str or artifact_kind not in _CONFIG_PROJECTIONS:
        raise _error("unsupported_artifact_kind")
    admitted = _admit_projection_inputs(inputs, artifact_kind)
    configuration = admitted["configuration"]
    projected_configuration = {
        key: configuration[key]
        for key in _CONFIG_PROJECTIONS[artifact_kind]
        if key in configuration
    }
    version = contract_version or ARTIFACT_CONTRACT_VERSIONS[artifact_kind]
    if type(version) is not str or not version or len(version) > 128:
        raise _error("invalid_fingerprint_inputs", kind=artifact_kind)
    projection = {
        "artifact_kind": artifact_kind,
        "configuration": projected_configuration,
        "contract_version": version,
        "identity": admitted["identity"],
        "source": admitted["source"],
        "upstream": admitted["upstream"],
    }
    canonical = _canonical_json(projection)
    return ArtifactFingerprint(
        algorithm=FINGERPRINT_ALGORITHM,
        digest=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        canonical_inputs_json=canonical,
    )


def manifest_path(payload_path: str | os.PathLike[str]) -> Path:
    path = Path(payload_path)
    return path.with_name(path.name + MANIFEST_SUFFIX)


__all__ = [
    "ARTIFACT_CONTRACT_VERSIONS",
    "ArtifactAdmission",
    "ArtifactFingerprint",
    "ArtifactManifest",
    "ArtifactReuseStatus",
    "ArtifactValidationError",
    "build_fingerprint",
    "manifest_path",
    "parse_manifest_bytes",
]
