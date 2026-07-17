"""Pure evidence for the shared durable-artifact contract."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

from utils.artifacts import (
    ARTIFACT_CONTRACT_VERSIONS,
    MANIFEST_SCHEMA_VERSION,
    ArtifactValidationError,
    admit_artifact,
    artifact_reuse_status,
    build_fingerprint,
    manifest_path,
    parse_manifest_bytes,
)

pytestmark = pytest.mark.offline


KINDS = tuple(ARTIFACT_CONTRACT_VERSIONS)


def _inputs() -> dict[str, object]:
    return {
        "configuration": {
            "preprocessing": {"n_top_genes": 2000},
            "labels": {"method": "marker_score"},
            "patches": {"size": 224},
            "foundation": {"model": "phikon"},
            "training": {"epochs": 5},
            "evaluation": {"metric": "auroc"},
            "reporting": {"style": "paper"},
            "plotting": {"palette": "magma"},
            "output_dir": "/checkout/output",
            "unrelated": {"value": 1},
        },
        "source": {"semantic_id": "slide-a", "content_sha256": "a" * 64},
        "upstream": {"processed_slide": "b" * 64, "labels": "c" * 64},
        "identity": {"slide_id": "slide-a", "model": "phikon", "fold": "2"},
    }


@pytest.mark.parametrize("kind", KINDS)
def test_fingerprint_projection_is_canonical_fresh_and_path_invariant(kind: str) -> None:
    first_inputs = _inputs()
    second_inputs = _inputs()
    second_inputs["configuration"] = dict(
        reversed(list(second_inputs["configuration"].items()))  # type: ignore[union-attr]
    )
    second_inputs["configuration"]["output_dir"] = "/elsewhere"  # type: ignore[index]
    second_inputs["configuration"]["plotting"] = {"palette": "viridis"}  # type: ignore[index]

    first = build_fingerprint(kind, first_inputs)
    second = build_fingerprint(kind, second_inputs)
    assert first.digest == second.digest
    assert first.canonical_inputs_json == second.canonical_inputs_json

    view = first.to_dict()
    view["inputs"]["contract_version"] = "mutated"
    assert first.to_dict()["inputs"]["contract_version"] != "mutated"


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dimension", ["configuration", "source", "upstream", "identity"])
def test_every_projection_invalidates_relevant_inputs(kind: str, dimension: str) -> None:
    baseline = _inputs()
    changed = _inputs()
    if dimension == "configuration":
        section = {
            "root_h5ad": "preprocessing",
            "processed_slide": "preprocessing",
            "label_table": "labels",
            "domain_table": "labels",
            "patch": "patches",
            "patch_index": "patches",
            "embedding": "foundation",
            "checkpoint": "training",
            "report": "evaluation",
            "summary": "evaluation",
            "cohort_manifest": "preprocessing",
            "preprocessing_manifest": "preprocessing",
        }[kind]
        changed["configuration"][section]["changed"] = True  # type: ignore[index]
    else:
        changed[dimension]["changed"] = "d" * 64  # type: ignore[index]
    assert build_fingerprint(kind, baseline).digest != build_fingerprint(
        kind, changed
    ).digest


def test_contract_version_changes_fingerprint() -> None:
    original = ARTIFACT_CONTRACT_VERSIONS["patch"]
    first = build_fingerprint("patch", _inputs())
    second = build_fingerprint("patch", _inputs(), contract_version=original + ".next")
    assert first.digest != second.digest


class HostileDict(dict):
    def items(self):
        raise AssertionError("hostile items hook executed")

    def __repr__(self):
        raise AssertionError("hostile repr hook executed")


@pytest.mark.parametrize(
    ("candidate", "reason"),
    [
        (HostileDict(), "malformed_manifest"),
        ({"x": object()}, "malformed_manifest"),
        ({"x": [0] * 300}, "malformed_manifest"),
        ({"x": "x" * 5000}, "malformed_manifest"),
        ({"x": 1 << 5000}, "malformed_manifest"),
    ],
)
def test_hostile_manifest_candidates_fail_without_hooks(candidate: object, reason: str) -> None:
    with pytest.raises(ArtifactValidationError) as caught:
        parse_manifest_bytes(b"{}", parsed_candidate=candidate)
    assert caught.value.reason_code == reason
    assert "regenerate" in str(caught.value).lower()
    assert "\n" not in str(caught.value)


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"\xff",
        b'{"schema_version":1',
        b'{"x":1,"x":2}',
        b"[1,2,3]",
        b"{" + b'\"x\":\"' + (b"x" * 70_000) + b'\"}',
    ],
)
def test_malformed_duplicate_and_oversized_manifest_bytes_are_bounded(raw: bytes) -> None:
    with pytest.raises(ArtifactValidationError) as caught:
        parse_manifest_bytes(raw)
    assert caught.value.reason_code == "malformed_manifest"
    assert len(str(caught.value)) < 300


def test_utils_artifacts_is_import_light() -> None:
    forbidden = {"numpy", "pandas", "torch", "anndata", "scanpy", "torchvision", "timm", "transformers"}
    imported = {name.split(".", 1)[0] for name in sys.modules}
    assert forbidden.isdisjoint(imported - {"numpy", "pandas"})


def test_duplicate_json_is_rejected_before_canonicalization() -> None:
    with pytest.raises(ArtifactValidationError, match="Regenerate"):
        parse_manifest_bytes(json.dumps({"safe": True}).replace("}", ',"safe":false}').encode())


def _write_pair(path: Path, payload: bytes = b"trusted-payload"):
    fingerprint = build_fingerprint("patch", _inputs())
    path.write_bytes(payload)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_kind": "patch",
        "contract_version": ARTIFACT_CONTRACT_VERSIONS["patch"],
        "complete": True,
        "fingerprint": fingerprint.to_dict(),
        "payload": {
            "filename": path.name,
            "format": "bytes",
            "byte_count": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "schema": {"encoding": "bytes"},
        },
    }
    manifest_path(path).write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    return fingerprint


def test_admission_checks_integrity_before_callback(tmp_path: Path) -> None:
    path = tmp_path / "patch.npz"
    fingerprint = _write_pair(path)
    calls: list[str] = []
    admission = admit_artifact(
        path,
        expected_kind="patch",
        expected_contract_version=ARTIFACT_CONTRACT_VERSIONS["patch"],
        expected_fingerprint=fingerprint,
        reader=lambda candidate: calls.append(candidate.read_bytes().decode()) or "decoded",
    )
    assert admission.value == "decoded"
    assert calls == ["trusted-payload"]
    assert admission.manifest.to_dict()["payload"]["filename"] == path.name


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda path: path.write_bytes(b"truncated"), "byte_count_mismatch"),
        (lambda path: manifest_path(path).write_text("", encoding="utf-8"), "malformed_manifest"),
        (lambda path: manifest_path(path).unlink(), "legacy_artifact"),
    ],
)
def test_incomplete_checksum_and_legacy_reject_before_callback(
    tmp_path: Path, mutation, reason: str
) -> None:
    path = tmp_path / "patch.npz"
    fingerprint = _write_pair(path)
    mutation(path)

    def forbidden(_path: Path):
        raise AssertionError("decoder ran before admission")

    with pytest.raises(ArtifactValidationError) as caught:
        admit_artifact(
            path,
            expected_kind="patch",
            expected_contract_version=ARTIFACT_CONTRACT_VERSIONS["patch"],
            expected_fingerprint=fingerprint,
            reader=forbidden,
        )
    assert caught.value.reason_code == reason


def test_stale_fingerprint_and_callback_schema_failure_are_typed(tmp_path: Path) -> None:
    path = tmp_path / "patch.npz"
    expected = _write_pair(path)
    stale_inputs = _inputs()
    stale_inputs["configuration"]["patches"]["size"] = 512  # type: ignore[index]
    stale = build_fingerprint("patch", stale_inputs)
    with pytest.raises(ArtifactValidationError) as caught:
        admit_artifact(
            path,
            expected_kind="patch",
            expected_contract_version=ARTIFACT_CONTRACT_VERSIONS["patch"],
            expected_fingerprint=stale,
            reader=lambda _path: pytest.fail("stale payload decoded"),
        )
    assert caught.value.reason_code == "stale_fingerprint"

    def invalid_schema(_path: Path):
        raise ValueError("attacker controlled parser prose")

    with pytest.raises(ArtifactValidationError) as caught:
        admit_artifact(
            path,
            expected_kind="patch",
            expected_contract_version=ARTIFACT_CONTRACT_VERSIONS["patch"],
            expected_fingerprint=expected,
            reader=invalid_schema,
        )
    assert caught.value.reason_code == "reader_validation_failed"
    assert "attacker" not in str(caught.value)


def test_symlink_payload_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "real.npz"
    fingerprint = _write_pair(target)
    link = tmp_path / "link.npz"
    link.symlink_to(target)
    manifest_path(link).write_bytes(
        manifest_path(target).read_bytes().replace(target.name.encode(), link.name.encode())
    )
    with pytest.raises(ArtifactValidationError) as caught:
        admit_artifact(
            link,
            expected_kind="patch",
            expected_contract_version=ARTIFACT_CONTRACT_VERSIONS["patch"],
            expected_fingerprint=fingerprint,
            reader=lambda _path: pytest.fail("symlink decoded"),
        )
    assert caught.value.reason_code == "invalid_payload_file"


def test_reader_replacement_race_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "patch.npz"
    fingerprint = _write_pair(path)

    def replacing_reader(candidate: Path):
        replacement = candidate.with_name("replacement.npz")
        replacement.write_bytes(candidate.read_bytes())
        os.replace(replacement, candidate)
        return "must-not-return"

    with pytest.raises(ArtifactValidationError) as caught:
        admit_artifact(
            path,
            expected_kind="patch",
            expected_contract_version=ARTIFACT_CONTRACT_VERSIONS["patch"],
            expected_fingerprint=fingerprint,
            reader=replacing_reader,
        )
    assert caught.value.reason_code == "unstable_payload"


def test_reuse_status_is_typed_and_missing_parent_has_no_side_effect(tmp_path: Path) -> None:
    path = tmp_path / "missing" / "patch.npz"
    fingerprint = build_fingerprint("patch", _inputs())
    status = artifact_reuse_status(
        path,
        expected_kind="patch",
        expected_contract_version=ARTIFACT_CONTRACT_VERSIONS["patch"],
        expected_fingerprint=fingerprint,
        reader=lambda _path: None,
    )
    assert not status.reusable
    assert status.reason_code == "missing_payload"
    assert not path.parent.exists()
