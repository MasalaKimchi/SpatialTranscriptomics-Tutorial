"""Pure evidence for the shared durable-artifact contract."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
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
    publish_artifact,
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
        if kind == "embedding":
            changed["configuration"][section]["model"] = "kaiko_vits16"  # type: ignore[index]
        else:
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


def test_manifest_contract_must_match_fingerprinted_contract(tmp_path: Path) -> None:
    path = tmp_path / "mismatch.npz"
    _write_pair(path)
    candidate = json.loads(manifest_path(path).read_text(encoding="utf-8"))
    candidate["contract_version"] = "patch-v2"
    with pytest.raises(ArtifactValidationError) as caught:
        parse_manifest_bytes(
            json.dumps(candidate, sort_keys=True, separators=(",", ":")).encode(),
            expected_basename=path.name,
        )
    assert caught.value.reason_code == "stale_fingerprint"


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


@pytest.mark.parametrize(
    "raw",
    [
        b"[" * 2000 + b"0" + b"]" * 2000,
        b'{"x":' * 900 + b"0" + b"}" * 900,
    ],
)
def test_deep_bounded_manifest_bytes_are_typed(raw: bytes) -> None:
    assert len(raw) < 65_536
    with pytest.raises(ArtifactValidationError) as caught:
        parse_manifest_bytes(raw)
    assert caught.value.reason_code == "malformed_manifest"


def test_utils_artifacts_is_import_light() -> None:
    root = Path(__file__).resolve().parents[3]
    code = """
import sys
import utils.artifacts
forbidden = {'numpy', 'pandas', 'torch', 'anndata', 'scanpy', 'torchvision', 'timm', 'transformers'}
imported = {name.split('.', 1)[0] for name in sys.modules}
raise SystemExit(1 if forbidden & imported else 0)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


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


def test_public_path_aba_cannot_change_admitted_decoder_bytes(tmp_path: Path) -> None:
    path = tmp_path / "patch.npz"
    fingerprint = _write_pair(path, b"good")

    def aba_reader(admitted_snapshot: Path):
        held = path.with_name("held.npz")
        path.rename(held)
        path.write_bytes(b"evil")
        path.unlink()
        held.rename(path)
        return admitted_snapshot.read_bytes()

    admission = admit_artifact(
        path,
        expected_kind="patch",
        expected_contract_version=ARTIFACT_CONTRACT_VERSIONS["patch"],
        expected_fingerprint=fingerprint,
        reader=aba_reader,
    )
    assert admission.value == b"good"
    assert path.read_bytes() == b"good"


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


def _publish(
    path: Path,
    payload: bytes,
    *,
    operation_log: list[str] | None = None,
    fault_at: str | None = None,
):
    fingerprint = build_fingerprint("patch", _inputs())

    def hook(operation: str) -> None:
        if operation_log is not None:
            operation_log.append(operation)
        if operation == fault_at:
            raise OSError("injected publication fault")

    manifest = publish_artifact(
        path,
        artifact_kind="patch",
        contract_version=ARTIFACT_CONTRACT_VERSIONS["patch"],
        fingerprint=fingerprint,
        payload_format="bytes",
        payload_schema={"encoding": "bytes"},
        write_payload=lambda temporary: temporary.write_bytes(payload),
        reader=lambda temporary: temporary.read_bytes(),
        observed_schema=lambda _decoded: {"encoding": "bytes"},
        _operation_hook=hook,
    )
    return fingerprint, manifest


def test_atomic_publication_validates_before_payload_first_manifest_last(tmp_path: Path) -> None:
    path = tmp_path / "result.npz"
    operations: list[str] = []
    fingerprint, manifest = _publish(path, b"new-generation", operation_log=operations)
    assert operations == [
        "write_payload",
        "fsync_payload",
        "write_manifest",
        "fsync_manifest",
        "validate",
        "replace_payload",
        "fsync_directory_first",
        "replace_manifest",
        "fsync_directory_final",
    ]
    assert operations.index("validate") < operations.index("replace_payload")
    assert path.read_bytes() == b"new-generation"
    assert manifest_path(path).read_bytes() == manifest.canonical_json.encode("utf-8")
    admission = admit_artifact(
        path,
        expected_kind="patch",
        expected_contract_version=ARTIFACT_CONTRACT_VERSIONS["patch"],
        expected_fingerprint=fingerprint,
        reader=lambda candidate: candidate.read_bytes(),
    )
    assert admission.value == b"new-generation"


def test_publication_rejects_observed_schema_drift_before_replace(tmp_path: Path) -> None:
    path = tmp_path / "result.npz"
    _publish(path, b"old-generation")
    old_payload = path.read_bytes()
    old_manifest = manifest_path(path).read_bytes()

    with pytest.raises(ArtifactValidationError, match="publication_failed"):
        publish_artifact(
            path,
            artifact_kind="patch",
            contract_version=ARTIFACT_CONTRACT_VERSIONS["patch"],
            fingerprint=build_fingerprint("patch", _inputs()),
            payload_format="bytes",
            payload_schema={"shape": [1]},
            write_payload=lambda temporary: temporary.write_bytes(b"new-generation"),
            reader=lambda temporary: temporary.read_bytes(),
            observed_schema=lambda _decoded: {"shape": [2]},
        )
    assert path.read_bytes() == old_payload
    assert manifest_path(path).read_bytes() == old_manifest


def test_embedding_projection_omits_execution_controls() -> None:
    baseline = _inputs()
    baseline["configuration"]["foundation"] = {  # type: ignore[index]
        "model": "phikon",
        "enabled": True,
        "cache": True,
        "device": "cpu",
        "batch_size": 8,
    }
    expected = build_fingerprint("embedding", baseline)
    for leaf, value in (
        ("enabled", False),
        ("cache", False),
        ("device", "cuda"),
        ("batch_size", 128),
    ):
        changed = json.loads(json.dumps(baseline))
        changed["configuration"]["foundation"][leaf] = value
        assert build_fingerprint("embedding", changed).digest == expected.digest

    changed_model = json.loads(json.dumps(baseline))
    changed_model["configuration"]["foundation"]["model"] = "kaiko_vits16"
    assert build_fingerprint("embedding", changed_model).digest != expected.digest


PUBLICATION_FAULTS = (
    "write_payload",
    "fsync_payload",
    "write_manifest",
    "fsync_manifest",
    "validate",
    "replace_payload",
    "fsync_directory_first",
    "replace_manifest",
    "fsync_directory_final",
)


@pytest.mark.parametrize("fault_at", PUBLICATION_FAULTS)
@pytest.mark.parametrize("replacement", [False, True])
def test_atomic_fault_states_are_never_falsely_reusable(
    tmp_path: Path, fault_at: str, replacement: bool
) -> None:
    path = tmp_path / "result.npz"
    fingerprint = build_fingerprint("patch", _inputs())
    if replacement:
        _publish(path, b"old-generation")

    with pytest.raises(ArtifactValidationError) as caught:
        _publish(path, b"new-generation", fault_at=fault_at)
    assert caught.value.reason_code == "publication_failed"

    status = artifact_reuse_status(
        path,
        expected_kind="patch",
        expected_contract_version=ARTIFACT_CONTRACT_VERSIONS["patch"],
        expected_fingerprint=fingerprint,
        reader=lambda candidate: candidate.read_bytes(),
    )
    if fault_at == "fsync_directory_final":
        assert status.reusable
        assert status.admission is not None
        assert status.admission.value == b"new-generation"
    elif fault_at in {"fsync_directory_first", "replace_manifest"}:
        assert not status.reusable
        assert status.reason_code in {"legacy_artifact", "byte_count_mismatch", "checksum_mismatch"}
    elif replacement:
        assert status.reusable
        assert status.admission is not None
        assert status.admission.value == b"old-generation"
    else:
        assert not status.reusable
        assert status.reason_code == "missing_payload"

    assert not list(tmp_path.glob(".result.npz.*"))


def test_publication_temp_paths_are_unique_same_directory_and_suffix_preserving(
    tmp_path: Path,
) -> None:
    path = tmp_path / "result.npz"
    seen: list[Path] = []

    def writer(temporary: Path) -> None:
        seen.append(temporary)
        assert temporary.parent == path.parent
        assert temporary.suffix == ".npz"
        temporary.write_bytes(b"generation-one")

    fingerprint = build_fingerprint("patch", _inputs())
    for payload in (b"generation-one", b"generation-two"):
        publish_artifact(
            path,
            artifact_kind="patch",
            contract_version=ARTIFACT_CONTRACT_VERSIONS["patch"],
            fingerprint=fingerprint,
            payload_format="bytes",
            payload_schema={"encoding": "bytes"},
            write_payload=(writer if payload == b"generation-one" else lambda temporary: (seen.append(temporary), temporary.write_bytes(payload))),
            reader=lambda temporary: temporary.read_bytes(),
            observed_schema=lambda _decoded: {"encoding": "bytes"},
        )
    assert len({candidate.name for candidate in seen}) == 2
