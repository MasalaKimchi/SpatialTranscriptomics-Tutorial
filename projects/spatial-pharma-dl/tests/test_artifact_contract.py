"""Pure evidence for the shared durable-artifact contract."""

from __future__ import annotations

import json
import sys

import pytest

from utils.artifacts import (
    ARTIFACT_CONTRACT_VERSIONS,
    ArtifactValidationError,
    build_fingerprint,
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
    with pytest.raises(ArtifactValidationError, match="regenerate"):
        parse_manifest_bytes(json.dumps({"safe": True}).replace("}", ',"safe":false}').encode())
