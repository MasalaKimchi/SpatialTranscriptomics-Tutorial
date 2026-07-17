"""Offline evidence for exact compound identity and complete alignment."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from src.identity import (
    IdentityValidationError,
    align_labels_with_metadata,
    validate_anndata_spot_identity,
)
from src import labels as label_module
from src import patches as patch_module
from src import foundation as foundation_module
from src import train as train_module
from src.foundation import FOUNDATION_MODELS
from utils.artifacts import (
    ARTIFACT_CONTRACT_VERSIONS,
    ArtifactValidationError,
    publish_artifact,
)

pytestmark = pytest.mark.offline


def _align(case, **kwargs):
    return align_labels_with_metadata(
        case["labels"],
        case["patch_index"],
        stage="test_alignment",
        **kwargs,
    )


@pytest.mark.parametrize(
    ("case_name", "code"),
    [
        ("missing_label_slide", "missing_column"),
        ("missing_metadata_spot", "missing_column"),
        ("null", "invalid_type"),
        ("blank", "blank_value"),
        ("wrong_type", "invalid_type"),
        ("duplicate", "duplicate_key"),
        ("duplicate_metadata", "duplicate_key"),
        ("reserved_label", "reserved_column"),
        ("reserved_metadata", "reserved_column"),
    ],
)
def test_key_defects_are_structured(key_adversary_factory, case_name, code):
    with pytest.raises(IdentityValidationError) as caught:
        _align(key_adversary_factory()[case_name])

    assert code in {issue.code for issue in caught.value.issues}
    assert isinstance(caught.value, ValueError)
    assert all(issue.count >= 1 for issue in caught.value.issues)


@pytest.mark.parametrize("case_name", ["hostile_label", "hostile_metadata"])
def test_hostile_key_subclasses_execute_no_hooks(key_adversary_factory, case_name):
    with pytest.raises(IdentityValidationError, match="invalid_type"):
        _align(key_adversary_factory()[case_name])


def test_value_row_mismatch_is_rejected(key_adversary_factory):
    case = key_adversary_factory()["value_row_mismatch"]
    with pytest.raises(IdentityValidationError) as caught:
        _align(case, value_row_count=case["value_row_count"])

    issue = caught.value.issues[0]
    assert issue.code == "cardinality_mismatch"
    assert issue.side == "values"


def test_wrong_slide_metadata_is_rejected(key_adversary_factory):
    case = key_adversary_factory()["wrong_slide_metadata"]
    with pytest.raises(IdentityValidationError) as caught:
        _align(case, expected_slide_id="slide_a")

    assert "wrong_slide" in {issue.code for issue in caught.value.issues}


def test_cross_slide_mismatch_is_distinguished(key_adversary_factory):
    case = key_adversary_factory()["cross_slide"]
    with pytest.raises(IdentityValidationError) as caught:
        _align(case)

    assert "cross_slide" in {issue.code for issue in caught.value.issues}


def test_diagnostics_are_deterministic_and_bounded():
    labels = pd.DataFrame(
        {
            "slide_id": ["slide_a"] * 7,
            "spot_id": [f"label_{index}" for index in range(7)],
        }
    )
    metadata = pd.DataFrame(
        {
            "slide_id": ["slide_a"] * 7,
            "spot_id": [f"meta_{index}" for index in reversed(range(7))],
        }
    )

    messages = []
    issues = []
    for frame in (metadata, metadata.iloc[::-1].reset_index(drop=True)):
        with pytest.raises(IdentityValidationError) as caught:
            align_labels_with_metadata(labels, frame, stage="stable")
        messages.append(str(caught.value))
        issues.append(caught.value.issues)

    assert messages[0] == messages[1]
    assert issues[0] == issues[1]
    assert all(len(issue.sample_keys) <= 5 for issue in issues[0])


def test_shuffled_complete_alignment_preserves_metadata_order_and_pairing(
    key_adversary_factory,
):
    case = key_adversary_factory()["shuffled_complete"]
    aligned = _align(case)

    assert aligned["spot_id"].tolist() == case["patch_index"]["spot_id"].tolist()
    expected = case["labels"].set_index(["slide_id", "spot_id"])
    for row in aligned.itertuples(index=False):
        source = expected.loc[(row.slide_id, row.spot_id)]
        assert row.tme_class_id == source["tme_class_id"]
        assert row.module_signal == source["module_signal"]
    assert aligned["_patch_source_row"].tolist() == list(range(len(aligned)))
    assert sorted(aligned["_label_source_row"].tolist()) == list(range(len(aligned)))


def test_repeated_barcode_across_slides_remains_compound_scoped():
    labels = pd.DataFrame(
        {
            "slide_id": ["slide_a", "slide_b"],
            "spot_id": ["shared", "shared"],
            "target": [1, 2],
        }
    )
    metadata = labels[["slide_id", "spot_id"]].iloc[::-1].reset_index(drop=True)

    aligned = align_labels_with_metadata(labels, metadata, stage="compound")

    assert aligned["target"].tolist() == [2, 1]


def test_anndata_identity_rejects_invalid_names_before_row_construction():
    adata = SimpleNamespace(obs_names=pd.Index(["spot_a", None], dtype=object))

    with pytest.raises(IdentityValidationError, match="invalid_type"):
        validate_anndata_spot_identity(adata, "slide_a", stage="producer")


@pytest.mark.parametrize(
    "persisted_slide_ids",
    [["slide_b", "slide_b"], ["slide_a", "slide_b"], ["slide_a", "   "]],
)
def test_anndata_identity_rejects_invalid_persisted_slide_identity(
    persisted_slide_ids,
):
    adata = SimpleNamespace(
        obs_names=pd.Index(["spot_a", "spot_b"], dtype=object),
        obs=pd.DataFrame({"slide_id": persisted_slide_ids}),
    )

    with pytest.raises(IdentityValidationError) as caught:
        validate_anndata_spot_identity(
            adata,
            "slide_a",
            stage="persisted_producer",
            require_slide_id=True,
        )

    assert {issue.code for issue in caught.value.issues} & {
        "wrong_slide",
        "blank_value",
    }


def test_raw_anndata_without_persisted_slide_identity_remains_compatible():
    adata = SimpleNamespace(
        obs_names=pd.Index(["spot_a", "spot_b"], dtype=object),
        obs=pd.DataFrame(index=["spot_a", "spot_b"]),
    )

    validate_anndata_spot_identity(adata, "slide_a", stage="raw_source")

    with pytest.raises(IdentityValidationError, match="missing_column"):
        validate_anndata_spot_identity(
            adata,
            "slide_a",
            stage="persisted_source",
            require_slide_id=True,
        )


def test_hostile_metaclass_type_evidence_executes_no_hooks():
    calls: list[str] = []

    class HostileMeta(type):
        def __getattribute__(cls, name):
            if name in {"__module__", "__name__", "__qualname__"}:
                calls.append(name)
                raise AssertionError("metaclass hook executed")
            return super().__getattribute__(name)

    class HostileValue(metaclass=HostileMeta):
        def __repr__(self):
            calls.append("repr")
            raise AssertionError("repr hook executed")

    hostile = HostileValue()
    for hostile_side in ("labels", "metadata"):
        tables = {
            "labels": pd.DataFrame(
                {"slide_id": ["slide_a"], "spot_id": ["p0"]}
            ),
            "metadata": pd.DataFrame(
                {"slide_id": ["slide_a"], "spot_id": ["p0"]}
            ),
        }
        tables[hostile_side]["spot_id"] = pd.Series([hostile], dtype=object)
        with pytest.raises(IdentityValidationError, match="invalid_type"):
            align_labels_with_metadata(
                tables["labels"],
                tables["metadata"],
                stage="hostile_cell",
            )
    with pytest.raises(IdentityValidationError, match="invalid_type"):
        align_labels_with_metadata(
            pd.DataFrame({"slide_id": ["slide_a"], "spot_id": ["p0"]}),
            pd.DataFrame({"slide_id": ["slide_a"], "spot_id": ["p0"]}),
            stage=hostile,
        )
    with pytest.raises(IdentityValidationError, match="invalid_type"):
        validate_anndata_spot_identity(
            SimpleNamespace(obs_names=pd.Index([hostile], dtype=object)),
            "slide_a",
            stage="hostile_obs",
        )
    assert calls == []


def test_persisted_slide_identity_rejects_null_and_string_subclass_without_hooks():
    calls: list[str] = []

    class HostileString(str):
        def strip(self, *_args, **_kwargs):
            calls.append("strip")
            raise AssertionError("strip hook executed")

        def __repr__(self):
            calls.append("repr")
            raise AssertionError("repr hook executed")

    adata = SimpleNamespace(
        obs_names=pd.Index(["spot_a", "spot_b"], dtype=object),
        obs=pd.DataFrame(
            {"slide_id": pd.Series([None, HostileString("slide_a")], dtype=object)}
        ),
    )

    with pytest.raises(IdentityValidationError) as caught:
        validate_anndata_spot_identity(
            adata,
            "slide_a",
            stage="persisted_types",
            require_slide_id=True,
        )

    assert caught.value.issues[0].code == "invalid_type"
    assert caught.value.issues[0].count == 2
    assert calls == []


def _evil_column_frame(calls: list[str]) -> pd.DataFrame:
    class EvilColumnMeta(type):
        def __getattribute__(cls, name):
            if name in {"__module__", "__name__", "__qualname__"}:
                calls.append(name)
                raise AssertionError("column metaclass naming hook executed")
            return super().__getattribute__(name)

    class EvilColumn(metaclass=EvilColumnMeta):
        def __eq__(self, _other):
            calls.append("eq")
            raise AssertionError("column equality hook executed")

        def __hash__(self):
            calls.append("hash")
            raise AssertionError("column hash hook executed")

        def __repr__(self):
            calls.append("repr")
            raise AssertionError("column repr hook executed")

        def __str__(self):
            calls.append("str")
            raise AssertionError("column str hook executed")

    frame = pd.DataFrame([["value"]])
    frame.columns = pd.Index([EvilColumn()], dtype=object)
    calls.clear()
    return frame


@pytest.mark.parametrize("evil_side", ["labels", "metadata"])
def test_alignment_rejects_hostile_column_labels_before_hooks(evil_side):
    calls: list[str] = []
    tables = {
        "labels": pd.DataFrame(
            {"slide_id": ["slide_a"], "spot_id": ["spot_a"]}
        ),
        "metadata": pd.DataFrame(
            {"slide_id": ["slide_a"], "spot_id": ["spot_a"]}
        ),
    }
    tables[evil_side] = _evil_column_frame(calls)

    with pytest.raises(IdentityValidationError) as caught:
        align_labels_with_metadata(
            tables["labels"],
            tables["metadata"],
            stage="hostile_schema",
        )

    issue = caught.value.issues[0]
    assert issue.code == "invalid_type"
    assert issue.side == evil_side
    assert issue.sample_rows == ((0, "column_label", "non_builtin_object"),)
    assert calls == []


@pytest.mark.parametrize("require_slide_id", [False, True])
def test_anndata_rejects_hostile_column_labels_before_hooks(require_slide_id):
    calls: list[str] = []
    adata = SimpleNamespace(
        obs_names=pd.Index(["spot_a"], dtype=object),
        obs=_evil_column_frame(calls),
    )

    with pytest.raises(IdentityValidationError) as caught:
        validate_anndata_spot_identity(
            adata,
            "slide_a",
            stage="hostile_anndata_schema",
            require_slide_id=require_slide_id,
        )

    issue = caught.value.issues[0]
    assert issue.code == "invalid_type"
    assert issue.side == "anndata_schema"
    assert issue.sample_rows == ((0, "column_label", "non_builtin_object"),)
    assert calls == []


def test_exact_string_schema_keeps_missing_and_reserved_issue_order():
    labels = pd.DataFrame({"spot_id": ["spot_a"], "_label_source_row": [0]})
    metadata = pd.DataFrame({"slide_id": ["slide_a"]})

    with pytest.raises(IdentityValidationError) as caught:
        align_labels_with_metadata(labels, metadata, stage="exact_schema")

    assert [(issue.code, issue.side) for issue in caught.value.issues] == [
        ("missing_column", "labels"),
        ("reserved_column", "labels"),
        ("missing_column", "metadata"),
    ]


def _duplicate_column_frame(name: str) -> pd.DataFrame:
    other = "spot_id" if name == "slide_id" else "slide_id"
    return pd.DataFrame(
        [["slide_a", "slide_a", "spot_a"]],
        columns=[name, name, other],
    )


@pytest.mark.parametrize("duplicate_side", ["labels", "metadata"])
@pytest.mark.parametrize("column", ["slide_id", "spot_id"])
def test_alignment_rejects_duplicate_required_columns_before_selection(
    duplicate_side, column
):
    tables = {
        "labels": pd.DataFrame(
            {"slide_id": ["slide_a"], "spot_id": ["spot_a"]}
        ),
        "metadata": pd.DataFrame(
            {"slide_id": ["slide_a"], "spot_id": ["spot_a"]}
        ),
    }
    tables[duplicate_side] = _duplicate_column_frame(column)

    with pytest.raises(IdentityValidationError) as caught:
        align_labels_with_metadata(
            tables["labels"],
            tables["metadata"],
            stage="duplicate_schema",
        )

    assert len(caught.value.issues) == 1
    issue = caught.value.issues[0]
    assert issue.code == "duplicate_column"
    assert issue.side == duplicate_side
    assert issue.count == 1
    assert issue.sample_rows == ((1, column, "builtins.str"),)


@pytest.mark.parametrize("column", ["slide_id", "spot_id"])
def test_persisted_anndata_rejects_duplicate_identity_columns(column):
    adata = SimpleNamespace(
        obs_names=pd.Index(["spot_a"], dtype=object),
        obs=_duplicate_column_frame(column),
    )

    with pytest.raises(IdentityValidationError) as caught:
        validate_anndata_spot_identity(
            adata,
            "slide_a",
            stage="duplicate_persisted_schema",
            require_slide_id=True,
        )

    issue = caught.value.issues[0]
    assert issue.code == "duplicate_column"
    assert issue.side == "anndata_schema"
    assert issue.count == 1
    assert issue.sample_rows == ((1, column, "builtins.str"),)


@pytest.mark.parametrize(
    ("side", "reserved"),
    [("labels", "_label_source_row"), ("metadata", "_patch_source_row")],
)
def test_duplicate_reserved_columns_are_counted_before_reserved_issue(
    side, reserved
):
    tables = {
        "labels": pd.DataFrame(
            {"slide_id": ["slide_a"], "spot_id": ["spot_a"]}
        ),
        "metadata": pd.DataFrame(
            {"slide_id": ["slide_a"], "spot_id": ["spot_a"]}
        ),
    }
    tables[side] = pd.DataFrame(
        [["slide_a", "spot_a", 0, 1, 2]],
        columns=["slide_id", "spot_id", reserved, reserved, reserved],
    )

    with pytest.raises(IdentityValidationError) as caught:
        align_labels_with_metadata(
            tables["labels"],
            tables["metadata"],
            stage="duplicate_reserved_schema",
        )

    assert [(issue.code, issue.count) for issue in caught.value.issues] == [
        ("duplicate_column", 2),
        ("reserved_column", 1),
    ]
    assert caught.value.issues[0].sample_rows == (
        (3, reserved, "builtins.str"),
        (4, reserved, "builtins.str"),
    )


@pytest.mark.parametrize("defect", ["duplicate", "unmatched", "cross_slide"])
def test_key_diagnostics_escape_controls_and_bound_long_unicode(defect):
    hostile_id = "line\n\t\x00" + "한🚀" * 100_000
    labels = pd.DataFrame(
        {
            "slide_id": ["slide_a", "slide_a"],
            "spot_id": [hostile_id, "p1"],
        }
    )
    metadata = pd.DataFrame(
        {
            "slide_id": ["slide_a", "slide_a"],
            "spot_id": [hostile_id, "p1"],
        }
    )
    if defect == "duplicate":
        labels.loc[1, "spot_id"] = hostile_id
    elif defect == "unmatched":
        metadata.loc[0, "spot_id"] = "metadata-only"
    else:
        metadata.loc[0, "slide_id"] = "slide_b"

    with pytest.raises(IdentityValidationError) as caught:
        align_labels_with_metadata(labels, metadata, stage="bounded")

    message = str(caught.value)
    assert "line\n" not in message
    assert "\\n\\t\\u0000" in message
    assert len(message) < 2_000
    assert any(
        hostile_id in component
        for issue in caught.value.issues
        for key in issue.sample_keys
        for component in key
    )


def test_public_alignment_facade_preserves_dataframe_contract(
    key_adversary_factory,
):
    case = key_adversary_factory()["shuffled_complete"]

    aligned = label_module.align_labels_with_patches(
        case["labels"], case["patch_index"]
    )

    assert isinstance(aligned, pd.DataFrame)
    assert aligned["spot_id"].tolist() == case["patch_index"]["spot_id"].tolist()


def test_label_producer_rejects_anndata_identity_before_scientific_work(monkeypatch):
    invalid = SimpleNamespace(obs_names=pd.Index(["spot_a", None], dtype=object))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("label scientific seam was reached")

    fake_scanpy = SimpleNamespace(
        tl=SimpleNamespace(rank_genes_groups=forbidden),
        get=SimpleNamespace(rank_genes_groups_df=forbidden),
    )
    monkeypatch.setitem(__import__("sys").modules, "scanpy", fake_scanpy)
    monkeypatch.setattr(label_module, "load_slide", lambda _slide_id: invalid)
    monkeypatch.setattr(label_module, "tme_class_to_id", forbidden)
    monkeypatch.setattr(label_module, "marker_genes_for_slide", forbidden)
    monkeypatch.setattr(label_module, "compute_module_scores", forbidden)
    monkeypatch.setattr(label_module.st, "genes_present", forbidden)

    with pytest.raises(IdentityValidationError, match="invalid_type"):
        label_module.build_labels_for_slide("slide_a", cfg={"seed": 0})


@pytest.mark.parametrize(
    "persisted_slide_ids", [["slide_b", "slide_b"], ["slide_a", "slide_b"]]
)
def test_label_producer_rejects_wrong_persisted_slide_before_science(
    monkeypatch, persisted_slide_ids
):
    invalid = SimpleNamespace(
        obs_names=pd.Index(["spot_a", "spot_b"], dtype=object),
        obs=pd.DataFrame({"slide_id": persisted_slide_ids}),
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("label scientific seam was reached")

    monkeypatch.setitem(
        __import__("sys").modules,
        "scanpy",
        SimpleNamespace(
            tl=SimpleNamespace(rank_genes_groups=forbidden),
            get=SimpleNamespace(rank_genes_groups_df=forbidden),
        ),
    )
    monkeypatch.setattr(label_module, "load_slide", lambda _slide_id: invalid)
    monkeypatch.setattr(label_module, "tme_class_to_id", forbidden)

    with pytest.raises(IdentityValidationError, match="wrong_slide"):
        label_module.build_labels_for_slide("slide_a", cfg={"seed": 0})


def test_patch_producer_rejects_anndata_identity_before_coordinates_or_stack(
    monkeypatch,
):
    invalid = SimpleNamespace(obs_names=pd.Index(["spot_a", None], dtype=object))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("patch scientific seam was reached")

    monkeypatch.setattr(patch_module, "coords_hires", forbidden)
    monkeypatch.setattr(patch_module.st, "get_image", forbidden)
    monkeypatch.setattr(patch_module.np, "stack", forbidden)

    with pytest.raises(IdentityValidationError, match="invalid_type"):
        patch_module._extract_spot_patches(
            invalid,
            "slide_a",
            np.eye(2, 3),
            {"patches": {}},
        )


@pytest.mark.parametrize(
    "persisted_slide_ids", [["slide_b", "slide_b"], ["slide_a", "slide_b"]]
)
def test_patch_producer_rejects_wrong_persisted_slide_before_coordinates(
    monkeypatch, persisted_slide_ids
):
    invalid = SimpleNamespace(
        obs_names=pd.Index(["spot_a", "spot_b"], dtype=object),
        obs=pd.DataFrame({"slide_id": persisted_slide_ids}),
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("patch scientific seam was reached")

    monkeypatch.setattr(patch_module, "coords_hires", forbidden)
    monkeypatch.setattr(patch_module.st, "get_image", forbidden)

    with pytest.raises(IdentityValidationError, match="wrong_slide"):
        patch_module._extract_spot_patches(
            invalid,
            "slide_a",
            np.eye(2, 3),
            {"patches": {}},
        )


def test_ordinary_consumer_uses_patch_source_order(monkeypatch):
    labels = pd.DataFrame(
        {
            "slide_id": ["slide_a", "slide_a", "slide_b"],
            "spot_id": ["p1", "p0", "other"],
            "target": [11, 10, 99],
        }
    )
    metadata = pd.DataFrame(
        {"slide_id": ["slide_a", "slide_a"], "spot_id": ["p0", "p1"]}
    )
    patch_values = np.asarray([[[[10.0]]], [[[11.0]]]], dtype=np.float32)
    monkeypatch.setattr(
        train_module,
        "load_patch_arrays",
        lambda *_args, **_kwargs: (patch_values, metadata),
    )

    patches, aligned = train_module.load_slide_patches(
        "slide_a", labels, cfg={}
    )

    np.testing.assert_array_equal(patches, patch_values)
    assert aligned["spot_id"].tolist() == ["p0", "p1"]
    assert aligned["target"].tolist() == [10, 11]
    assert aligned["_patch_source_row"].tolist() == [0, 1]


def test_ordinary_consumer_rejects_cardinality_before_indexing(monkeypatch):
    labels = pd.DataFrame(
        {"slide_id": ["slide_a"], "spot_id": ["p0"], "target": [1]}
    )
    metadata = labels[["slide_id", "spot_id"]].copy()
    patch_values = np.zeros((2, 1, 1, 1), dtype=np.float32)
    monkeypatch.setattr(
        train_module,
        "load_patch_arrays",
        lambda *_args, **_kwargs: (patch_values, metadata),
    )

    with pytest.raises(IdentityValidationError, match="cardinality_mismatch"):
        train_module.load_slide_patches("slide_a", labels, cfg={})


def _foundation_test_config() -> dict:
    cfg = foundation_module.load_config()
    cfg["foundation"].update(model="kaiko_vits16", cache=True, batch_size=2)
    cfg["patches"]["version"] = "v1"
    return cfg


def _publish_test_embedding(cache_path, embeddings, spot_ids, labels, cfg):
    expected = foundation_module._expected_spot_ids(labels, "slide_a")
    spec = FOUNDATION_MODELS["kaiko_vits16"]
    try:
        schema = foundation_module._embedding_schema(
            embeddings,
            spot_ids,
            expected_spots=expected,
            expected_dim=spec.embedding_dim,
        )
    except Exception:
        schema = {"test_semantic_corruption": True}

    def write_payload(path):
        with path.open("wb") as handle:
            np.savez_compressed(
                handle, embeddings=embeddings, spot_ids=spot_ids
            )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    publish_artifact(
        cache_path,
        artifact_kind="embedding",
        contract_version=ARTIFACT_CONTRACT_VERSIONS["embedding"],
        fingerprint=foundation_module._embedding_fingerprint(
            "slide_a", expected, cfg
        ),
        payload_format="npz-safe-primitives",
        payload_schema=schema,
        write_payload=write_payload,
        reader=lambda _path: None,
        observed_schema=lambda _decoded: schema,
    )


def test_foundation_cache_hit_rejects_subset_before_patch_or_encoder(
    monkeypatch, tmp_path
):
    cache_path = tmp_path / "cached.npz"
    labels = pd.DataFrame(
        {
            "slide_id": ["slide_a", "slide_a"],
            "spot_id": ["p0", "p1"],
            "target": [0, 1],
        }
    )
    cfg = _foundation_test_config()
    _publish_test_embedding(
        cache_path,
        np.zeros((1, FOUNDATION_MODELS["kaiko_vits16"].embedding_dim), dtype=np.float32),
        np.asarray(["p0"], dtype=np.str_),
        labels,
        cfg,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("foundation fallback or encoder seam was reached")

    monkeypatch.setattr(foundation_module, "_resolved_config_arg", lambda cfg: cfg)
    monkeypatch.setattr(
        foundation_module, "_embedding_cache_path", lambda *_args: cache_path
    )
    monkeypatch.setattr(foundation_module, "load_slide_patches", forbidden)
    monkeypatch.setattr(foundation_module, "load_frozen_encoder", forbidden)

    with pytest.raises(ArtifactValidationError, match="reader_validation_failed"):
        foundation_module.load_or_extract_slide_embeddings(
            "slide_a", labels, cfg=cfg
        )


def test_foundation_cache_hit_preserves_cache_order(monkeypatch, tmp_path):
    cache_path = tmp_path / "cached.npz"
    embeddings = np.zeros(
        (2, FOUNDATION_MODELS["kaiko_vits16"].embedding_dim), dtype=np.float32
    )
    embeddings[:, :2] = np.asarray([[11.0, 1.0], [10.0, 0.0]])
    labels = pd.DataFrame(
        {
            "slide_id": ["slide_a", "slide_a", "slide_b"],
            "spot_id": ["p0", "p1", "other"],
            "target": [10, 11, 99],
        }
    ).iloc[::-1].reset_index(drop=True)
    cfg = _foundation_test_config()
    _publish_test_embedding(
        cache_path,
        embeddings,
        np.asarray(["p1", "p0"], dtype=np.str_),
        labels,
        cfg,
    )
    monkeypatch.setattr(foundation_module, "_resolved_config_arg", lambda cfg: cfg)
    monkeypatch.setattr(
        foundation_module, "_embedding_cache_path", lambda *_args: cache_path
    )

    actual, aligned = foundation_module.load_or_extract_slide_embeddings(
        "slide_a", labels, cfg=cfg
    )

    np.testing.assert_array_equal(actual, embeddings)
    assert aligned["spot_id"].tolist() == ["p1", "p0"]
    assert aligned["target"].tolist() == [11, 10]


def test_foundation_cache_miss_checks_extracted_cardinality_before_write(
    monkeypatch, tmp_path
):
    cache_path = tmp_path / "missing.npz"
    labels = pd.DataFrame(
        {
            "slide_id": ["slide_a", "slide_a"],
            "spot_id": ["p0", "p1"],
            "target": [10, 11],
            "_label_source_row": [0, 1],
            "_patch_source_row": [0, 1],
        }
    )
    patches = np.zeros((2, 3, 4, 4), dtype=np.float32)
    spec = FOUNDATION_MODELS["kaiko_vits16"]
    monkeypatch.setattr(foundation_module, "_resolved_config_arg", lambda cfg: cfg)
    monkeypatch.setattr(
        foundation_module, "_embedding_cache_path", lambda *_args: cache_path
    )
    monkeypatch.setattr(
        foundation_module,
        "load_slide_patches",
        lambda *_args, **_kwargs: (patches, labels),
    )
    monkeypatch.setattr(
        foundation_module,
        "extract_frozen_embeddings",
        lambda *_args, **_kwargs: np.zeros((1, spec.embedding_dim), dtype=np.float32),
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("cache write was reached")

    monkeypatch.setattr(foundation_module.np, "savez_compressed", forbidden)

    with pytest.raises(IdentityValidationError, match="cardinality_mismatch"):
        foundation_module.load_or_extract_slide_embeddings(
            "slide_a",
            labels.drop(columns=["_label_source_row", "_patch_source_row"]),
            cfg=_foundation_test_config(),
            encoder_bundle=(object(), "cpu", spec),
        )


class _LocalFoundationEncoder(torch.nn.Module):
    """Deterministic local encoder with the registered Kaiko output width."""

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        signal = batch.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        return signal.repeat(1, FOUNDATION_MODELS["kaiko_vits16"].embedding_dim)


def test_ordinary_and_foundation_cache_outcomes_share_exact_alignment(
    monkeypatch, tmp_path
):
    """One shuffled table must pair identically through every image consumer."""
    metadata = pd.DataFrame(
        {
            "slide_id": ["slide_a", "slide_a", "slide_a"],
            "spot_id": ["shared", "p0", "p1"],
        }
    )
    labels = pd.DataFrame(
        {
            "slide_id": ["slide_b", "slide_a", "slide_a", "slide_a"],
            "spot_id": ["shared", "p1", "shared", "p0"],
            "target": [900, 11, 10, 12],
            "provenance": ["other-slide", "label-p1", "label-shared", "label-p0"],
        }
    )
    patches = np.stack(
        [np.full((3, 4, 4), value, dtype=np.float32) for value in (0.1, 0.2, 0.3)]
    )
    monkeypatch.setattr(
        train_module,
        "load_patch_arrays",
        lambda *_args, **_kwargs: (patches, metadata.copy()),
    )
    config = _foundation_test_config()
    cache_path = tmp_path / "cross_arm.npz"
    monkeypatch.setattr(foundation_module, "_resolved_config_arg", lambda cfg: cfg)
    monkeypatch.setattr(
        foundation_module, "_embedding_cache_path", lambda *_args: cache_path
    )

    ordinary_values, ordinary_labels = train_module.load_slide_patches(
        "slide_a", labels, cfg=config
    )
    miss_values, miss_labels = foundation_module.load_or_extract_slide_embeddings(
        "slide_a",
        labels,
        cfg=config,
        encoder_bundle=(
            _LocalFoundationEncoder(),
            torch.device("cpu"),
            FOUNDATION_MODELS["kaiko_vits16"],
        ),
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("cache hit attempted patch or encoder work")

    monkeypatch.setattr(foundation_module, "load_slide_patches", forbidden)
    monkeypatch.setattr(foundation_module, "load_frozen_encoder", forbidden)
    hit_values, hit_labels = foundation_module.load_or_extract_slide_embeddings(
        "slide_a", labels, cfg=config
    )

    expected_pairs = [
        ("slide_a", "shared", 10, "label-shared"),
        ("slide_a", "p0", 12, "label-p0"),
        ("slide_a", "p1", 11, "label-p1"),
    ]
    for aligned in (ordinary_labels, miss_labels, hit_labels):
        assert list(
            aligned[["slide_id", "spot_id", "target", "provenance"]].itertuples(
                index=False, name=None
            )
        ) == expected_pairs
        assert aligned["_patch_source_row"].tolist() == [0, 1, 2]
        assert aligned["_label_source_row"].tolist() == [2, 3, 1]
    np.testing.assert_array_equal(ordinary_values, patches)
    np.testing.assert_array_equal(hit_values, miss_values)
    assert cache_path.is_file()


class _HostileConsumerString(str):
    def _fail(self, operation):
        raise AssertionError(f"consumer executed hostile {operation}")

    def strip(self, *_args, **_kwargs):
        return self._fail("strip")

    def __hash__(self):
        return self._fail("hash")

    def __eq__(self, _other):
        return self._fail("equality")

    def __repr__(self):
        return self._fail("repr")


def _consumer_adversary(defect: str) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    labels = pd.DataFrame(
        {
            "slide_id": ["slide_a", "slide_a"],
            "spot_id": ["p0", "p1"],
            "target": [0, 1],
        }
    )
    metadata = labels[["slide_id", "spot_id"]].copy()
    value_rows = 2
    if defect == "null":
        labels.loc[0, "spot_id"] = None
    elif defect == "blank":
        labels.loc[0, "spot_id"] = "   "
    elif defect == "hostile":
        labels["spot_id"] = labels["spot_id"].astype(object)
        labels.loc[0, "spot_id"] = _HostileConsumerString("p0")
    elif defect == "duplicate":
        labels.loc[1, "spot_id"] = "p0"
    elif defect == "label_only":
        labels.loc[0, "spot_id"] = "label-only"
    elif defect == "metadata_only":
        metadata.loc[0, "spot_id"] = "metadata-only"
    elif defect == "cross_slide":
        labels.loc[0, "slide_id"] = "slide_b"
    elif defect == "wrong_slide":
        metadata.loc[0, "slide_id"] = "slide_b"
    elif defect == "row_count":
        value_rows = 1
    else:  # pragma: no cover - test parameter invariant
        raise AssertionError(defect)
    return labels, metadata, value_rows


@pytest.mark.parametrize("consumer", ["ordinary", "foundation_miss"])
@pytest.mark.parametrize(
    "defect",
    [
        "null",
        "blank",
        "hostile",
        "duplicate",
        "label_only",
        "metadata_only",
        "cross_slide",
        "wrong_slide",
        "row_count",
    ],
)
def test_public_consumers_guard_identity_before_merge_index_encoder_or_write(
    monkeypatch, tmp_path, consumer, defect
):
    labels, metadata, value_rows = _consumer_adversary(defect)
    patches = np.zeros((value_rows, 3, 4, 4), dtype=np.float32)
    monkeypatch.setattr(
        train_module,
        "load_patch_arrays",
        lambda *_args, **_kwargs: (patches, metadata),
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("post-admission consumer seam was reached")

    monkeypatch.setattr(pd.DataFrame, "merge", forbidden)
    if consumer == "ordinary":
        def call():
            return train_module.load_slide_patches("slide_a", labels, cfg={})
    else:
        cache_path = tmp_path / f"{defect}.npz"
        monkeypatch.setattr(foundation_module, "_resolved_config_arg", lambda cfg: cfg)
        monkeypatch.setattr(
            foundation_module, "_embedding_cache_path", lambda *_args: cache_path
        )
        monkeypatch.setattr(foundation_module, "load_frozen_encoder", forbidden)
        monkeypatch.setattr(foundation_module.np, "savez_compressed", forbidden)
        def call():
            return foundation_module.load_or_extract_slide_embeddings(
                "slide_a",
                labels,
                cfg=_foundation_test_config(),
                encoder_bundle=(
                    _LocalFoundationEncoder(),
                    torch.device("cpu"),
                    FOUNDATION_MODELS["kaiko_vits16"],
                ),
            )

    with pytest.raises(IdentityValidationError):
        call()


@pytest.mark.parametrize(
    "defect",
    ["null", "blank", "hostile", "duplicate", "label_only", "cross_slide"],
)
def test_foundation_cache_hit_guards_labels_before_merge_or_encoder(
    monkeypatch, tmp_path, defect
):
    labels, metadata, _ = _consumer_adversary(defect)
    cache_path = tmp_path / f"hit_{defect}.npz"
    cfg = _foundation_test_config()
    if defect in {"label_only", "cross_slide"}:
        _publish_test_embedding(
            cache_path,
            np.zeros(
                (2, FOUNDATION_MODELS["kaiko_vits16"].embedding_dim),
                dtype=np.float32,
            ),
            metadata["spot_id"].to_numpy(dtype=np.str_),
            labels,
            cfg,
        )
    monkeypatch.setattr(foundation_module, "_resolved_config_arg", lambda cfg: cfg)
    monkeypatch.setattr(
        foundation_module, "_embedding_cache_path", lambda *_args: cache_path
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("cache-hit post-admission seam was reached")

    monkeypatch.setattr(pd.DataFrame, "merge", forbidden)
    monkeypatch.setattr(foundation_module, "load_slide_patches", forbidden)
    monkeypatch.setattr(foundation_module, "load_frozen_encoder", forbidden)

    with pytest.raises((IdentityValidationError, ArtifactValidationError)):
        foundation_module.load_or_extract_slide_embeddings(
            "slide_a", labels, cfg=cfg
        )


def test_foundation_cache_hit_rejects_value_row_count_before_merge(
    monkeypatch, tmp_path
):
    labels, metadata, _ = _consumer_adversary("row_count")
    cache_path = tmp_path / "hit_row_count.npz"
    cfg = _foundation_test_config()
    _publish_test_embedding(
        cache_path,
        np.zeros((1, FOUNDATION_MODELS["kaiko_vits16"].embedding_dim), dtype=np.float32),
        metadata["spot_id"].to_numpy(dtype=np.str_),
        labels,
        cfg,
    )
    monkeypatch.setattr(foundation_module, "_resolved_config_arg", lambda cfg: cfg)
    monkeypatch.setattr(
        foundation_module, "_embedding_cache_path", lambda *_args: cache_path
    )
    monkeypatch.setattr(
        pd.DataFrame,
        "merge",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("merge was reached")
        ),
    )

    with pytest.raises(ArtifactValidationError, match="reader_validation_failed"):
        foundation_module.load_or_extract_slide_embeddings(
            "slide_a", labels, cfg=cfg
        )
