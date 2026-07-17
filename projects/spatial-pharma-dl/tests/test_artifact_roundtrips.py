"""Offline format evidence for primitive scientific fixture artifacts."""

from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.offline


def test_numeric_and_unicode_npz_round_trip(tmp_path) -> None:
    path = tmp_path / "primitive_arrays.npz"
    patches = np.arange(2 * 3 * 4 * 4, dtype=np.float32).reshape(2, 3, 4, 4)
    spot_ids = np.asarray(["slide_a_spot_00", "slide_a_spot_α"], dtype=np.str_)

    np.savez_compressed(path, patches=patches, spot_ids=spot_ids)

    with np.load(path, allow_pickle=False) as cached:
        assert set(cached.files) == {"patches", "spot_ids"}
        assert cached["patches"].dtype == np.float32
        assert cached["patches"].shape == (2, 3, 4, 4)
        np.testing.assert_array_equal(cached["patches"], patches)
        assert cached["spot_ids"].dtype.kind == "U"
        assert cached["spot_ids"].tolist() == spot_ids.tolist()


def test_object_npz_payload_is_rejected(
    artifact_adversary_factory,
) -> None:
    object_path = artifact_adversary_factory()["object_npz"]["path"]

    with np.load(object_path, allow_pickle=False) as cached:
        assert cached.files == ["payload"]
        with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
            cached["payload"]


def test_parquet_and_primitive_json_round_trip(tmp_path) -> None:
    parquet_path = tmp_path / "spot_metadata.parquet"
    json_path = tmp_path / "manifest.json"
    metadata = pd.DataFrame(
        {
            "slide_id": ["slide_a", "slide_a", "slide_b"],
            "spot_id": ["spot_00", "spot_01", "spot_02"],
            "row": pd.Series([0, 1, 2], dtype="int64"),
            "score": pd.Series([0.25, 0.5, 0.75], dtype="float32"),
        }
    )
    manifest = {
        "schema_version": 1,
        "complete": True,
        "format": "fixture",
        "row_count": len(metadata),
        "keys": ["slide_id", "spot_id"],
    }

    metadata.to_parquet(parquet_path, index=False)
    json_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    restored_metadata = pd.read_parquet(parquet_path)
    restored_manifest = json.loads(json_path.read_text(encoding="utf-8"))
    pd.testing.assert_frame_equal(restored_metadata, metadata)
    assert restored_metadata["spot_id"].tolist() == metadata["spot_id"].tolist()
    assert restored_manifest == manifest
    assert all(path.parent == tmp_path for path in (parquet_path, json_path))


def test_h5ad_axes_and_spatial_metadata_round_trip(
    tmp_path, synthetic_anndata_factory
) -> None:
    path = tmp_path / "synthetic_slide.h5ad"
    original = synthetic_anndata_factory(slide_id="slide_roundtrip", n_spots=6)
    # AnnData 0.10 cannot serialize pandas 3 Arrow-backed inferred strings.
    original.obs_names = pd.Index(original.obs_names.to_numpy(dtype=object), dtype=object)
    original.var_names = pd.Index(original.var_names.to_numpy(dtype=object), dtype=object)
    original.obs["slide_id"] = original.obs["slide_id"].astype(object)
    original.write_h5ad(path)

    restored = ad.read_h5ad(path)
    library_id = "library_slide_roundtrip"
    assert restored.obs_names.tolist() == original.obs_names.tolist()
    assert restored.var_names.tolist() == original.var_names.tolist()
    np.testing.assert_array_equal(restored.X, original.X)
    np.testing.assert_array_equal(restored.obsm["spatial"], original.obsm["spatial"])
    np.testing.assert_array_equal(
        restored.uns["spatial"][library_id]["images"]["hires"],
        original.uns["spatial"][library_id]["images"]["hires"],
    )
    assert restored.uns["spatial"][library_id]["scalefactors"] == original.uns[
        "spatial"
    ][library_id]["scalefactors"]
    assert path.parent == tmp_path
