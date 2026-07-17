"""Offline integration evidence for the tiny Visium-shaped AnnData fixture."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from src.patches import coords_hires, extract_all_patches_for_slide
from utils import st_helpers as st

pytestmark = pytest.mark.offline


def _prepare_strings_for_h5ad(adata) -> None:
    """Keep the fixture writable across pandas string-storage defaults."""
    adata.obs_names = pd.Index(adata.obs_names.to_numpy(dtype=object), dtype=object)
    adata.var_names = pd.Index(adata.var_names.to_numpy(dtype=object), dtype=object)
    adata.obs["slide_id"] = adata.obs["slide_id"].astype(object)


def test_real_anndata_round_trip_preserves_spatial_axes(
    tmp_path, synthetic_anndata_factory
) -> None:
    path = tmp_path / "spatial_fixture.h5ad"
    original = synthetic_anndata_factory(slide_id="slide_spatial", n_spots=8)
    _prepare_strings_for_h5ad(original)
    original.write_h5ad(path)

    restored = ad.read_h5ad(path)
    library_id = "library_slide_spatial"
    assert restored.obs_names.tolist() == original.obs_names.tolist()
    assert restored.var_names.tolist() == original.var_names.tolist()
    np.testing.assert_array_equal(restored.obsm["spatial"], original.obsm["spatial"])
    np.testing.assert_array_equal(
        restored.uns["spatial"][library_id]["images"]["hires"],
        original.uns["spatial"][library_id]["images"]["hires"],
    )
    assert restored.uns["spatial"][library_id]["scalefactors"] == {
        "spot_diameter_fullres": 12.0,
        "tissue_hires_scalef": 0.5,
    }
    assert path.parent == tmp_path


def test_public_image_scale_and_coordinate_accessors(
    synthetic_anndata_factory,
) -> None:
    adata = synthetic_anndata_factory(slide_id="slide_access", n_spots=6)
    library_id = "library_slide_access"

    np.testing.assert_array_equal(
        st.get_image(adata, "hires"),
        adata.uns["spatial"][library_id]["images"]["hires"],
    )
    assert st.get_scalefactors(adata) == {
        "tissue_hires_scalef": 0.5,
        "spot_diameter_fullres": 12.0,
    }
    np.testing.assert_allclose(coords_hires(adata), adata.obsm["spatial"] * 0.5)


def test_patch_extraction_and_valid_alignment_preserve_spot_order(
    synthetic_anndata_factory,
) -> None:
    slide_id = "slide_patches"
    adata = synthetic_anndata_factory(slide_id=slide_id, n_spots=8)
    cfg = {
        "patches": {
            "min_patch_px": 8,
            "output_size": 16,
            "context_scale": 1.0,
            "per_slide_stain_norm": False,
        }
    }
    reference_stain = np.asarray(
        [[0.65, 0.70, 0.29], [0.07, 0.99, 0.11]], dtype=np.float64
    )

    patches, patch_metadata = extract_all_patches_for_slide(
        adata, slide_id, reference_stain, cfg=cfg
    )

    expected_spots = adata.obs_names.tolist()
    assert patches.shape == (adata.n_obs, 3, 16, 16)
    assert patches.dtype == np.float32
    assert len(patch_metadata) == adata.n_obs
    assert patch_metadata["spot_id"].tolist() == expected_spots
    assert patch_metadata["slide_id"].tolist() == [slide_id] * adata.n_obs
    np.testing.assert_allclose(
        patch_metadata[["x", "y"]].to_numpy(), coords_hires(adata)
    )

    labels = pd.DataFrame(
        {
            "slide_id": [slide_id] * adata.n_obs,
            "spot_id": expected_spots,
            "tme_class_id": np.arange(adata.n_obs) % 2,
        }
    )
    aligned = patch_metadata.merge(
        labels, on=["slide_id", "spot_id"], how="inner", validate="one_to_one"
    )
    assert len(aligned) == adata.n_obs
    assert aligned["spot_id"].tolist() == expected_spots
