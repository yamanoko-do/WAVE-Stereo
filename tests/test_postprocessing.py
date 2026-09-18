from __future__ import annotations

import numpy as np
import pytest

from wavestereo.inference.postprocessing import (
    DisparityPostprocessor,
    SpatialTransform,
)


def test_spatial_transform_uses_height_width_order_and_resizes_once():
    transform = SpatialTransform.from_sizes((2, 4), (1, 2))
    image = np.arange(2 * 4 * 3, dtype=np.uint8).reshape(2, 4, 3)

    prepared = transform.prepare_stereo({"left": image, "right": image})

    assert transform.source_size == (2, 4)
    assert transform.model_size == (1, 2)
    assert transform.width_scale == 2.0
    assert prepared["left"].shape == (1, 2, 3)
    assert prepared["left"].dtype == np.uint8
    assert prepared["left"].flags.c_contiguous

    # A fused camera path can submit an already resized model frame without a
    # second interpolation pass.
    already_model_sized = prepared["left"]
    assert transform.prepare_image(already_model_sized) is already_model_sized


def test_spatial_transform_rejects_undeclared_image_geometry_and_dtype():
    transform = SpatialTransform(2, 4, 1, 2)
    with pytest.raises(ValueError, match="source|model"):
        transform.prepare_image(np.zeros((3, 3, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="uint8"):
        transform.prepare_image(np.zeros((2, 4, 3), dtype=np.float32))


def test_disparity_postprocessor_is_single_cleanup_clip_scale_restore_boundary():
    transform = SpatialTransform(4, 4, 2, 2)
    postprocessor = DisparityPostprocessor(method="nearest")
    native = np.array(
        [[np.nan, np.inf], [-1.0, 300.0]],
        dtype=np.float32,
    )

    result = postprocessor.restore(native, transform, max_disp=192.0)

    expected_native = np.array(
        [[0.0, 0.0], [0.0, 384.0]], dtype=np.float32
    )
    expected = np.repeat(np.repeat(expected_native, 2, axis=0), 2, axis=1)
    np.testing.assert_array_equal(result.disparity, expected)
    assert result.effective_max_disp == 384.0
    # The backend-owned array is never mutated by postprocessing.
    assert np.isnan(native[0, 0])
    assert np.isposinf(native[0, 1])


def test_disparity_postprocessor_defaults_to_adaptive_sigma_two():
    postprocessor = DisparityPostprocessor()

    assert postprocessor.method == "adaptive"
    assert postprocessor.sigma == 2.0

    transform = SpatialTransform(4, 4, 2, 2)
    result = postprocessor.restore(
        np.full((2, 2), 3.0, dtype=np.float32),
        transform,
        max_disp=192.0,
    )
    np.testing.assert_allclose(result.disparity, 6.0)


def test_disparity_postprocessor_requires_native_model_shape():
    transform = SpatialTransform(4, 4, 2, 2)
    with pytest.raises(ValueError, match="model size"):
        DisparityPostprocessor().restore(
            np.zeros((4, 4), dtype=np.float32), transform, 192.0
        )
