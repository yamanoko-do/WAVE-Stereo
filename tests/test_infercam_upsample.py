import numpy as np

from wavestereo.inference.postprocessing import (
    DisparityPostprocessor,
    SpatialTransform,
)


def _restore(disparity, *, output_w, output_h, method, sigma=2.0):
    transform = SpatialTransform.from_sizes(
        (output_h, output_w),
        disparity.shape,
    )
    return DisparityPostprocessor(method=method, sigma=sigma).restore(
        disparity,
        transform,
        max_disp=192.0,
    ).disparity


def test_adaptive_upsample_uses_bilinear_behavior_in_flat_regions():
    disparity = np.full((2, 2), 7.0, dtype=np.float32)

    restored = _restore(
        disparity,
        output_w=4,
        output_h=4,
        method="adaptive",
        sigma=2.0,
    )

    # Horizontal resizing doubles disparity values from 7 to 14 pixels.
    np.testing.assert_allclose(restored, 14.0)


def test_adaptive_upsample_does_not_mix_across_large_disparity_step():
    disparity = np.array([[5.0, 25.0], [5.0, 25.0]], dtype=np.float32)

    bilinear = _restore(
        disparity, output_w=8, output_h=4, method="bilinear"
    )
    adaptive = _restore(
        disparity,
        output_w=8,
        output_h=4,
        method="adaptive",
        sigma=2.0,
    )

    assert np.any((bilinear > 20.0) & (bilinear < 100.0))
    assert not np.any((adaptive > 21.0) & (adaptive < 99.0))
