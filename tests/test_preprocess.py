import numpy as np
import pytest

from wavestereo.inference.preprocess import (
    StereoPreprocessor,
    normalization_from_config,
)


def test_preprocess_divisible_shape_cpu():
    left = np.zeros((31, 33, 3), dtype=np.uint8)
    right = np.zeros((31, 33, 3), dtype=np.uint8)
    pre = StereoPreprocessor(device="cpu", divisible_by=32)
    sample, pad = pre.prepare(left, right)
    assert sample["left"].shape == (1, 3, 32, 64)
    assert sample["right"].shape == (1, 3, 32, 64)
    disp = np.zeros((32, 64), dtype=np.float32)
    assert pre.crop_disparity(disp, pad).shape == (31, 33)


def test_configured_normalization_is_required_and_validated():
    assert normalization_from_config(
        {"MEAN": [0.1, 0.2, 0.3], "STD": [1, 2, 3]}
    ) == ([0.1, 0.2, 0.3], [1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="MEAN/STD"):
        normalization_from_config({})
    with pytest.raises(ValueError, match="positive"):
        normalization_from_config(
            {"MEAN": [0.1, 0.2, 0.3], "STD": [1.0, 0.0, 1.0]}
        )
