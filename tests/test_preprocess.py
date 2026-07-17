import numpy as np

from wavestereo.inference.preprocess import StereoPreprocessor


def test_preprocess_divisible_shape_cpu():
    left = np.zeros((31, 33, 3), dtype=np.uint8)
    right = np.zeros((31, 33, 3), dtype=np.uint8)
    pre = StereoPreprocessor(device="cpu", divisible_by=32)
    sample, pad = pre.prepare(left, right)
    assert sample["left"].shape == (1, 3, 32, 64)
    assert sample["right"].shape == (1, 3, 32, 64)
    disp = np.zeros((32, 64), dtype=np.float32)
    assert pre.crop_disparity(disp, pad).shape == (31, 33)
