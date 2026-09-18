import cv2
import numpy as np
import pytest

from tools.infercam import parse_args
from wavestereo.visual.core import AsyncCamera, resize_rectification_maps


def _identity_maps(width, height):
    x, y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    return x, y


class _MappedCamera:
    fps = 30

    def __init__(self, image):
        self.image = image
        self.map1x, self.map1y = _identity_maps(image.shape[1], image.shape[0])
        self.map2x = self.map1x.copy()
        self.map2y = self.map1y.copy()
        self.raw_calls = 0
        self.rectified_calls = 0

    def get_frame(self):
        self.raw_calls += 1
        return {"left": self.image, "right": self.image}

    def get_rectifyframe(self):
        self.rectified_calls += 1
        left = cv2.remap(
            self.image, self.map1x, self.map1y, interpolation=cv2.INTER_LINEAR
        )
        right = cv2.remap(
            self.image, self.map2x, self.map2y, interpolation=cv2.INTER_LINEAR
        )
        return {"left": left, "right": right}


class _RectifiedCamera:
    fps = 30

    def __init__(self, image):
        self.image = image

    def get_rectifyframe(self):
        return {"left": self.image[:, ::-1], "right": self.image[:, ::-1]}


def test_resize_rectification_maps_preserves_source_coordinate_domain():
    map_x, map_y = _identity_maps(16, 8)

    resized_x, resized_y = resize_rectification_maps(map_x, map_y, 8, 4)

    assert resized_x.shape == (4, 8)
    assert resized_y.shape == (4, 8)
    assert resized_x.dtype == np.float32
    assert resized_y.dtype == np.float32
    assert resized_x.max() > 13.0
    assert resized_y.max() > 5.0


def test_async_camera_auto_fuses_rectification_and_resize():
    image = np.arange(8 * 16 * 3, dtype=np.uint8).reshape(8, 16, 3)
    source = _MappedCamera(image)
    camera = AsyncCamera(
        camera=source,
        inference_resolution=(8, 4),
        preprocessing_mode="auto",
    )

    frame = camera._capture_frame()

    assert camera.fused_rectify_resize
    assert source.raw_calls == 1
    assert source.rectified_calls == 0
    assert frame["left"].shape == (4, 8, 3)
    assert frame["right"].shape == (4, 8, 3)
    assert frame["left"].dtype == np.uint8
    assert frame["left"].flags.c_contiguous


def test_async_camera_legacy_rectifies_full_resolution_before_resize():
    image = np.arange(8 * 16 * 3, dtype=np.uint8).reshape(8, 16, 3)
    source = _MappedCamera(image)
    camera = AsyncCamera(
        camera=source,
        inference_resolution=(8, 4),
        preprocessing_mode="legacy",
    )

    frame = camera._capture_frame()

    assert not camera.fused_rectify_resize
    assert source.raw_calls == 0
    assert source.rectified_calls == 1
    assert frame["left"].shape == (4, 8, 3)
    assert frame["right"].shape == (4, 8, 3)


def test_async_camera_auto_uses_regular_rectification_when_size_matches():
    image = np.zeros((8, 16, 3), dtype=np.uint8)
    source = _MappedCamera(image)
    camera = AsyncCamera(
        camera=source,
        inference_resolution=(16, 8),
        preprocessing_mode="auto",
    )

    frame = camera._capture_frame()

    assert not camera.fused_rectify_resize
    assert source.raw_calls == 0
    assert source.rectified_calls == 1
    assert frame["left"].shape == (8, 16, 3)


def test_async_camera_resizes_external_source_without_maps():
    image = np.zeros((8, 16, 3), dtype=np.uint8)
    camera = AsyncCamera(
        camera=_RectifiedCamera(image),
        inference_resolution=(8, 4),
        preprocessing_mode="auto",
    )

    frame = camera._capture_frame()

    assert not camera.fused_rectify_resize
    assert frame["left"].shape == (4, 8, 3)
    assert frame["right"].shape == (4, 8, 3)
    assert frame["left"].flags.c_contiguous
    assert frame["right"].flags.c_contiguous


def test_async_camera_rejects_invalid_preprocessing_mode():
    image = np.zeros((8, 16, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="auto.*legacy"):
        AsyncCamera(
            camera=_RectifiedCamera(image),
            inference_resolution=(8, 4),
            preprocessing_mode="fused",
        )


def test_infercam_parser_defaults_to_auto_camera_preprocessing():
    args = parse_args([])

    assert args.camera_preprocess == "auto"
    assert args.disparity_upsample == "adaptive"
    assert args.disparity_upsample_sigma == 2.0


def test_infercam_parser_accepts_legacy_camera_preprocessing():
    args = parse_args(["--camera-preprocess", "legacy"])

    assert args.camera_preprocess == "legacy"


def test_infercam_parser_rejects_removed_fused_camera_flag():
    with pytest.raises(SystemExit):
        parse_args(["--fused-camera-preprocess"])


def test_infercam_parser_exposes_only_public_backends():
    with pytest.raises(SystemExit):
        parse_args(["--backend", "tensorrt"])
