from __future__ import annotations

from argparse import Namespace

import numpy as np
import pytest
import torch

from tools.infer import parse_args
from wavestereo.inference.runtime import (
    RuntimeSetup,
    _external_size_from_config,
    _padded_size_from_config,
    run_once,
    validate_backend_args,
)
from wavestereo.inference.postprocessing import (
    DisparityPostprocessor,
    SpatialTransform,
)
from wavestereo.visual.async_inference import (
    AsyncInference,
    AsyncInferenceResult,
    BackendResult,
    OpenVINOBackend,
    PyTorchBackend,
)
from wavestereo.visual.core import Preprocessor


def _args(**overrides) -> Namespace:
    values = {
        "backend": "pytorch",
        "config": None,
        "weights": None,
        "model_dir": None,
        "openvino_device": None,
        "openvino_cache_dir": None,
        "device": None,
        "debug": False,
    }
    values.update(overrides)
    return Namespace(**values)


@pytest.mark.parametrize(
    "args",
    [
        _args(),
        _args(
            backend="openvino",
            model_dir="bundle",
            openvino_device="GPU.0",
        ),
        _args(
            backend="openvino",
            model_dir="bundle",
            openvino_device="NPU",
            openvino_cache_dir="cache",
        ),
    ],
)
def test_validate_backend_args_accepts_supported_combinations(args):
    validate_backend_args(args)


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (_args(backend="tensorrt"), "Unsupported public backend"),
        (_args(backend="openvino", openvino_device="NPU"), "requires --model-dir"),
        (_args(model_dir="bundle"), "--model-dir is only valid"),
        (
            _args(backend="openvino", model_dir="bundle", config="cfg.yaml"),
            "--config is only valid",
        ),
        (
            _args(backend="openvino", model_dir="bundle"),
            "requires --openvino-device",
        ),
        (_args(openvino_device="GPU.0"), "--openvino-device is only valid"),
        (
            _args(openvino_cache_dir="cache"),
            "--openvino-cache-dir is only valid",
        ),
        (
            _args(
                backend="openvino",
                model_dir="bundle",
                openvino_device="NPU",
                device="cpu",
            ),
            "--device is not valid",
        ),
    ],
)
def test_validate_backend_args_rejects_cross_backend_options(args, message):
    with pytest.raises(SystemExit, match=message):
        validate_backend_args(args)


def test_image_parser_uses_shared_openvino_interface():
    args = parse_args(
        [
            "--backend",
            "openvino",
            "--model-dir",
            "bundle",
            "--openvino-device",
            "NPU",
            "--left",
            "left.png",
            "--right",
            "right.png",
        ]
    )

    assert args.backend == "openvino"
    assert args.openvino_device == "NPU"
    assert args.config is None
    assert args.disparity_upsample == "adaptive"
    assert args.disparity_upsample_sigma == 2.0
    validate_backend_args(args)


def test_image_parser_exposes_only_public_backends():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--backend",
                "tensorrt",
                "--left",
                "left.png",
                "--right",
                "right.png",
            ]
        )


def test_image_parser_removes_runtime_graph_preprocessing_flag():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--backend",
                "openvino",
                "--openvino-fused-preprocess",
                "--left",
                "left.png",
                "--right",
                "right.png",
            ]
        )


def test_image_parser_accepts_shared_disparity_restoration_controls():
    args = parse_args(
        [
            "--left",
            "left.png",
            "--right",
            "right.png",
            "--disparity-upsample",
            "nearest",
            "--disparity-upsample-sigma",
            "3.5",
        ]
    )

    assert args.disparity_upsample == "nearest"
    assert args.disparity_upsample_sigma == 3.5


class _IdentityDisparity(torch.nn.Module):
    def forward(self, sample):
        return {"disp_pred": sample["left"][:, :1]}


class _PaddedCPUPreprocessor:
    def prepare(self, left, right):
        grid = torch.arange(24, dtype=torch.float32).reshape(1, 1, 4, 6)
        tensor = grid.expand(1, 3, 4, 6)
        return {"left": tensor, "right": tensor, "pad": [2, 3]}, (2, 3)


def test_pytorch_backend_cpu_timing_and_exact_padding_crop():
    backend = PyTorchBackend(
        _IdentityDisparity(),
        torch.device("cpu"),
        use_amp=True,
        amp_dtype=torch.float16,
        max_disp=192.0,
    )
    frame = {
        "left": np.zeros((2, 3, 3), dtype=np.uint8),
        "right": np.zeros((2, 3, 3), dtype=np.uint8),
    }

    result = backend.forward(
        frame,
        _PaddedCPUPreprocessor(),
        frame_id=0,
    )

    np.testing.assert_array_equal(
        result.native_disparity,
        np.array([[12.0, 13.0, 14.0], [18.0, 19.0, 20.0]], dtype=np.float32),
    )
    assert result.model_ms >= 0.0


@pytest.mark.parametrize(
    ("pad_name", "expected_top"),
    [("RightTopPad", 2), ("RightBottomPad", 0)],
)
def test_cpu_preprocessor_preserves_padding_mode_and_edge_values(
    pad_name,
    expected_top,
):
    image = np.array(
        [
            [[10, 20, 30], [40, 50, 60]],
            [[70, 80, 90], [100, 110, 120]],
        ],
        dtype=np.uint8,
    )
    preprocessor = Preprocessor(
        torch.device("cpu"),
        transform_config=[
            {"NAME": pad_name, "SIZE": [4, 3]},
            {"NAME": "NormalizeImage", "MEAN": [0, 0, 0], "STD": [1, 1, 1]},
        ],
        target_height=4,
        target_width=3,
    )

    sample, original_size = preprocessor.prepare(image, image)
    actual = sample["left"].squeeze(0).permute(1, 2, 0).numpy() * 255.0
    expected = image[..., ::-1]
    expected = np.pad(
        expected,
        ((expected_top, 2 - expected_top), (0, 1), (0, 0)),
        mode="edge",
    )

    assert original_size == (2, 2)
    assert sample["pad"] == [expected_top, 1]
    np.testing.assert_allclose(actual, expected, atol=1e-5)


def test_pytorch_external_size_does_not_expose_internal_alignment_padding():
    cfg = {"INFERENCE": {"SIZE": None, "DIVISIBLE_BY": 32}}

    external_size = _external_size_from_config(cfg, 720, 1280)
    padded_size = _padded_size_from_config(cfg, *external_size)

    assert external_size == (720, 1280)
    assert padded_size == (736, 1280)


class _FakeOpenVINOBackend:
    max_disp = 192.0

    def forward(self, frame, preprocessor, frame_id):
        height, width = frame["left"].shape[:2]
        disparity = np.ones((height, width), dtype=np.float32)
        return BackendResult(disparity, 5.0)


def test_run_once_scales_openvino_max_disp_after_resize():
    setup = RuntimeSetup(
        backend_name="openvino",
        backend=_FakeOpenVINOBackend(),
        preprocessor=None,
        target_h=2,
        target_w=4,
        spatial_transform=SpatialTransform(4, 8, 2, 4),
        postprocessor=DisparityPostprocessor(method="bilinear"),
    )
    image = np.zeros((4, 8, 3), dtype=np.uint8)

    disparity, model_ms, max_disp = run_once(setup, image, image)

    assert disparity.shape == (4, 8)
    assert model_ms == 5.0
    assert max_disp == 384.0


def test_openvino_input_is_strict_exact_uint8_nhwc_bgr():
    backend = OpenVINOBackend.__new__(OpenVINOBackend)
    backend.target_h = 2
    backend.target_w = 3
    image = np.zeros((2, 3, 3), dtype=np.uint8)

    prepared = backend._prepare(image)

    assert prepared.shape == (1, 2, 3, 3)
    assert prepared.dtype == np.uint8
    assert prepared.flags.c_contiguous
    assert np.shares_memory(prepared, image)

    with pytest.raises(ValueError, match="exact HWC shape"):
        backend._prepare(np.zeros((3, 2, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="uint8 BGR"):
        backend._prepare(np.zeros((2, 3, 3), dtype=np.float32))


class _RecordingLatencyMonitor:
    def __init__(self):
        self.started = []
        self.ended = []

    def record_inference_start(self, frame_id):
        self.started.append(frame_id)

    def record_inference_end(self, frame_id):
        self.ended.append(frame_id)


class _RecordingBackend:
    max_disp = 192.0

    def __init__(self):
        self.left = None

    def forward(self, frame, preprocessor, frame_id):
        self.left = frame["left"]
        return BackendResult(
            np.ones(frame["left"].shape[:2], dtype=np.float32),
            1.25,
        )


def test_async_inference_returns_named_native_result_without_double_resize():
    backend = _RecordingBackend()
    monitor = _RecordingLatencyMonitor()
    transform = SpatialTransform(4, 8, 2, 4)
    model_image = np.zeros((2, 4, 3), dtype=np.uint8)
    async_inference = AsyncInference(
        backend,
        preprocessor=None,
        latency_monitor=monitor,
        spatial_transform=transform,
    )
    try:
        async_inference.submit(
            {"left": model_image, "right": model_image},
            frame_id=7,
            source_size=(4, 8),
        )
        result = async_inference.get_result(block_timeout=1.0)
    finally:
        async_inference.stop()

    assert isinstance(result, AsyncInferenceResult)
    assert result.frame_id == 7
    assert result.source_size == (4, 8)
    assert result.spatial_transform is transform
    assert result.native_disparity.shape == (2, 4)
    assert result.model_ms == 1.25
    assert backend.left is model_image
    assert monitor.started == [7]
    assert monitor.ended == [7]
