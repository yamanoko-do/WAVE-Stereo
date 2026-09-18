from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from wavestereo.config import load_config
from wavestereo.inference.postprocessing import (
    DisparityPostprocessor,
    SpatialTransform,
)
from wavestereo.inference.preprocess import normalization_from_config
from wavestereo.visual.async_inference import (
    OpenVINOBackend,
    PyTorchBackend,
)
from wavestereo.visual.core import Preprocessor


BACKEND_CHOICES = ("pytorch", "openvino")
DEFAULT_CONFIG = "cfgs/wavestereo.yaml"
DEFAULT_OPENVINO_CACHE_DIR = "outputs/openvino_cache"


@dataclass
class RuntimeSetup:
    backend_name: str
    backend: Any
    preprocessor: Any | None
    target_h: int
    target_w: int
    spatial_transform: SpatialTransform
    postprocessor: DisparityPostprocessor


def add_backend_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the backend options shared by image and camera inference."""
    parser.add_argument(
        "--backend",
        choices=BACKEND_CHOICES,
        default="pytorch",
        help="Inference runtime (Intel GPU/NPU use the OpenVINO backend)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help=f"PyTorch config (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument("--weights", default=None, help="PyTorch checkpoint")
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Exact exported model bundle for OpenVINO",
    )
    parser.add_argument(
        "--openvino-device",
        default=None,
        help="OpenVINO device, e.g. GPU.0 for Intel GPU or NPU",
    )
    parser.add_argument(
        "--openvino-cache-dir",
        default=None,
        help=(
            "OpenVINO compiled-model cache directory "
            f"(default: {DEFAULT_OPENVINO_CACHE_DIR})"
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device for PyTorch, e.g. cuda:0 or cpu",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use the PyTorch debug forward path when available",
    )


def add_postprocessing_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the source-resolution disparity restoration controls."""
    parser.add_argument(
        "--disparity-upsample",
        choices=("nearest", "bilinear", "adaptive"),
        default="adaptive",
        help=(
            "Method used to restore disparity to the source resolution; "
            "adaptive preserves depth discontinuities"
        ),
    )
    parser.add_argument(
        "--disparity-upsample-sigma",
        type=float,
        default=2.0,
        help=(
            "Disparity consistency sigma in restored-resolution pixels used "
            "by --disparity-upsample adaptive"
        ),
    )


def validate_backend_args(args: argparse.Namespace) -> None:
    """Validate backend-specific CLI options without loading a model."""
    if args.backend not in BACKEND_CHOICES:
        supported = ", ".join(BACKEND_CHOICES)
        raise SystemExit(
            f"Unsupported public backend {args.backend!r}; choose: {supported}"
        )
    is_openvino = args.backend == "openvino"
    if is_openvino and not args.model_dir:
        raise SystemExit("--backend openvino requires --model-dir")
    if not is_openvino and args.model_dir:
        raise SystemExit("--model-dir is only valid with the OpenVINO backend")
    if is_openvino and args.weights:
        raise SystemExit("--weights is only valid with the PyTorch backend")
    if is_openvino and args.config:
        raise SystemExit("--config is only valid with the PyTorch backend")
    if is_openvino and args.debug:
        raise SystemExit("--debug is only valid with the PyTorch backend")
    if is_openvino and not args.openvino_device:
        raise SystemExit("--backend openvino requires --openvino-device")
    if not is_openvino and args.openvino_device:
        raise SystemExit(
            "--openvino-device is only valid with the OpenVINO backend"
        )
    if is_openvino and args.device:
        raise SystemExit("--device is not valid with the OpenVINO backend")
    if not is_openvino and args.openvino_cache_dir:
        raise SystemExit(
            "--openvino-cache-dir is only valid with the OpenVINO backend"
        )


def _resolve_torch_device(args: argparse.Namespace) -> torch.device:
    if args.backend == "openvino":
        return torch.device("cpu")

    device = torch.device(
        args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; falling back to CPU")
        device = torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def _ceil_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _external_size_from_config(
    cfgs: Any, orig_h: int, orig_w: int
) -> tuple[int, int]:
    infer_cfg = cfgs.get("INFERENCE", {})
    size = infer_cfg.get("SIZE")
    if size:
        return int(size[0]), int(size[1])
    return orig_h, orig_w


def _padded_size_from_config(
    cfgs: Any, external_h: int, external_w: int
) -> tuple[int, int]:
    infer_cfg = cfgs.get("INFERENCE", {})
    divisible_by = int(infer_cfg.get("DIVISIBLE_BY", 32))
    if divisible_by <= 0:
        raise ValueError("INFERENCE.DIVISIBLE_BY must be positive")
    return (
        _ceil_to_multiple(external_h, divisible_by),
        _ceil_to_multiple(external_w, divisible_by),
    )


def _transform_config(cfgs: Any, target_h: int, target_w: int) -> list[dict[str, Any]]:
    infer_cfg = cfgs.get("INFERENCE", {})
    mean, std = normalization_from_config(infer_cfg)
    pad_mode = str(infer_cfg.get("PAD_MODE", "right_top")).lower().replace("-", "_")
    if pad_mode not in ("right_top", "righttoppad", "right_bottom", "rightbottompad"):
        raise ValueError(
            "Shared image/camera inference supports PAD_MODE right_top or right_bottom, "
            f"got {pad_mode!r}"
        )
    pad_name = (
        "RightTopPad"
        if pad_mode in ("right_top", "righttoppad")
        else "RightBottomPad"
    )
    return [
        {"NAME": pad_name, "SIZE": [target_h, target_w]},
        {
            "NAME": "NormalizeImage",
            "MEAN": mean,
            "STD": std,
        },
    ]


def build_runtime(
    args: argparse.Namespace,
    *,
    orig_h: int,
    orig_w: int,
    allow_resize: bool = False,
) -> RuntimeSetup:
    """Build the selected backend for one known input image size."""
    device = _resolve_torch_device(args)

    if args.backend == "openvino":
        backend = OpenVINOBackend(
            args.model_dir,
            device=args.openvino_device,
            cache_dir=args.openvino_cache_dir or DEFAULT_OPENVINO_CACHE_DIR,
        )
        target_h, target_w = backend.target_h, backend.target_w
        preprocessor = None
    else:
        cfgs = load_config(args.config or DEFAULT_CONFIG)
        target_h, target_w = _external_size_from_config(cfgs, orig_h, orig_w)
        padded_h, padded_w = _padded_size_from_config(
            cfgs, target_h, target_w
        )
        transform_config = _transform_config(cfgs, padded_h, padded_w)
        preprocessor = Preprocessor(
            device,
            transform_config=transform_config,
            target_height=padded_h,
            target_width=padded_w,
        )
        backend = PyTorchBackend.build_from_config(args, cfgs, device)

    spatial_transform = SpatialTransform(
        source_height=orig_h,
        source_width=orig_w,
        model_height=target_h,
        model_width=target_w,
    )
    postprocessor = DisparityPostprocessor(
        method=getattr(args, "disparity_upsample", "adaptive"),
        sigma=getattr(args, "disparity_upsample_sigma", 2.0),
    )

    return RuntimeSetup(
        backend_name=args.backend,
        backend=backend,
        preprocessor=preprocessor,
        target_h=target_h,
        target_w=target_w,
        spatial_transform=spatial_transform,
        postprocessor=postprocessor,
    )


def run_once(
    setup: RuntimeSetup,
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    *,
    frame_id: int = 0,
) -> tuple[np.ndarray, float, float]:
    """Run one synchronous inference and return disparity, model ms and max disparity."""
    if left_bgr.shape != right_bgr.shape:
        raise ValueError(
            f"Left/right image shapes differ: {left_bgr.shape} vs {right_bgr.shape}"
        )
    if left_bgr.ndim != 3 or left_bgr.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 images, got {left_bgr.shape}")

    model_frame = setup.spatial_transform.prepare_stereo(
        {"left": left_bgr, "right": right_bgr}
    )
    with torch.no_grad():
        backend_result = setup.backend.forward(
            model_frame,
            setup.preprocessor,
            frame_id,
        )
    if backend_result.native_disparity is None:
        raise RuntimeError(f"{setup.backend_name} backend returned no disparity")

    processed = setup.postprocessor.restore(
        backend_result.native_disparity,
        setup.spatial_transform,
        setup.backend.max_disp,
    )
    return (
        processed.disparity,
        float(backend_result.model_ms),
        processed.effective_max_disp,
    )
