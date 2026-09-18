#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from wavestereo.inference.runtime import (
    add_backend_arguments,
    add_postprocessing_arguments,
    build_runtime,
    run_once,
    validate_backend_args,
)
from wavestereo.inference.visualization import save_outputs


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run WAVE-Stereo inference on one stereo image pair"
    )
    add_backend_arguments(parser)
    add_postprocessing_arguments(parser)
    parser.add_argument("--left", required=True, help="Path to left image")
    parser.add_argument("--right", required=True, help="Path to right image")
    parser.add_argument("--output", default="outputs/disparity", help="Output file prefix")
    return parser.parse_args(argv)


def _load_bgr_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def main(argv=None):
    args = parse_args(argv)
    validate_backend_args(args)

    left = _load_bgr_image(args.left)
    right = _load_bgr_image(args.right)
    if left.shape != right.shape:
        raise ValueError(
            f"Left/right image shapes differ: {left.shape} vs {right.shape}"
        )
    orig_h, orig_w = left.shape[:2]

    runtime = build_runtime(
        args,
        orig_h=orig_h,
        orig_w=orig_w,
    )
    if orig_w * runtime.target_h != runtime.target_w * orig_h:
        print(
            "[WARNING] Source and model aspect ratios differ; input images "
            "will be stretched directly (no letterbox)"
        )
    disparity, inference_ms, effective_max_disp = run_once(runtime, left, right)

    backend_label = (
        args.openvino_device
        if args.backend == "openvino"
        else "PyTorch"
    )
    print(f"[INFO] Input image: {orig_w}x{orig_h}")
    print(
        f"[INFO] Inference resolution: "
        f"{runtime.target_w}x{runtime.target_h}"
    )
    print(
        f"[INFO] {backend_label} inference time: {inference_ms:.2f} ms "
        f"({inference_ms / 1000.0:.3f} s)"
    )

    output = Path(args.output)
    save_outputs(
        output,
        disparity,
        max_disp=effective_max_disp,
        save_npy=True,
    )
    print(
        f"[INFO] Disparity range: {float(disparity.min()):.3f} .. "
        f"{float(disparity.max()):.3f}"
    )
    print(f"[INFO] Saved outputs with prefix: {output}")


if __name__ == "__main__":
    main()
