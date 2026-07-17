#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch

from wavestereo.config import load_config
from wavestereo.visual.async_inference import (
    AsyncInference,
    PyTorchBackend,
    TensorRTBackend,
    create_visualization_from_raw,
)
from wavestereo.visual.core import (
    AsyncCamera,
    LatencyMonitor,
    Preprocessor,
    display_latency_overlay,
    load_camera_params,
    load_camera_resolution,
    print_latency_stats,
)
from wavestereo.visual.pointcloud import O3DVisualizer, PointCloudProcessor
from wavestereo.visual.video_source import StereoVideoSource
from wavestereo.visual.visualization import check_and_fix_invalid, create_disparity_colormap


def parse_args():
    parser = argparse.ArgumentParser(description="Run WAVE-Stereo live camera/video inference")
    parser.add_argument("--config", "--cfg_file", dest="config", default="cfgs/wavestereo.yaml")
    parser.add_argument("--weights", "--pretrained_model", dest="weights", default=None)
    parser.add_argument("--onnx-dir", "--onnx_dir", dest="onnx_dir", default=None)
    parser.add_argument("--trt", action="store_true", help="Use TensorRT backend; requires --onnx-dir")
    parser.add_argument("--cam_file", "--cam-file", dest="cam_file", default="cfgs/camera/pxyzd435/")
    parser.add_argument("--video", default=None, help="Side-by-side stereo video; replaces live camera")
    parser.add_argument("--device", default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--skip-frames", "--skip_frames", dest="skip_frames", type=int, default=1)
    parser.add_argument("--display-stride", "--display_stride", dest="display_stride", type=int, default=1)
    parser.add_argument("--zfar", type=float, default=9.0)
    parser.add_argument("--subsample", type=int, default=2)
    return parser.parse_args()


def _ceil_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _infer_target_size_from_config(cfgs, orig_h: int, orig_w: int):
    infer_cfg = cfgs.get("INFERENCE", {})
    size = infer_cfg.get("SIZE", None)
    if size:
        return int(size[0]), int(size[1])
    by = int(infer_cfg.get("DIVISIBLE_BY", 32))
    return _ceil_to_multiple(orig_h, by), _ceil_to_multiple(orig_w, by)


def _make_transform_config(cfgs, target_h: int, target_w: int):
    infer_cfg = cfgs.get("INFERENCE", {})
    pad_mode = str(infer_cfg.get("PAD_MODE", "right_top")).lower().replace("-", "_")
    pad_name = "RightTopPad" if pad_mode in ("right_top", "righttoppad") else "RightBottomPad"
    return [
        {"NAME": pad_name, "SIZE": [target_h, target_w]},
        {"NAME": "NormalizeImage", "MEAN": infer_cfg.get("MEAN", [0.485, 0.456, 0.406]),
         "STD": infer_cfg.get("STD", [0.229, 0.224, 0.225])},
    ]


def _load_trt_image_size(onnx_dir: str):
    import yaml

    cfg_path = Path(onnx_dir) / "onnx.yaml"
    if not cfg_path.exists():
        return 736, 1280
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    size = cfg.get("image_size", [736, 1280])
    return int(size[0]), int(size[1])


@torch.no_grad()
def main():
    args = parse_args()
    if args.trt and not args.onnx_dir:
        raise SystemExit("--trt requires --onnx-dir")

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.set_device(device)

    K, baseline, vidpid_content = load_camera_params(args.cam_file)
    orig_w, orig_h = load_camera_resolution(args.cam_file)
    latency_monitor = LatencyMonitor()
    prefix = "[TRT] " if args.trt else ""

    if args.trt:
        target_h, target_w = _load_trt_image_size(args.onnx_dir)
        if target_h < orig_h or target_w < orig_w:
            raise ValueError(
                f"TensorRT engine input {target_w}x{target_h} is smaller than camera resolution {orig_w}x{orig_h}"
            )
        crop_pad = max(0, target_h - orig_h)
        backend = TensorRTBackend.load_from_onnx_dir(args.onnx_dir, crop_pad=crop_pad)
        transform_config = [
            {"NAME": "RightTopPad", "SIZE": [target_h, target_w]},
            {"NAME": "NormalizeImage", "MEAN": backend.trt_runner.cfg.get("normalize_mean", [0.485, 0.456, 0.406]),
             "STD": backend.trt_runner.cfg.get("normalize_std", [0.229, 0.224, 0.225])},
        ]
    else:
        cfgs = load_config(args.config)
        if args.weights:
            cfgs.MODEL.PRETRAINED_MODEL = args.weights
        target_h, target_w = _infer_target_size_from_config(cfgs, orig_h, orig_w)
        if target_h < orig_h or target_w < orig_w:
            raise ValueError(
                f"Inference target {target_w}x{target_h} is smaller than camera resolution {orig_w}x{orig_h}"
            )
        crop_pad = max(0, target_h - orig_h)
        transform_config = _make_transform_config(cfgs, target_h, target_w)
        logger = logging.getLogger("infercam")
        logging.basicConfig(level=logging.INFO)
        backend = PyTorchBackend.build_from_config(
            args, cfgs, device, logger, crop_pad=crop_pad
        )

    preprocessor = Preprocessor(device, transform_config=transform_config,
                                target_height=target_h, target_width=target_w)
    async_infer = AsyncInference(backend, preprocessor, latency_monitor,
                                  input_queue_size=2, output_queue_size=2)

    if args.video:
        video_src = StereoVideoSource(args.video, loop=True)
        async_cam = AsyncCamera(max_queue_size=2, map_dir=args.cam_file, vid_pid=vidpid_content,
                                latency_monitor=latency_monitor, camera=video_src)
    else:
        async_cam = AsyncCamera(max_queue_size=200, map_dir=args.cam_file, vid_pid=vidpid_content,
                                latency_monitor=latency_monitor,
                                camera_resolution=(orig_w, orig_h))
    async_cam.start()

    print("[INFO] Waiting for first frame to validate camera resolution...")
    frame, frame_id = async_cam.get_frame()
    while frame is None:
        time.sleep(0.05)
        frame, frame_id = async_cam.get_frame()
    actual_h, actual_w = frame["left"].shape[:2]
    if (actual_w, actual_h) != (orig_w, orig_h):
        async_cam.stop()
        async_infer.stop()
        raise ValueError(
            f"Camera frame resolution mismatch: resolution.txt={orig_w}x{orig_h}, actual frame={actual_w}x{actual_h}"
        )

    print(f"[INFO] Original resolution from camera config: {orig_w}x{orig_h}")
    print(f"[INFO] Inference resolution: {target_w}x{target_h}, crop_pad={crop_pad}")
    print("[INFO] Press 'q' or Esc to quit, 'd' to toggle debug")

    fps_intervals = deque(maxlen=30)
    model_gpu_times = deque(maxlen=30)
    frame_buffer = deque(maxlen=120)
    last_frame_time = time.time()
    fps_start_time = time.time()
    skip_counter = 0
    display_counter = 0
    last_submitted_frame_id = -1
    total_frames = 0
    show_debug = args.debug
    view_mode = "2d"
    pc_processor = PointCloudProcessor(K, baseline, args.zfar, subsample=args.subsample, flip_y_z=True)
    o3d_vis = O3DVisualizer(window_name="WAVE-Stereo 3D Point Cloud", width=1280, height=720)
    last_v_toggle = 0.0

    try:
        while True:
            frame, frame_id = async_cam.get_frame()
            if frame is not None and frame_id != last_submitted_frame_id:
                last_submitted_frame_id = frame_id
                skip_counter += 1
                if skip_counter % max(1, args.skip_frames) == 0:
                    frame_buffer.append((frame_id, frame.get("left"), frame.get("right")))
                    latency_monitor.record_submit_to_inference_queue(frame_id)
                    async_infer.submit(frame, frame_id, (orig_h, orig_w))

            raw_result = async_infer.get_result(block_timeout=0.001)
            if raw_result is None or raw_result[1] is None:
                if view_mode == "3d" and o3d_vis.is_initialized and not o3d_vis.spin_once():
                    view_mode = "2d"
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("d"):
                    show_debug = not show_debug
                if key == ord("v") and time.time() - last_v_toggle > 0.35:
                    last_v_toggle = time.time()
                    view_mode = "3d" if view_mode == "2d" else "2d"
                    print(f"[INFO] View mode: {'3D PointCloud' if view_mode == '3d' else '2D Disparity'}")
                    if view_mode == "3d" and not o3d_vis.is_initialized:
                        o3d_vis.init_window()
                    elif view_mode == "2d":
                        o3d_vis.close()
                continue

            _, disp_raw_cpu, _, _, result_id, orig_size, model_gpu_ms = raw_result
            latency_monitor.record_result_fetched(result_id)
            disp_raw_cpu = check_and_fix_invalid(disp_raw_cpu, "disparity")

            left_img = None
            for fid, l_img, _ in frame_buffer:
                if fid == result_id:
                    left_img = l_img
                    break

            if orig_size:
                oh, ow = orig_size
                disp_raw_cpu = disp_raw_cpu[:oh, :ow]

            disp_vis_cpu = create_visualization_from_raw(disp_raw_cpu)
            disp_color = create_disparity_colormap(disp_vis_cpu)

            total_frames += 1
            now = time.time()
            interval = now - last_frame_time
            if interval > 0:
                fps_intervals.append(interval)
            avg_fps = len(fps_intervals) / sum(fps_intervals) if fps_intervals else 0.0
            if model_gpu_ms is not None:
                model_gpu_times.append(model_gpu_ms)
            model_fps = 1000.0 * len(model_gpu_times) / sum(model_gpu_times) if model_gpu_times else 0.0
            last_frame_time = now

            if view_mode == "3d" and left_img is not None:
                pc_processor.submit(disp_raw_cpu, left_img, result_id)

            if view_mode == "3d":
                points, colors, pixel_map, _ = pc_processor.get_result()
                if points is not None and colors is not None:
                    o3d_vis.set_pointcloud_data(
                        points,
                        colors,
                        pixel_map=pixel_map,
                        disp_map=disp_raw_cpu,
                        fx=K[0, 0],
                        baseline=baseline,
                    )
                if o3d_vis.is_initialized and not o3d_vis.spin_once():
                    view_mode = "2d"

            latency_stats = latency_monitor.get_latency_stats()
            mode_str = "3D PointCloud" if view_mode == "3d" else "2D Disparity"
            disp_color = display_latency_overlay(
                disp_color, result_id, latency_stats, avg_fps,
                max(1, args.skip_frames), mode_str, prefix,
                model_fps=model_fps,
            )
            if show_debug and result_id % 30 == 0 and latency_stats:
                print_latency_stats(result_id, latency_stats, avg_fps, prefix, model_fps=model_fps)

            display_counter += 1
            if display_counter % max(1, args.display_stride) == 0:
                cv2.imshow(f"WAVE-Stereo InferCam {prefix}", disp_color)
                latency_monitor.record_display(result_id)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("d"):
                show_debug = not show_debug
                print(f"[INFO] Debug display: {'ON' if show_debug else 'OFF'}")
            if key == ord("v") and time.time() - last_v_toggle > 0.35:
                last_v_toggle = time.time()
                view_mode = "3d" if view_mode == "2d" else "2d"
                print(f"[INFO] View mode: {'3D PointCloud' if view_mode == '3d' else '2D Disparity'}")
                if view_mode == "3d" and not o3d_vis.is_initialized:
                    o3d_vis.init_window()
                elif view_mode == "2d":
                    o3d_vis.close()

    except KeyboardInterrupt:
        print("\n[INFO] User interrupted")
    finally:
        async_cam.stop()
        async_infer.stop()
        pc_processor.stop()
        o3d_vis.close()
        cv2.destroyAllWindows()
        elapsed = time.time() - fps_start_time
        if elapsed > 0:
            print(f"\nFinal average FPS: {total_frames / elapsed:.2f} "
                  f"(total frames: {total_frames}, total time: {elapsed:.2f}s)")


if __name__ == "__main__":
    main()
