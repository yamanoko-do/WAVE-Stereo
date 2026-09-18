#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np


_FONT_SUFFIXES = {".otf", ".ttc", ".ttf"}


def _directory_has_fonts(directory: Path) -> bool:
    try:
        return any(
            entry.is_file() and entry.suffix.lower() in _FONT_SUFFIXES
            for entry in directory.iterdir()
        )
    except OSError:
        return False


def _find_system_qt_font_dir() -> Path | None:
    fc_match = shutil.which("fc-match")
    if fc_match:
        try:
            result = subprocess.run(
                [fc_match, "--format=%{file}\\n", "sans-serif"],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
            for line in result.stdout.splitlines():
                font_file = Path(line.strip()).expanduser()
                if font_file.is_file() and font_file.suffix.lower() in _FONT_SUFFIXES:
                    return font_file.parent
        except (OSError, subprocess.SubprocessError):
            pass

    font_roots = (
        Path(sys.prefix) / "share" / "fonts",
        Path(sys.prefix) / "lib" / "fonts",
        Path("/usr/local/share/fonts"),
        Path("/usr/share/fonts"),
        Path.home() / ".local" / "share" / "fonts",
        Path.home() / ".fonts",
    )
    for root in font_roots:
        if not root.is_dir():
            continue
        try:
            for font_file in root.rglob("*"):
                if font_file.is_file() and font_file.suffix.lower() in _FONT_SUFFIXES:
                    return font_file.parent
        except OSError:
            continue
    return None


def _configure_qt_font_dir() -> None:
    """Replace OpenCV/Conda missing font paths with a real system font directory."""
    if not sys.platform.startswith("linux"):
        return

    configured_dir = os.environ.get("QT_QPA_FONTDIR")
    if configured_dir and _directory_has_fonts(Path(configured_dir).expanduser()):
        return

    system_font_dir = _find_system_qt_font_dir()
    if system_font_dir is not None:
        os.environ["QT_QPA_FONTDIR"] = str(system_font_dir)
    else:
        os.environ.pop("QT_QPA_FONTDIR", None)


_configure_qt_font_dir()

from wavestereo.inference.runtime import (
    add_backend_arguments,
    add_postprocessing_arguments,
    build_runtime,
    validate_backend_args,
)
from wavestereo.visual.async_inference import (
    AsyncInference,
    create_visualization_from_raw,
)
from wavestereo.visual.core import (
    AsyncCamera,
    LatencyMonitor,
    display_latency_overlay,
    load_camera_params,
    load_camera_resolution,
    print_latency_stats,
)
from wavestereo.visual.pointcloud import O3DVisualizer, PointCloudProcessor
from wavestereo.visual.video_source import StereoVideoSource
from wavestereo.visual.visualization import create_disparity_colormap


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run WAVE-Stereo live camera/video inference")
    add_backend_arguments(parser)
    parser.add_argument("--cam-file", default="cfgs/camera/pxyzd435/")
    parser.add_argument("--video", default=None, help="Side-by-side stereo video; replaces live camera")
    parser.add_argument("--skip-frames", type=int, default=1)
    parser.add_argument("--display-stride", type=int, default=1)
    parser.add_argument("--zfar", type=float, default=9.0)
    parser.add_argument("--subsample", type=int, default=2)
    add_postprocessing_arguments(parser)
    parser.add_argument(
        "--camera-preprocess",
        choices=("auto", "legacy"),
        default="auto",
        help=(
            "Camera rectification policy. auto combines rectification and "
            "resize into one target-resolution remap when calibration maps "
            "are available; legacy rectifies at source resolution first"
        ),
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()
    validate_backend_args(args)
    if args.disparity_upsample_sigma <= 0:
        raise SystemExit("--disparity-upsample-sigma must be positive")

    K, baseline, vidpid_content = load_camera_params(args.cam_file)
    calibration_w, calibration_h = load_camera_resolution(args.cam_file)
    video_src = None
    if args.video:
        video_src = StereoVideoSource(args.video, loop=True)
        orig_w, orig_h = video_src.half_w, video_src.frame_height
        if (orig_w, orig_h) != (calibration_w, calibration_h):
            scale_x = orig_w / calibration_w
            scale_y = orig_h / calibration_h
            K = K.copy()
            K[0, 0] *= scale_x
            K[0, 2] *= scale_x
            K[1, 1] *= scale_y
            K[1, 2] *= scale_y
            print(
                "[WARNING] Video per-eye resolution differs from the camera "
                "calibration; intrinsic parameters were scaled to the video size"
            )
    else:
        orig_w, orig_h = calibration_w, calibration_h
    latency_monitor = LatencyMonitor()
    prefix = (
        f"[{args.openvino_device}] " if args.backend == "openvino"
        else ""
    )

    runtime = build_runtime(
        args,
        orig_h=orig_h,
        orig_w=orig_w,
        allow_resize=True,
    )
    backend = runtime.backend
    preprocessor = runtime.preprocessor
    target_h, target_w = runtime.target_h, runtime.target_w
    spatial_transform = runtime.spatial_transform
    disparity_postprocessor = runtime.postprocessor
    async_infer = AsyncInference(backend, preprocessor, latency_monitor,
                                  input_queue_size=2, output_queue_size=2,
                                  spatial_transform=spatial_transform)

    source_resize_for_inference = spatial_transform.requires_resize
    camera_inference_resolution = (target_w, target_h)

    if args.video:
        async_cam = AsyncCamera(max_queue_size=2, map_dir=args.cam_file, vid_pid=vidpid_content,
                                latency_monitor=latency_monitor, camera=video_src,
                                inference_resolution=camera_inference_resolution,
                                preprocessing_mode=args.camera_preprocess)
    else:
        async_cam = AsyncCamera(max_queue_size=200, map_dir=args.cam_file, vid_pid=vidpid_content,
                                latency_monitor=latency_monitor,
                                camera_resolution=(orig_w, orig_h),
                                inference_resolution=camera_inference_resolution,
                                preprocessing_mode=args.camera_preprocess)
    async_cam.start()

    print("[INFO] Waiting for first frame to validate camera resolution...")
    frame, frame_id = async_cam.get_frame()
    while frame is None:
        time.sleep(0.05)
        frame, frame_id = async_cam.get_frame()
    actual_h, actual_w = frame["left"].shape[:2]
    expected_capture_size = camera_inference_resolution
    if (actual_w, actual_h) != expected_capture_size:
        async_cam.stop()
        async_infer.stop()
        raise ValueError(
            f"Camera frame resolution mismatch: expected "
            f"{expected_capture_size[0]}x{expected_capture_size[1]}, "
            f"actual frame={actual_w}x{actual_h}"
        )

    source_label = "video per-eye" if args.video else "camera config"
    print(f"[INFO] Source resolution ({source_label}): {orig_w}x{orig_h}")
    if source_resize_for_inference:
        aspect_note = ""
        if orig_w * target_h != target_w * orig_h:
            aspect_note = " The aspect ratio will change."
        print(
            f"[WARNING] Camera resolution {orig_w}x{orig_h} does not match "
            f"the inference input {target_w}x{target_h}; camera frames will be "
            f"resized before inference.{aspect_note}"
        )
        method = (
            "auto rectification + resize"
            if async_cam.fused_rectify_resize
            else (
                "legacy rectification then resize"
                if args.camera_preprocess == "legacy"
                else "capture-thread resize"
            )
        )
        print(f"[INFO] Camera preprocessing method: {method}")
    print(f"[INFO] Inference resolution: {target_w}x{target_h}")
    if source_resize_for_inference:
        print(f"[INFO] Disparity upsampling: {args.disparity_upsample}")
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

            inference_result = async_infer.get_result(block_timeout=0.001)
            if inference_result is None:
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

            result_id = inference_result.frame_id
            model_gpu_ms = inference_result.model_ms
            latency_monitor.record_result_fetched(result_id)
            restored = disparity_postprocessor.restore(
                inference_result.native_disparity,
                inference_result.spatial_transform,
                max_disp=backend.max_disp,
            )
            disp_raw_cpu = restored.disparity

            left_img = None
            for fid, l_img, _ in frame_buffer:
                if fid == result_id:
                    left_img = l_img
                    break

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
                if left_img.shape[:2] != disp_raw_cpu.shape[:2]:
                    left_img = cv2.resize(
                        left_img,
                        (disp_raw_cpu.shape[1], disp_raw_cpu.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
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
