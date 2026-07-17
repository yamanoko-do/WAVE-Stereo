"""
Core Module - Camera capture, latency monitoring, and preprocessing
"""
import os
import time
import threading
import queue
from collections import deque
from typing import Tuple, Optional, Dict, Any

import cv2
import numpy as np
import torch



class LatencyMonitor:
    """
    Latency monitoring class - tracks timing through each pipeline stage
    """
    def __init__(self, window_size: int = 30):
        self.frame_timestamps: Dict[int, Dict[str, Optional[float]]] = {}
        self.latencies = deque(maxlen=window_size)
        self.stage_latencies = {
            'capture_to_queue': deque(maxlen=window_size),
            'queue_to_infer': deque(maxlen=window_size),
            'inference_proc': deque(maxlen=window_size),
            'output_queue_wait': deque(maxlen=window_size),
            'real_post_proc': deque(maxlen=window_size)
        }
        self.lock = threading.Lock()

    def record_frame_capture_start(self, frame_id: int, start_time: float):
        """Record when frame capture starts"""
        with self.lock:
            self.frame_timestamps[frame_id] = {
                'capture': start_time,
                'queue_in': None,
                'inference_start': None,
                'inference_end': None,
                'fetched': None,
                'display': None
            }

    def record_capture_complete(self, frame_id: int, capture_end_time: float):
        """Record when capture is complete (Stage 1: Capture)"""
        with self.lock:
            if frame_id in self.frame_timestamps:
                capture_time = self.frame_timestamps[frame_id]['capture']
                if capture_time:
                    self.stage_latencies['capture_to_queue'].append((capture_end_time - capture_time) * 1000)

    def record_submit_to_inference_queue(self, frame_id: int):
        """Record when frame is submitted to inference input queue (start of Queue Wait)"""
        with self.lock:
            if frame_id in self.frame_timestamps:
                self.frame_timestamps[frame_id]['queue_in'] = time.time()

    def record_queue_in(self, frame_id: int):
        """DEPRECATED: Kept for compatibility, use record_capture_complete instead"""
        pass

    def record_inference_start(self, frame_id: int):
        """Record when inference starts (end of Queue Wait, start of Inference)"""
        with self.lock:
            if frame_id in self.frame_timestamps:
                ts = time.time()
                self.frame_timestamps[frame_id]['inference_start'] = ts
                queue_in_time = self.frame_timestamps[frame_id]['queue_in']
                if queue_in_time:
                    self.stage_latencies['queue_to_infer'].append((ts - queue_in_time) * 1000)

    def record_inference_end(self, frame_id: int):
        """Record when inference ends"""
        with self.lock:
            if frame_id in self.frame_timestamps:
                ts = time.time()
                self.frame_timestamps[frame_id]['inference_end'] = ts
                start_time = self.frame_timestamps[frame_id]['inference_start']
                if start_time:
                    self.stage_latencies['inference_proc'].append((ts - start_time) * 1000)

    def record_result_fetched(self, frame_id: int):
        """Record when result is fetched from output queue"""
        with self.lock:
            if frame_id in self.frame_timestamps:
                ts = time.time()
                self.frame_timestamps[frame_id]['fetched'] = ts
                infer_end_time = self.frame_timestamps[frame_id].get('inference_end')
                if infer_end_time:
                    self.stage_latencies['output_queue_wait'].append((ts - infer_end_time) * 1000)

    def record_display(self, frame_id: int):
        """Record when result is displayed"""
        with self.lock:
            if frame_id in self.frame_timestamps:
                timestamps = self.frame_timestamps[frame_id]
                ts = time.time()
                timestamps['display'] = ts

                fetched_time = timestamps.get('fetched')
                if fetched_time:
                    self.stage_latencies['real_post_proc'].append((ts - fetched_time) * 1000)

                total_latency = ts - timestamps['capture']
                self.latencies.append(total_latency)

                # Clean up old data
                if len(self.frame_timestamps) > 100:
                    oldest = min(self.frame_timestamps.keys())
                    if oldest < frame_id - 100:
                        del self.frame_timestamps[oldest]

    def get_latency_stats(self) -> Optional[Dict[str, float]]:
        """Get detailed latency statistics"""
        if not self.latencies:
            return None

        latencies = list(self.latencies)

        def safe_avg(d):
            return sum(d) / len(d) if d else 0

        return {
            'total_avg': safe_avg(latencies) * 1000,
            'total_min': min(latencies) * 1000,
            'total_max': max(latencies) * 1000,
            'frame_diff': self._calculate_frame_diff(),
            'stage_capture': safe_avg(self.stage_latencies['capture_to_queue']),
            'stage_queue_wait': safe_avg(self.stage_latencies['queue_to_infer']),
            'stage_inference': safe_avg(self.stage_latencies['inference_proc']),
            'stage_output_wait': safe_avg(self.stage_latencies['output_queue_wait']),
            'stage_real_postproc': safe_avg(self.stage_latencies['real_post_proc']),
        }

    def _calculate_frame_diff(self) -> int:
        """Calculate frame lag"""
        if not self.frame_timestamps:
            return 0
        latest_capture = 0
        latest_frame_id = 0
        for frame_id, timestamps in self.frame_timestamps.items():
            if timestamps.get('capture', 0) > latest_capture:
                latest_capture = timestamps['capture']
                latest_frame_id = frame_id
        last_display_frame = 0
        for frame_id, timestamps in self.frame_timestamps.items():
            if timestamps.get('display') and frame_id > last_display_frame:
                last_display_frame = frame_id
        return latest_frame_id - last_display_frame


class AsyncCamera:
    """
    Async camera capture class - uses queue to avoid frame drops, keeps only latest frame.
    Supports real USB cameras and video file sources.
    """
    def __init__(self, max_queue_size: int = 2, map_dir: str = "./data",
                 latency_monitor: Optional[LatencyMonitor] = None,
                 vid_pid: str = "0211:5838", camera=None,
                 camera_resolution: Optional[Tuple[int, int]] = None):
        # External camera source (video file, etc.)
        if camera is not None:
            self.cam = camera
            self.video_fps = getattr(camera, 'fps', 30)
        else:
            if vid_pid and vid_pid.lower().startswith("2b03:"):
                from wavestereo.visual.zed_uvc import ZEDUVC

                print(f"[INFO] Detected ZED camera (VID:PID={vid_pid}), using ZEDUVC class")
                resolution_map = {
                    (2208, 1242): 'HD2K',
                    (1920, 1080): 'HD1080',
                    (1280, 720): 'HD720',
                    (672, 376): 'VGA',
                }
                if camera_resolution is None:
                    raise ValueError(
                        "ZED camera requires camera resolution from resolution.txt, e.g. 1280,720"
                    )
                resolution = resolution_map.get(tuple(camera_resolution))
                if resolution is None:
                    raise ValueError(
                        f"Unsupported ZED per-eye resolution {camera_resolution}; "
                        f"supported: {sorted(resolution_map.keys())}"
                    )
                print(f"[INFO] ZED resolution from camera config: {resolution}")
                self.cam = ZEDUVC(vid_pid=vid_pid, map_dir=map_dir, resolution=resolution)
            else:
                from wavestereo.visual.usbbinocam import BinocularCam

                print(f"[INFO] Using generic BinocularCam class (VID:PID={vid_pid})")
                self.cam = BinocularCam(vid_pid=vid_pid, map_dir=map_dir,
                                        resolution=camera_resolution)
        self.frame_queue = queue.Queue(maxsize=max_queue_size)
        self.stopped = False
        self.frame_count = 0
        self.latency_monitor = latency_monitor
        self.thread = None
        self.lock = threading.Lock()

    def start(self):
        """Start camera thread"""
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
        return self

    def update(self):
        """Camera update loop"""
        last_capture_time = time.time()
        while not self.stopped:
            try:
                current_id = self.frame_count
                capture_ts = time.time()

                if self.latency_monitor:
                    self.latency_monitor.record_frame_capture_start(current_id, last_capture_time)

                frame = self.cam.get_rectifyframe()

                if frame is not None:
                    now = time.time()
                    if self.latency_monitor:
                        self.latency_monitor.record_capture_complete(current_id, now)

                    # Drop old frames if queue is full
                    try:
                        while self.frame_queue.full():
                            self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait((frame, current_id))
                        with self.lock:
                            self.frame_count += 1
                    except queue.Full:
                        pass

                    last_capture_time = now
                time.sleep(0.001)
            except Exception as e:
                print(f"[ERROR] Camera update error: {e}")
                time.sleep(0.1)

    def get_frame(self) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[int]]:
        """Get latest frame - drops all old frames in queue, keeps only newest"""
        latest_frame = None
        latest_id = None
        try:
            while True:
                frame, frame_id = self.frame_queue.get_nowait()
                latest_frame = frame
                latest_id = frame_id
        except queue.Empty:
            pass
        return latest_frame, latest_id

    def stop(self):
        """Stop camera thread"""
        self.stopped = True
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        # Stop the underlying camera
        if hasattr(self.cam, 'stop'):
            self.cam.stop()


class Preprocessor:
    """
    Preprocessing class - config-driven transform
    Supports both PyTorch transform pipeline and direct numpy preprocessing.
    Uses a fused fast path when the transform is standard (RightTopPad + Transpose + ToTensor + Normalize).
    """
    def __init__(self, device: torch.device, transform_config: Optional[list] = None,
                 target_size: Optional[Tuple[int, int]] = None,
                 pad_type: str = 'RightTopPad',
                 target_height: Optional[int] = None,
                 target_width: Optional[int] = None):
        self.device = device
        self.target_size = target_size
        self.pad_type = pad_type
        self.transform = None
        self.target_h = target_height
        self.target_w = target_width

        # Pre-allocated buffers for fast fused path
        self._fast_buf_l = None   # float32 numpy buffer
        self._fast_buf_r = None
        self._fast_pad_h = 0
        self._fast_pad_w = 0
        self._use_fast_path = False

        if transform_config is not None:
            # Keep only the public inference padding/normalization settings.
            # The full OpenStereo dataset transform pipeline is intentionally not imported.
            self._parse_pad_type(transform_config)

        # Fused normalization: (x/255 - mean) / std = x * scale + bias
        mean = [0.0, 0.0, 0.0]
        std = [1.0, 1.0, 1.0]
        if transform_config is not None:
            for t in transform_config:
                if isinstance(t, dict):
                    t_name = t.get('type', t.get('name', t.get('NAME', '')))
                    if 'NormalizeImage' in t_name:
                        mean = t['MEAN']
                        std = t['STD']
                        break
        scale = [1.0 / (255.0 * s) for s in std]
        bias = [-m / s for m, s in zip(mean, std)]
        self._norm_scale = torch.tensor(scale, device=device).view(1, 3, 1, 1)
        self._norm_bias = torch.tensor(bias, device=device).view(1, 3, 1, 1)

        # Detect if we can use the fast fused path
        if (self.pad_type == 'RightTopPad' and self.target_h and self.target_w
                and self.device.type == 'cuda'):
            # Pre-allocate GPU tensors to avoid repeated numpy→torch→GPU copies
            self._fast_buf_l_cpu = np.zeros((self.target_h, self.target_w, 3), dtype=np.float32)
            self._fast_buf_r_cpu = np.zeros((self.target_h, self.target_w, 3), dtype=np.float32)
            self._fast_gpu_l = torch.zeros(1, 3, self.target_h, self.target_w,
                                           device=device, dtype=torch.float32)
            self._fast_gpu_r = torch.zeros(1, 3, self.target_h, self.target_w,
                                           device=device, dtype=torch.float32)
            self._use_fast_path = True
            print(f"[INFO] Preprocessor: fused fast path enabled "
                  f"({self.target_w}×{self.target_h})")

    def _parse_pad_type(self, transform_config: list):
        """Parse padding type and target size from transform config"""
        for t in transform_config:
            if isinstance(t, dict):
                t_name = t.get('type', t.get('name', t.get('NAME', '')))
                if 'RightTopPad' in t_name:
                    self.pad_type = 'RightTopPad'
                    if 'SIZE' in t:
                        self.target_h, self.target_w = t['SIZE']
                elif 'RightBottomPad' in t_name:
                    self.pad_type = 'RightBottomPad'
                    if 'SIZE' in t:
                        self.target_h, self.target_w = t['SIZE']
                elif 'DivisiblePad' in t_name:
                    self.pad_type = 'DivisiblePad'
        print(f"[INFO] Preprocessor padding mode: {self.pad_type}")

    def prepare(self, left_img: np.ndarray, right_img: np.ndarray) -> Tuple[Dict[str, Any], Tuple[int, int]]:
        orig_h, orig_w = left_img.shape[:2]

        # ── Fast fused path ──
        if self._use_fast_path:
            pad_h = self.target_h - orig_h
            pad_w = self.target_w - orig_w

            if pad_h < 0 or pad_w < 0:
                raise ValueError(
                    f"RightTopPad target ({self.target_w}x{self.target_h}) is smaller than "
                    f"input frame ({orig_w}x{orig_h}); check camera resolution and inference SIZE"
                )

            # Upload uint8 HWC(BGR) → GPU and convert to RGB before ImageNet
            # normalization, matching the non-fast inference path.
            left_u8 = torch.from_numpy(left_img).unsqueeze(0)
            right_u8 = torch.from_numpy(right_img).unsqueeze(0)

            gpu_l = left_u8.to(self.device, non_blocking=True).permute(0, 3, 1, 2).float()
            gpu_r = right_u8.to(self.device, non_blocking=True).permute(0, 3, 1, 2).float()
            if gpu_l.shape[1] == 3:
                gpu_l = gpu_l[:, [2, 1, 0], :, :]
                gpu_r = gpu_r[:, [2, 1, 0], :, :]

            # Fused: (x/255 - mean)/std = x * scale + bias.
            gpu_l = gpu_l.mul_(self._norm_scale).add_(self._norm_bias)
            gpu_r = gpu_r.mul_(self._norm_scale).add_(self._norm_bias)

            # Match RightTopPad semantics: top/right padding uses edge values, not
            # zeros/stale buffer contents. This keeps live infercam preprocessing
            # consistent with the normal OpenCV/PIL inference pipeline.
            self._fast_gpu_l[:, :, pad_h:pad_h + orig_h, :orig_w] = gpu_l
            self._fast_gpu_r[:, :, pad_h:pad_h + orig_h, :orig_w] = gpu_r
            if pad_h > 0:
                self._fast_gpu_l[:, :, :pad_h, :orig_w] = gpu_l[:, :, :1, :].expand(-1, -1, pad_h, -1)
                self._fast_gpu_r[:, :, :pad_h, :orig_w] = gpu_r[:, :, :1, :].expand(-1, -1, pad_h, -1)
            if pad_w > 0:
                self._fast_gpu_l[:, :, :, orig_w:orig_w + pad_w] = \
                    self._fast_gpu_l[:, :, :, orig_w - 1:orig_w].expand(-1, -1, self.target_h, pad_w)
                self._fast_gpu_r[:, :, :, orig_w:orig_w + pad_w] = \
                    self._fast_gpu_r[:, :, :, orig_w - 1:orig_w].expand(-1, -1, self.target_h, pad_w)

            return {'left': self._fast_gpu_l, 'right': self._fast_gpu_r,
                    'pad': [pad_h, pad_w]}, (orig_h, orig_w)

        # ── Legacy path (config-driven transform or CPU device) ──
        if left_img.dtype != np.float32:
            left_img = left_img.astype(np.float32, copy=False)
        if right_img.dtype != np.float32:
            right_img = right_img.astype(np.float32, copy=False)

        if left_img.shape[-1] == 3:
            left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
            right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            sample = {'left': left_img, 'right': right_img}
            sample = self.transform(sample)
            pad = sample.get('pad', [0, 0])
            left_tensor = sample['left'].unsqueeze(0).to(self.device)
            right_tensor = sample['right'].unsqueeze(0).to(self.device)
            return {'left': left_tensor, 'right': right_tensor, 'pad': pad}, (orig_h, orig_w)
        else:
            pad_h = max(0, self.target_h - orig_h) if self.target_h else 0
            pad_w = max(0, self.target_w - orig_w) if self.target_w else 0

            if self.target_h and self.target_w:
                if self._fast_buf_l is None:
                    self._fast_buf_l = np.zeros((self.target_h, self.target_w, 3), dtype=np.float32)
                    self._fast_buf_r = np.zeros((self.target_h, self.target_w, 3), dtype=np.float32)
                else:
                    self._fast_buf_l.fill(0)
                    self._fast_buf_r.fill(0)
                self._fast_buf_l[pad_h:pad_h + orig_h, :orig_w, :] = left_img
                self._fast_buf_r[pad_h:pad_h + orig_h, :orig_w, :] = right_img
                left_tensor = torch.from_numpy(self._fast_buf_l).permute(2, 0, 1).unsqueeze(0).contiguous()
                right_tensor = torch.from_numpy(self._fast_buf_r).permute(2, 0, 1).unsqueeze(0).contiguous()
            else:
                left_tensor = torch.from_numpy(left_img).permute(2, 0, 1).unsqueeze(0).contiguous()
                right_tensor = torch.from_numpy(right_img).permute(2, 0, 1).unsqueeze(0).contiguous()

            left_tensor = left_tensor.float().div_(255.0).mul_(self._norm_scale.cpu()).add_(self._norm_bias.cpu())
            right_tensor = right_tensor.float().div_(255.0).mul_(self._norm_scale.cpu()).add_(self._norm_bias.cpu())

            if self.device.type == 'cuda':
                left_tensor = left_tensor.to(self.device, non_blocking=True)
                right_tensor = right_tensor.to(self.device, non_blocking=True)

            return {'left': left_tensor, 'right': right_tensor, 'pad': [pad_h, pad_w]}, (orig_h, orig_w)


def load_camera_resolution(cam_file: str) -> Tuple[int, int]:
    """Load per-eye camera resolution from cam_file/resolution.txt.

    Returns:
        (width, height) for one rectified camera image, not the combined
        binocular frame width.
    """
    if not cam_file or not os.path.exists(cam_file):
        raise FileNotFoundError(f"Camera config directory not found: {cam_file}")

    resolution_path = os.path.join(cam_file, "resolution.txt")
    if os.path.exists(resolution_path):
        with open(resolution_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.split("#", 1)[0].strip()
                if not line:
                    continue
                parts = line.replace(",", " ").split()
                if len(parts) != 2:
                    raise ValueError(
                        f"Invalid resolution.txt format in {resolution_path}: {raw_line.strip()!r}; "
                        "expected 'width,height', e.g. 1280,720"
                    )
                width, height = int(parts[0]), int(parts[1])
                print(f"[INFO] Loaded camera resolution: {width}x{height}")
                return width, height

    map1x_path = os.path.join(cam_file, "map1x.npy")
    if os.path.exists(map1x_path):
        map1x = np.load(map1x_path, mmap_mode="r")
        height, width = map1x.shape[:2]
        print(f"[WARNING] resolution.txt not found; inferred camera resolution from map1x.npy: {width}x{height}")
        return int(width), int(height)

    raise FileNotFoundError(
        f"Camera resolution not found in {cam_file}. Add resolution.txt with per-eye width,height, e.g. 1280,720"
    )


def load_camera_params(cam_file: str) -> Tuple[np.ndarray, float, str]:
    """
    Load camera intrinsic parameters, baseline, and vidpid

    Args:
        cam_file: Directory containing K.txt and vidpid.txt

    Returns:
        K: Intrinsic matrix (3x3)
        baseline: Baseline distance
        vidpid: USB vendor/product ID string
    """
    K = np.array([[700, 0, 640], [0, 700, 360], [0, 0, 1]], dtype=np.float32)
    baseline = 0.05
    vidpid_content = "0211:5838"

    if cam_file and os.path.exists(cam_file):
        vidpid_filepath = os.path.join(cam_file, "vidpid.txt")
        if os.path.exists(vidpid_filepath):
            with open(vidpid_filepath, 'r') as f:
                vidpid_content = f.read().strip()
                print(f"[INFO] Loaded vidpid.txt: {vidpid_content}")
        else:
            print(f"[WARNING] vidpid.txt not found: {vidpid_filepath}")

        intrinsic_filepath = os.path.join(cam_file, "K.txt")
        if os.path.exists(intrinsic_filepath):
            with open(intrinsic_filepath, 'r') as f:
                lines = f.readlines()
                K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3, 3)
                baseline = float(lines[1].strip())
            print(f"[INFO] Loaded camera intrinsics, baseline: {baseline}")
        else:
            print(f"[WARNING] K.txt not found: {intrinsic_filepath}, using defaults")
    else:
        print(f"[WARNING] Camera file not found: {cam_file}, using defaults")

    return K, baseline, vidpid_content


def display_latency_overlay(disp_color: np.ndarray, frame_id: int,
                            latency_stats: Optional[Dict[str, float]],
                            avg_fps: float, skip_frames: int,
                            mode_str: str = "2D Disparity",
                            prefix: str = "",
                            model_fps: float = 0.0) -> np.ndarray:
    """
    Display latency information overlay on disparity image

    Args:
        disp_color: Color disparity image
        frame_id: Current frame ID
        latency_stats: Latency statistics from LatencyMonitor
        avg_fps: Average FPS
        skip_frames: Number of frames to skip
        mode_str: Current mode string
        prefix: Prefix for title (e.g., "[TRT]")

    Returns:
        Image with overlay
    """
    # Semi-transparent panel behind the text: blend only the small ROI instead
    # of copying/blending the whole image (saves ~1ms of per-frame CPU/GIL work,
    # which matters once the display stops being the GPU bottleneck).
    _x1, _y1, _x2, _y2 = 10, 10, 380, 320
    _roi = disp_color[_y1:_y2, _x1:_x2]
    _panel = _roi.copy()
    cv2.rectangle(_panel, (0, 0), (_x2 - _x1, _y2 - _y1), (0, 0, 0), -1)
    cv2.addWeighted(_panel, 0.3, _roi, 0.7, 0, _roi)

    title = f"{prefix} FPS: {avg_fps:.1f}" if prefix else f"FPS: {avg_fps:.1f}"
    cv2.putText(disp_color, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    # Model GPU FPS — pure model ceiling, matching profile_runtime reciprocal
    if model_fps > 0:
        cv2.putText(disp_color, f"Model: {model_fps:.1f} FPS (GPU)",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_text = 95
    else:
        y_text = 70
    cv2.putText(disp_color, f"Frame: {frame_id}", (20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(disp_color, f"Skip: {skip_frames}", (20, y_text + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(disp_color, f"Mode (v): {mode_str}", (20, y_text + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if latency_stats:
        y_offset = y_text + 75
        step = 25
        color_info = (255, 255, 0)

        cv2.putText(disp_color, f"Total: {latency_stats['total_avg']:.1f}ms",
                    (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_info, 2)

        cv2.putText(disp_color, f"Capture: {latency_stats['stage_capture']:.1f}ms",
                    (20, y_offset + step), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(disp_color, f"QueueWait: {latency_stats['stage_queue_wait']:.1f}ms",
                    (20, y_offset + step * 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(disp_color, f"Inference: {latency_stats['stage_inference']:.1f}ms",
                    (20, y_offset + step * 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(disp_color, f"OutputWait: {latency_stats['stage_output_wait']:.1f}ms",
                    (20, y_offset + step * 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(disp_color, f"PostProc: {latency_stats['stage_real_postproc']:.1f}ms",
                    (20, y_offset + step * 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        stages = {
            'Capture': latency_stats['stage_capture'],
            'QueueWait': latency_stats['stage_queue_wait'],
            'Inference': latency_stats['stage_inference'],
            'OutputWait': latency_stats['stage_output_wait'],
            'PostProc': latency_stats['stage_real_postproc']
        }
        bottleneck = max(stages, key=stages.get)
        bottleneck_val = stages[bottleneck]

        cv2.putText(disp_color, f"Bottleneck: {bottleneck} ({bottleneck_val:.1f}ms)",
                    (20, y_offset + step * 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if latency_stats['total_avg'] < 50:
            status = "LOW"
            color = (0, 255, 0)
        elif latency_stats['total_avg'] < 100:
            status = "MED"
            color = (0, 255, 255)
        else:
            status = "HIGH"
            color = (0, 0, 255)

        cv2.putText(disp_color, f"Status: {status}",
                    (20, y_offset + step * 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return disp_color


def print_latency_stats(frame_id: int, latency_stats: Dict[str, float], avg_fps: float,
                        prefix: str = "", model_fps: float = 0.0):
    """Print latency statistics to console"""
    total = latency_stats['total_avg']
    print(f"\n[LATENCY] {prefix} Frame {frame_id} Performance Analysis:")
    print(f"  {'Stage':<18} | {'Time (ms)':<10} | {'Percent':<10}")
    print(f"  {'-'*18}-{'-'*10}-{'-'*10}")
    if total > 0:
        print(f"  {'1. Capture':<18} | {latency_stats['stage_capture']:<10.1f} | {latency_stats['stage_capture']/total*100:>5.1f}%")
        print(f"  {'2. Queue Wait':<18} | {latency_stats['stage_queue_wait']:<10.1f} | {latency_stats['stage_queue_wait']/total*100:>5.1f}%")
        print(f"  {'3. Inference':<18} | {latency_stats['stage_inference']:<10.1f} | {latency_stats['stage_inference']/total*100:>5.1f}%")
        print(f"  {'4. Output Wait':<18} | {latency_stats['stage_output_wait']:<10.1f} | {latency_stats['stage_output_wait']/total*100:>5.1f}%")
        print(f"  {'5. Post-process':<18} | {latency_stats['stage_real_postproc']:<10.1f} | {latency_stats['stage_real_postproc']/total*100:>5.1f}%")
        print(f"  {'-'*18}-{'-'*10}-{'-'*10}")
        print(f"  {'Total (avg)':<18} | {total:<10.1f} | 100.0%")
        print(f"  {'Total (min/max)':<18} | {latency_stats['total_min']:.1f}/{latency_stats['total_max']:.1f} | ")
        print(f"  {'Frame Lag':<18} | {latency_stats['frame_diff']} frames | ")
        if model_fps > 0:
            print(f"  {'Model FPS (GPU)':<18} | {model_fps:.1f} | ")
        print(f"  {'Current FPS':<18} | {avg_fps:.1f} | ")
