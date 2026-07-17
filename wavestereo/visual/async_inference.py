"""
Inference Module - PyTorch and TensorRT backends with async inference
"""
import os
import time
import threading
import queue
from typing import Optional, Tuple, Dict, Any, Callable

import torch
import torch.nn.functional as F
import numpy as np


class AsyncInference:
    """
    Async inference wrapper - works with any inference backend
    Keeps only latest frame in queues for low latency
    """
    def __init__(self, inference_backend: Any, preprocessor: Any,
                 latency_monitor: Any, input_queue_size: int = 1,
                 output_queue_size: int = 1):
        """
        Initialize async inference

        Args:
            inference_backend: Backend with forward() method
            preprocessor: Preprocessor instance
            latency_monitor: LatencyMonitor instance
            input_queue_size: Size of input queue (default 1 for latest only)
            output_queue_size: Size of output queue (default 1 for latest only)
        """
        self.backend = inference_backend
        self.preprocessor = preprocessor
        self.latency_monitor = latency_monitor
        self.input_queue = queue.Queue(maxsize=input_queue_size)
        self.output_queue = queue.Queue(maxsize=output_queue_size)
        self.stopped = False
        self.inference_thread = threading.Thread(target=self.inference_loop, daemon=True)
        self.inference_thread.start()

    def inference_loop(self):
        """Continuous inference thread — non-blocking GPU→CPU transfer"""
        with torch.no_grad():
            while not self.stopped:
                try:
                    frame, frame_id, orig_size = self.input_queue.get(timeout=0.01)
                    if frame is None:
                        continue

                    self.latency_monitor.record_inference_start(frame_id)

                    # Run inference — returns CPU tensors (non_blocking, data
                    # may still be transferring while thread moves to next frame)
                    disp_vis, disp_raw, disp_low, disp_init, model_gpu_ms = \
                        self.backend.forward(frame, self.preprocessor, frame_id)

                    self.latency_monitor.record_inference_end(frame_id)
                    self.output_queue.put((disp_vis, disp_raw, disp_low, disp_init,
                                           frame_id, orig_size, model_gpu_ms))

                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Inference error: {e}")
                    import traceback
                    traceback.print_exc()

    def submit(self, frame: Dict[str, np.ndarray], frame_id: int, orig_size: Tuple[int, int]):
        """Submit inference task - drops old frames if queue is full"""
        try:
            while self.input_queue.full():
                self.input_queue.get_nowait()
            self.input_queue.put_nowait((frame, frame_id, orig_size))
        except queue.Full:
            pass

    def get_result(self, block_timeout: float = 0.001):
        """Get latest inference result — drains all old frames, then optionally
        blocks briefly on the output queue for zero-latency wakeup.

        Args:
            block_timeout: Max seconds to wait for a result (0 = non-blocking).
        """
        # 1. Drain all available results (non-blocking)
        latest_item = None
        try:
            while True:
                latest_item = self.output_queue.get_nowait()
        except queue.Empty:
            pass
        if latest_item is not None:
            return latest_item

        # 2. No result yet — block briefly (Queue Condition wakes us immediately
        #    when inference puts a result, no polling waste)
        if block_timeout > 0:
            try:
                latest_item = self.output_queue.get(timeout=block_timeout)
                # Drain any extras that accumulated
                while True:
                    try:
                        latest_item = self.output_queue.get_nowait()
                    except queue.Empty:
                        break
            except queue.Empty:
                pass

        return latest_item if latest_item else (None, None, None, None, None, None, None)

    def stop(self):
        """Stop inference thread"""
        self.stopped = True


class PyTorchBackend:
    """
    PyTorch inference backend
    """
    def __init__(self, model: torch.nn.Module, device: torch.device,
                 use_amp: bool = True, amp_dtype: torch.dtype = torch.bfloat16,
                 crop_pad: int = 16, max_disp: float = 192.0,
                 use_debug_forward: bool = False):
        self.model = model
        self.device = device
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.crop_pad = crop_pad
        self.max_disp = max_disp
        self.use_debug_forward = use_debug_forward
        self.latest_cost_prob = None  # [48, H/4, W/4] numpy, written by forward(), read by main thread
        self.latest_disp_init = None  # [H, W] float32, pre-GRU init disparity

    def forward(self, frame: Dict[str, np.ndarray], preprocessor: Any,
                frame_id: int):
        # Returns (disp_vis, disp_pred, disp_low, disp_init, model_gpu_ms)
        left_img, right_img = frame['left'], frame['right']
        sample_dict, (orig_h, orig_w) = preprocessor.prepare(left_img, right_img)

        # ---- model-only GPU time (matches profile_runtime) ----
        _ev_s = torch.cuda.Event(enable_timing=True)
        _ev_e = torch.cuda.Event(enable_timing=True)
        _ev_s.record()
        with torch.amp.autocast('cuda', dtype=self.amp_dtype, enabled=self.use_amp):
            # Use debug forward only if enabled AND available
            if self.use_debug_forward and hasattr(self.model, 'forward_debug'):
                model_pred = self.model.forward_debug(sample_dict)
            else:
                model_pred = self.model(sample_dict)
        _ev_e.record()

        disp_pred = model_pred['disp_pred'].squeeze()
        disp_low = model_pred.get('disp_low_res', None)
        if disp_low is not None:
            disp_low = disp_low.squeeze()  # [H/4, W/4] at native resolution
        disp_init = model_pred.get('init_disp_up', None)
        if disp_init is not None:
            disp_init = disp_init.squeeze()  # [H, W] pre-GRU init

        # Extract cost volume probability (for mouse-over prob curve)
        prob = model_pred.get('prob', None)  # [B, 48, H/4, W/4]
        if prob is not None:
            prob = prob.squeeze(0)  # [48, Hpad/4, Wpad/4]
            if self.crop_pad > 0:
                crop_h4 = self.crop_pad // 4
                prob = prob[:, crop_h4:crop_h4 + orig_h // 4, :orig_w // 4]
            self.latest_cost_prob = prob.detach().float().cpu().numpy()

        if self.crop_pad > 0:
            disp_pred = disp_pred[self.crop_pad:, :]
            if disp_low is not None:
                # crop at H/4 spatial scale
                disp_low = disp_low[self.crop_pad // 4:, :]
            if disp_init is not None:
                disp_init = disp_init[self.crop_pad:, :]

        disp_pred = disp_pred.clamp(min=0, max=self.max_disp)
        if disp_low is not None:
            disp_low = disp_low.clamp(min=0, max=self.max_disp)
            # Nearest-neighbor upsample to full resolution (zero interpolation —
            # each H/4 pixel becomes a 4×4 block of identical values)
            disp_low = F.interpolate(
                disp_low[None, None].float(), scale_factor=4,
                mode='nearest').squeeze()
        if disp_init is not None:
            disp_init = disp_init.clamp(min=0, max=self.max_disp)
            self.latest_disp_init = disp_init.detach().float().cpu().numpy()

        disp_vis = disp_pred.mul(255.0 / self.max_disp).to(torch.uint8)

        def _to_np(t):
            return t.detach().cpu().numpy() if t is not None else None

        # model GPU time — synchronize the event before reading (default-stream
        # sync via .cpu() above is usually enough, but the model may use internal
        # streams that aren't flushed by a D2H copy alone).
        _ev_e.synchronize()
        model_gpu_ms = _ev_s.elapsed_time(_ev_e)

        return (_to_np(disp_vis),
                _to_np(disp_pred),
                _to_np(disp_low),
                _to_np(disp_init),
                model_gpu_ms)

    @staticmethod
    def build_from_config(args: Any, cfgs: Any, device: torch.device,
                          logger: Any, compile_model: bool = False,
                          crop_pad: int = 16):
        """
        Build PyTorch backend from config

        Args:
            args: Command line args
            cfgs: Config dict
            device: Torch device
            logger: Logger instance
            compile_model: Whether to use torch.compile
            crop_pad: Pixels to crop from top (for RightTopPad)

        Returns:
            PyTorchBackend instance
        """
        from wavestereo.inference.checkpoint import load_checkpoint
        from wavestereo.model import WAVEStereo

        model = WAVEStereo(cfgs.MODEL).to(device).eval()
        weights = getattr(args, 'weights', None) or cfgs.MODEL.get('PRETRAINED_MODEL', '')
        if weights:
            load_checkpoint(model, weights, strict=False, map_location=device)
        else:
            print("[WARN] No weights provided; model will use random initialization")

        if compile_model:
            print("[INFO] Using torch.compile to optimize model...")
            try:
                model = torch.compile(
                    model,
                    dynamic=True,
                    mode="reduce-overhead",
                    fullgraph=False
                )
                print("[INFO] torch.compile optimization complete")
            except Exception as e:
                print(f"[WARNING] torch.compile failed, using original model: {e}")

        max_disp = float(cfgs.get('EVALUATOR', {}).get('MAX_DISP', cfgs.MODEL.get('MAX_DISP', 192)))
        use_debug = bool(getattr(args, 'debug', False))
        return PyTorchBackend(model, device, crop_pad=crop_pad, max_disp=max_disp,
                              use_debug_forward=use_debug)


class TensorRTBackend:
    """
    TensorRT inference backend (WAVEStereo 单引擎)
    """
    def __init__(self, trt_runner: Any, crop_pad: int = 0, max_disp: float = 192.0):
        self.trt_runner = trt_runner
        self.crop_pad = crop_pad
        self.max_disp = max_disp
        self.model = None  # TRT has no PyTorch model

    def forward(self, frame: Dict[str, np.ndarray], preprocessor: Any,
                frame_id: int):
        """
        Run TensorRT inference

        Returns:
            (disp_vis_cpu, disp_raw_cpu, disp_low_cpu, disp_init_cpu, model_gpu_ms)
        """
        left_img, right_img = frame['left'], frame['right']
        sample_dict, (orig_h, orig_w) = preprocessor.prepare(left_img, right_img)

        # ---- model-only GPU time (matches profile_runtime) ----
        _ev_s = torch.cuda.Event(enable_timing=True)
        _ev_e = torch.cuda.Event(enable_timing=True)
        _ev_s.record()
        disp_pred = self.trt_runner.forward_normalized(
            sample_dict['left'],
            sample_dict['right'],
        )
        _ev_e.record()

        disp_pred = disp_pred.squeeze()

        # Crop padding
        pad_h, pad_w = sample_dict['pad']
        crop_amount = self.crop_pad if self.crop_pad > 0 else pad_h
        if crop_amount > 0:
            disp_pred = disp_pred[crop_amount:, :]

        # Clamp to valid range (handles NaN/Inf implicitly)
        disp_pred = disp_pred.clamp(min=0, max=self.max_disp)

        # GPU visualization + single transfer (avoids CPU min/max scan)
        disp_vis = disp_pred.mul(255.0 / self.max_disp).to(torch.uint8)
        _ev_e.synchronize()
        model_gpu_ms = _ev_s.elapsed_time(_ev_e)
        return disp_vis.cpu().numpy(), disp_pred.cpu().numpy(), None, None, model_gpu_ms

    @staticmethod
    def load_from_onnx_dir(onnx_dir: str, crop_pad: int = 0):
        """
        Load TensorRT backend from ONNX directory

        Args:
            onnx_dir: Directory containing onnx.yaml and TensorRT engine
            crop_pad: Pixels to crop from top of output disparity

        Returns:
            TensorRTBackend instance
        """
        import yaml, os
        from wavestereo.trt.trt_runner import load_trt_runner
        trt_runner = load_trt_runner(onnx_dir)
        # Read max_disp from onnx config
        onnx_cfg_path = os.path.join(onnx_dir, 'onnx.yaml')
        max_disp = 192.0
        if os.path.exists(onnx_cfg_path):
            with open(onnx_cfg_path) as f:
                onnx_cfg = yaml.safe_load(f)
                max_disp = float(onnx_cfg.get('max_disp', 192))
        print("[INFO] Loaded WAVEStereo TensorRT engine")
        return TensorRTBackend(trt_runner, crop_pad=crop_pad, max_disp=max_disp)


def create_visualization_from_raw(disp_raw_cpu: np.ndarray) -> np.ndarray:
    """
    Create visualization disparity image from raw disparity

    Args:
        disp_raw_cpu: Raw disparity array

    Returns:
        Visualization disparity (uint8, 0-255)
    """
    disp_min = np.min(disp_raw_cpu)
    disp_max = np.max(disp_raw_cpu)
    if disp_max - disp_min > 1e-6:
        disp_vis_cpu = ((disp_raw_cpu - disp_min) / (disp_max - disp_min + 1e-6) * 255).astype(np.uint8)
    else:
        disp_vis_cpu = np.zeros_like(disp_raw_cpu, dtype=np.uint8)
    return disp_vis_cpu
