"""Asynchronous PyTorch, TensorRT, and OpenVINO inference backends."""
import time
import threading
import queue
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Any

import torch
import numpy as np

from wavestereo.artifacts import (
    bundle_input_hw,
    bundle_input_shape,
    resolve_openvino_model,
    validate_profile_device,
)
from wavestereo.inference.postprocessing import SpatialTransform


@dataclass(frozen=True)
class BackendResult:
    """Native, un-restored model output from one backend invocation."""

    native_disparity: np.ndarray
    model_ms: float


@dataclass(frozen=True)
class AsyncInferenceResult:
    """Named asynchronous result shared by camera and video inference."""

    frame_id: int
    native_disparity: np.ndarray
    source_size: tuple[int, int]
    model_ms: float
    spatial_transform: SpatialTransform


class AsyncInference:
    """
    Async inference wrapper - works with any inference backend
    Keeps only latest frame in queues for low latency
    """
    def __init__(self, inference_backend: Any, preprocessor: Any,
                 latency_monitor: Any, input_queue_size: int = 1,
                 output_queue_size: int = 1,
                 spatial_transform: SpatialTransform | None = None):
        """
        Initialize async inference

        Args:
            inference_backend: Backend with forward() method
            preprocessor: Preprocessor instance
            latency_monitor: LatencyMonitor instance
            input_queue_size: Size of input queue (default 1 for latest only)
            output_queue_size: Size of output queue (default 1 for latest only)
            spatial_transform: Declared source/model geometry. Model-size
                camera frames are accepted without another resize.
        """
        self.backend = inference_backend
        self.preprocessor = preprocessor
        self.latency_monitor = latency_monitor
        self.spatial_transform = spatial_transform
        self.input_queue = queue.Queue(maxsize=input_queue_size)
        self.output_queue = queue.Queue(maxsize=output_queue_size)
        self.stopped = False
        self.inference_thread = threading.Thread(target=self.inference_loop, daemon=True)
        self.inference_thread.start()

    def inference_loop(self) -> None:
        """Continuous inference thread — non-blocking GPU→CPU transfer"""
        with torch.no_grad():
            while not self.stopped:
                try:
                    frame, frame_id, source_size = self.input_queue.get(timeout=0.01)
                    if frame is None:
                        continue

                    self.latency_monitor.record_inference_start(frame_id)

                    transform = self.spatial_transform
                    if transform is None:
                        model_size = frame["left"].shape[:2]
                        transform = SpatialTransform.from_sizes(
                            source_size, model_size
                        )
                    elif transform.source_size != source_size:
                        raise ValueError(
                            f"Submitted source size {source_size} does not match "
                            f"runtime source size {transform.source_size}"
                        )
                    model_frame = transform.prepare_stereo(frame)
                    backend_result = self.backend.forward(
                        model_frame, self.preprocessor, frame_id
                    )

                    self.latency_monitor.record_inference_end(frame_id)
                    result = AsyncInferenceResult(
                        frame_id=frame_id,
                        native_disparity=backend_result.native_disparity,
                        source_size=source_size,
                        model_ms=backend_result.model_ms,
                        spatial_transform=transform,
                    )
                    while self.output_queue.full():
                        try:
                            self.output_queue.get_nowait()
                        except queue.Empty:
                            break
                    try:
                        self.output_queue.put_nowait(result)
                    except queue.Full:
                        pass

                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Inference error: {e}")
                    import traceback
                    traceback.print_exc()

    def submit(self, frame: Dict[str, np.ndarray], frame_id: int,
               source_size: Tuple[int, int]) -> None:
        """Submit inference task - drops old frames if queue is full"""
        try:
            while self.input_queue.full():
                self.input_queue.get_nowait()
            self.input_queue.put_nowait((frame, frame_id, tuple(source_size)))
        except queue.Full:
            pass

    def get_result(
        self, block_timeout: float = 0.001
    ) -> AsyncInferenceResult | None:
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

        return latest_item

    def stop(self) -> None:
        """Stop inference thread"""
        self.stopped = True
        if self.inference_thread.is_alive():
            self.inference_thread.join(timeout=5.0)


class PyTorchBackend:
    """
    PyTorch inference backend
    """
    def __init__(self, model: torch.nn.Module, device: torch.device,
                 use_amp: bool = True, amp_dtype: torch.dtype = torch.bfloat16,
                 max_disp: float = 192.0,
                 use_debug_forward: bool = False):
        self.model = model
        self.device = device
        self.use_amp = bool(
            use_amp and device.type == "cuda" and amp_dtype is not torch.float32
        )
        self.amp_dtype = amp_dtype
        self.max_disp = max_disp
        self.use_debug_forward = use_debug_forward
        self.latest_cost_prob = None  # [48, H/4, W/4] numpy, written by forward(), read by main thread
        self.latest_disp_init = None  # [H, W] float32, pre-GRU init disparity

    def forward(self, frame: Dict[str, np.ndarray], preprocessor: Any,
                frame_id: int) -> BackendResult:
        """Return native disparity in the submitted frame geometry."""
        left_img, right_img = frame['left'], frame['right']
        sample_dict, (orig_h, orig_w) = preprocessor.prepare(left_img, right_img)
        pad_h, _ = sample_dict['pad']

        # Model-only timing. CUDA events avoid a full-device synchronization;
        # CPU inference uses a regular monotonic wall-clock timer.
        if self.device.type == "cuda":
            event_start = torch.cuda.Event(enable_timing=True)
            event_end = torch.cuda.Event(enable_timing=True)
            event_start.record()
        else:
            model_start = time.perf_counter()
        with torch.amp.autocast(
            self.device.type,
            dtype=self.amp_dtype,
            enabled=self.use_amp,
        ):
            # Use debug forward only if enabled AND available
            if self.use_debug_forward and hasattr(self.model, 'forward_debug'):
                model_pred = self.model.forward_debug(sample_dict)
            else:
                model_pred = self.model(sample_dict)
        if self.device.type == "cuda":
            event_end.record()
        else:
            model_gpu_ms = (time.perf_counter() - model_start) * 1000.0

        disp_pred = model_pred['disp_pred'].squeeze()
        disp_init = model_pred.get('init_disp_up', None)
        if disp_init is not None:
            disp_init = disp_init.squeeze()  # [H, W] pre-GRU init

        # Extract cost volume probability (for mouse-over prob curve)
        prob = model_pred.get('prob', None)  # [B, 48, H/4, W/4]
        if prob is not None:
            prob = prob.squeeze(0)  # [48, Hpad/4, Wpad/4]
            crop_h4 = pad_h // 4
            orig_h4 = (orig_h + 3) // 4
            orig_w4 = (orig_w + 3) // 4
            prob = prob[
                :,
                crop_h4:crop_h4 + orig_h4,
                :orig_w4,
            ]
            self.latest_cost_prob = prob.detach().float().cpu().numpy()

        # Both supported padding modes keep the left edge fixed. Crop the
        # exact original height and width so right-side padding never leaks
        # into static-image or camera results.
        disp_pred = disp_pred[pad_h:pad_h + orig_h, :orig_w]
        if disp_init is not None:
            disp_init = disp_init[pad_h:pad_h + orig_h, :orig_w]
            self.latest_disp_init = disp_init.detach().float().cpu().numpy()

        if self.device.type == "cuda":
            # Synchronize the end event before reading it. The output D2H copy
            # above normally flushes the default stream, but models may use
            # internal streams.
            event_end.synchronize()
            model_gpu_ms = event_start.elapsed_time(event_end)

        return BackendResult(
            native_disparity=(
                disp_pred.detach().float().cpu().numpy().astype(
                    np.float32, copy=False
                )
            ),
            model_ms=float(model_gpu_ms),
        )

    @staticmethod
    def build_from_config(args: Any, cfgs: Any, device: torch.device,
                          compile_model: bool = False):
        """
        Build PyTorch backend from config

        Args:
            args: Command line args
            cfgs: Config dict
            device: Torch device
            compile_model: Whether to use torch.compile

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
        infer_cfg = cfgs.get('INFERENCE', {})
        amp_dtype_value = infer_cfg.get('AMP_DTYPE', 'bfloat16')
        if isinstance(amp_dtype_value, torch.dtype):
            amp_dtype = amp_dtype_value
        else:
            amp_dtypes = {
                'fp16': torch.float16,
                'float16': torch.float16,
                'bf16': torch.bfloat16,
                'bfloat16': torch.bfloat16,
                'fp32': torch.float32,
                'float32': torch.float32,
            }
            try:
                amp_dtype = amp_dtypes[str(amp_dtype_value).lower()]
            except KeyError as exc:
                raise ValueError(
                    f"Unsupported INFERENCE.AMP_DTYPE: {amp_dtype_value!r}"
                ) from exc
        use_amp = bool(infer_cfg.get('AMP', True))
        use_debug = bool(getattr(args, 'debug', False))
        return PyTorchBackend(
            model,
            device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            max_disp=max_disp,
            use_debug_forward=use_debug,
        )


class TensorRTBackend:
    """
    TensorRT inference backend (WAVEStereo 单引擎)
    """
    def __init__(self, trt_runner: Any, max_disp: float):
        self.trt_runner = trt_runner
        self.max_disp = max_disp
        self.model = None  # TRT has no PyTorch model

    def forward(self, frame: Dict[str, np.ndarray], preprocessor: Any,
                frame_id: int) -> BackendResult:
        """Run TensorRT inference and return native disparity only."""
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

        # Remove the exact RightTopPad region recorded by the preprocessor.
        pad_h, pad_w = sample_dict['pad']
        if pad_h > 0 or pad_w > 0:
            disp_pred = disp_pred[pad_h:pad_h + orig_h, :orig_w]

        _ev_e.synchronize()
        model_gpu_ms = _ev_s.elapsed_time(_ev_e)
        return BackendResult(
            native_disparity=(
                disp_pred.detach().float().cpu().numpy().astype(
                    np.float32, copy=False
                )
            ),
            model_ms=float(model_gpu_ms),
        )

    @staticmethod
    def load_from_model_dir(model_dir: str | Path):
        """Load a TensorRT backend from one canonical model bundle."""
        from wavestereo.deployment.tensorrt import load_trt_runner
        trt_runner = load_trt_runner(model_dir)
        print("[INFO] Loaded WAVEStereo TensorRT engine")
        return TensorRTBackend(
            trt_runner,
            max_disp=trt_runner.max_disp,
        )


class OpenVINOBackend:
    """OpenVINO backend backed by one canonical WAVEStereo model bundle."""

    def __init__(self, model_dir: str | Path, device: str,
                 cache_dir: str = "outputs/openvino_cache"):
        import openvino as ov

        model_dir = Path(model_dir)
        model_path, self.bundle_config = resolve_openvino_model(model_dir)
        expected_shape = bundle_input_shape(self.bundle_config)
        self.target_h, self.target_w = bundle_input_hw(self.bundle_config)
        validate_profile_device(self.bundle_config, device)
        self.core = ov.Core()
        available_devices = [value.upper() for value in self.core.available_devices]
        normalized_device = str(device).upper()
        virtual_device = normalized_device.startswith(("AUTO", "MULTI", "HETERO"))
        physical_device_available = normalized_device in available_devices or any(
            value.startswith(f"{normalized_device}.") for value in available_devices
        )
        if not virtual_device and not physical_device_available:
            raise RuntimeError(
                f"OpenVINO device {device!r} is unavailable; found: {self.core.available_devices}"
            )
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            self.core.set_property({"CACHE_DIR": str(Path(cache_dir).resolve())})

        model = self.core.read_model(model_path)
        if len(model.inputs) != 2:
            raise ValueError(
                f"WAVEStereo OpenVINO model must have two inputs, got "
                f"{len(model.inputs)}"
            )
        input_names = [port.get_any_name() for port in model.inputs]
        if input_names != self.bundle_config["input"]["names"]:
            raise ValueError(
                f"OpenVINO model inputs {input_names} do not match bundle metadata "
                f"{self.bundle_config['input']['names']}"
            )
        for port in model.inputs:
            shape = tuple(int(value) for value in port.shape)
            if shape != expected_shape:
                raise ValueError(
                    f"OpenVINO model input {port.get_any_name()!r} shape {shape} "
                    f"does not match bundle metadata {expected_shape}"
                )
            if port.get_element_type() != ov.Type.u8:
                raise ValueError(
                    f"OpenVINO model input {port.get_any_name()!r} must be uint8"
                )
        if len(model.outputs) != 1 or (
            model.output(0).get_any_name()
            != self.bundle_config["output"]["name"]
        ):
            raise ValueError("OpenVINO model output does not match bundle metadata")
        expected_output_shape = tuple(self.bundle_config["output"]["shape"])
        output_shape = tuple(int(value) for value in model.output(0).shape)
        if output_shape != expected_output_shape:
            raise ValueError(
                f"OpenVINO model output shape {output_shape} does not match "
                f"bundle metadata {expected_output_shape}"
            )
        if model.output(0).get_element_type() != ov.Type.f32:
            raise ValueError("OpenVINO model output must be float32")

        compile_start = time.perf_counter()
        self.compiled_model = self.core.compile_model(
            model,
            device,
            {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1"},
        )
        self.compile_time = time.perf_counter() - compile_start
        self.request = self.compiled_model.create_infer_request()
        self.input_left = self.compiled_model.input(0)
        self.input_right = self.compiled_model.input(1)
        self.output = self.compiled_model.output(0)
        self.max_disp = float(self.bundle_config["model"]["max_disp"])
        self.device = device
        self.model = None

        try:
            device_name = self.core.get_property(device, "FULL_DEVICE_NAME")
        except RuntimeError:
            device_name = str(device)
        print(f"[INFO] OpenVINO device: {device_name}")
        print(
            f"[INFO] OpenVINO model input: "
            f"1x{self.target_h}x{self.target_w}x3 uint8/NHWC/BGR"
        )
        print(
            f"[INFO] OpenVINO export profile: "
            f"{self.bundle_config['profile']}"
        )
        print(f"[INFO] OpenVINO compile time: {self.compile_time:.3f} s")

    def _prepare(self, image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            raise TypeError("OpenVINO input must be a numpy array")
        if image.dtype != np.uint8:
            raise ValueError(
                f"OpenVINO input must be uint8 BGR, got {image.dtype}"
            )
        expected = (self.target_h, self.target_w, 3)
        if image.shape != expected:
            raise ValueError(
                f"OpenVINO input must have exact HWC shape {expected}, "
                f"got {image.shape}"
            )
        return np.ascontiguousarray(image[None])

    def forward(self, frame: Dict[str, np.ndarray], preprocessor: Any,
                frame_id: int) -> BackendResult:
        left_img, right_img = frame['left'], frame['right']
        left = self._prepare(left_img)
        right = self._prepare(right_img)

        start = time.perf_counter()
        result = self.request.infer({self.input_left: left, self.input_right: right})
        model_ms = (time.perf_counter() - start) * 1000.0
        output = np.asarray(result[self.output])
        expected = (1, 1, self.target_h, self.target_w)
        if output.shape != expected:
            raise ValueError(
                f"OpenVINO output shape must be {expected}, got {output.shape}"
            )
        return BackendResult(
            native_disparity=output[0, 0].astype(np.float32, copy=True),
            model_ms=float(model_ms),
        )


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
