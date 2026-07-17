from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml


def normalize_image(img: torch.Tensor, mean=None, std=None) -> torch.Tensor:
    """Normalize BCHW image tensors in [0, 1] with ImageNet defaults."""
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    mean_t = img.new_tensor(mean).view(1, 3, 1, 1)
    std_t = img.new_tensor(std).view(1, 3, 1, 1)
    return (img - mean_t) / std_t


class TrtRunner:
    """Single-engine WAVEStereo TensorRT runner."""

    def __init__(self, cfg: dict[str, Any], engine_path: str | Path):
        import tensorrt as trt

        self.cfg = cfg
        self.normalize_mean = cfg.get("normalize_mean", [0.485, 0.456, 0.406])
        self.normalize_std = cfg.get("normalize_std", [0.229, 0.224, 0.225])
        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)

        engine_path = Path(engine_path)
        with engine_path.open("rb") as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(
                f"Failed to deserialize TensorRT engine: {engine_path}. "
                "TensorRT versions may differ; rebuild the engine with trtexec."
            )
        self.context = self.engine.create_execution_context()

    def _trt_to_torch_dtype(self, dt):
        trt = self.trt
        mapping = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
            trt.DataType.BF16: torch.bfloat16,
            trt.DataType.INT32: torch.int32,
            trt.DataType.INT8: torch.int8,
            trt.DataType.BOOL: torch.bool,
        }
        if dt not in mapping:
            raise RuntimeError(f"Unsupported TensorRT dtype: {dt}")
        return mapping[dt]

    def _get_io_names(self):
        trt = self.trt
        input_names, output_names = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                input_names.append(name)
            else:
                output_names.append(name)
        return input_names, output_names

    def forward_normalized(self, left_norm: torch.Tensor, right_norm: torch.Tensor) -> torch.Tensor:
        """Run inference on already-normalized BCHW tensors."""
        inputs = {"left_image": left_norm, "right_image": right_norm}
        for name, tensor in list(inputs.items()):
            expected = self._trt_to_torch_dtype(self.engine.get_tensor_dtype(name))
            if tensor.dtype != expected:
                tensor = tensor.to(expected)
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            inputs[name] = tensor
            self.context.set_input_shape(name, tuple(tensor.shape))

        _, output_names = self._get_io_names()
        outputs = {}
        for name in output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = self._trt_to_torch_dtype(self.engine.get_tensor_dtype(name))
            outputs[name] = torch.empty(shape, device=left_norm.device, dtype=dtype)

        for name, tensor in inputs.items():
            self.context.set_tensor_address(name, int(tensor.data_ptr()))
        for name, tensor in outputs.items():
            self.context.set_tensor_address(name, int(tensor.data_ptr()))

        ok = self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT execution failed")
        return outputs.get("disparity", next(iter(outputs.values())))

    def forward_image(self, left_img: torch.Tensor, right_img: torch.Tensor) -> torch.Tensor:
        """Run inference on BCHW image tensors in [0, 1]."""
        left_norm = normalize_image(left_img, self.normalize_mean, self.normalize_std)
        right_norm = normalize_image(right_img, self.normalize_mean, self.normalize_std)
        return self.forward_normalized(left_norm, right_norm)

    def forward(self, left_img: torch.Tensor, right_img: torch.Tensor) -> torch.Tensor:
        return self.forward_image(left_img, right_img)

    def __call__(self, left_img: torch.Tensor, right_img: torch.Tensor) -> torch.Tensor:
        return self.forward(left_img, right_img)


def resolve_onnx_cfg_path(onnx_dir: str | Path) -> Path:
    onnx_dir = Path(onnx_dir)
    candidates = [onnx_dir / "onnx.yaml", onnx_dir.parent / "onnx.yaml"]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"onnx.yaml not found. Searched: {candidates}")


def make_res_tag_from_cfg(cfg: dict[str, Any]) -> str:
    h, w = cfg.get("image_size", [736, 1280])
    iters = cfg.get("valid_iters", 8)
    return f"{h}x{w}_iter{iters}"


def find_engine(onnx_dir: str | Path, res_tag: str | None = None, cfg: dict[str, Any] | None = None) -> Path:
    onnx_dir = Path(onnx_dir)
    candidates = []
    if cfg and cfg.get("engine_filename"):
        candidates.append(onnx_dir / cfg["engine_filename"])
    if res_tag is None and cfg:
        res_tag = make_res_tag_from_cfg(cfg)
    if res_tag is not None:
        candidates.append(onnx_dir / f"wavestereo_{res_tag}.engine")
    if cfg:
        h, w = cfg.get("image_size", [736, 1280])
        iters = cfg.get("valid_iters", 8)
        candidates.append(onnx_dir / f"wavestereo_{h}x{w}_iter{iters}.engine")
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"WAVEStereo engine not found in {onnx_dir}. Searched: {', '.join(map(str, candidates))}"
    )


def load_trt_runner(onnx_dir: str | Path) -> TrtRunner:
    cfg_path = resolve_onnx_cfg_path(onnx_dir)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    engine_path = find_engine(onnx_dir, make_res_tag_from_cfg(cfg), cfg)
    print(f"[INFO] Loaded TensorRT engine: {engine_path.name}")
    return TrtRunner(cfg, engine_path)
