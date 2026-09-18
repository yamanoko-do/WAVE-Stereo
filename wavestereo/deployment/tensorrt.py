from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from wavestereo.artifacts import (
    bundle_input_shape,
    bundle_normalization,
    resolve_tensorrt_engine,
)


def normalize_image(img: torch.Tensor, mean, std) -> torch.Tensor:
    """Normalize BCHW image tensors in [0, 1] from bundle metadata."""
    mean_t = img.new_tensor(mean).view(1, 3, 1, 1)
    std_t = img.new_tensor(std).view(1, 3, 1, 1)
    return (img - mean_t) / std_t


class TrtRunner:
    """Single-engine WAVEStereo TensorRT runner."""

    def __init__(self, cfg: dict[str, Any], engine_path: str | Path):
        import tensorrt as trt

        self.cfg = cfg
        self.input_shape = bundle_input_shape(cfg)
        self.normalize_mean, self.normalize_std = bundle_normalization(cfg)
        self.max_disp = float(cfg["model"]["max_disp"])
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
        input_names, output_names = self._get_io_names()
        if set(input_names) != {"left_image", "right_image"}:
            raise ValueError(
                f"TensorRT engine inputs must be left_image/right_image, got {input_names}"
            )
        if output_names != ["disparity"]:
            raise ValueError(
                f"TensorRT engine output must be disparity, got {output_names}"
            )
        for name in input_names:
            engine_shape = tuple(int(value) for value in self.engine.get_tensor_shape(name))
            if engine_shape != self.input_shape:
                raise ValueError(
                    f"TensorRT engine input {name!r} has shape {engine_shape}, "
                    f"but bundle metadata declares {self.input_shape}"
                )

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


def load_trt_runner(model_dir: str | Path) -> TrtRunner:
    """Load one canonical TensorRT model bundle.

    ``model_dir`` must be the exact directory containing ``metadata.yaml`` and
    ``model.engine``.  Parent-directory discovery and legacy filenames are
    intentionally unsupported.
    """
    model_dir = Path(model_dir)
    engine_path, cfg = resolve_tensorrt_engine(model_dir)
    print(f"[INFO] Loaded TensorRT engine: {engine_path.name}")
    return TrtRunner(cfg, engine_path)
