from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from wavestereo.config import Config, load_config
from wavestereo.inference.checkpoint import load_checkpoint
from wavestereo.inference.preprocess import (
    StereoPreprocessor,
    normalization_from_config,
)
from wavestereo.model import WAVEStereo


_AMP_DTYPES = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


class WAVEStereoPredictor:
    def __init__(
        self,
        config: str | Path | Config,
        weights: str | Path | None = None,
        device: str | torch.device | None = None,
        amp: bool | None = None,
        amp_dtype: str | torch.dtype | None = None,
        strict: bool = False,
    ):
        self.cfg = load_config(config) if not isinstance(config, Config) else config
        infer_cfg = self.cfg.get("INFERENCE", Config())
        self.device = torch.device(device or infer_cfg.get("DEVICE", "cuda"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            print("[WARN] CUDA requested but unavailable; falling back to CPU")
            self.device = torch.device("cpu")

        self.amp = bool(infer_cfg.get("AMP", True) if amp is None else amp)
        dtype_value = amp_dtype or infer_cfg.get("AMP_DTYPE", "bfloat16")
        self.amp_dtype = dtype_value if isinstance(dtype_value, torch.dtype) else _AMP_DTYPES[str(dtype_value).lower()]

        self.model = WAVEStereo(self.cfg.MODEL).to(self.device).eval()
        weight_path = weights or self.cfg.MODEL.get("PRETRAINED_MODEL", "")
        if weight_path:
            load_checkpoint(self.model, weight_path, strict=strict, map_location=self.device)
        else:
            print("[WARN] No weights provided; model will use random initialization")

        mean, std = normalization_from_config(infer_cfg)
        self.preprocessor = StereoPreprocessor(
            size=infer_cfg.get("SIZE", None),
            pad_mode=infer_cfg.get("PAD_MODE", "right_top"),
            divisible_by=int(infer_cfg.get("DIVISIBLE_BY", 32)),
            mean=mean,
            std=std,
            device=self.device,
        )
        self.last_inference_time = 0.0

    @torch.no_grad()
    def predict(self, left: Any, right: Any) -> np.ndarray:
        sample, pad = self.preprocessor.prepare(left, right)
        autocast_enabled = self.amp and self.device.type == "cuda" and self.amp_dtype is not torch.float32
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start_time = perf_counter()
        with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=autocast_enabled):
            pred = self.model(sample)["disp_pred"]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.last_inference_time = perf_counter() - start_time
        disparity = pred.squeeze().detach().float().cpu().numpy()
        return self.preprocessor.crop_disparity(disparity, pad).astype(np.float32, copy=False)
