from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image


@dataclass
class PadInfo:
    top: int
    bottom: int
    left: int
    right: int
    original_hw: tuple[int, int]


def load_image(image: str | Path | Image.Image | np.ndarray, rgb_input: bool = True) -> np.ndarray:
    if isinstance(image, (str, Path)):
        arr = np.array(Image.open(image).convert("RGB"))
    elif isinstance(image, Image.Image):
        arr = np.array(image.convert("RGB"))
    elif isinstance(image, np.ndarray):
        arr = image
    else:
        raise TypeError(f"Unsupported image type: {type(image)!r}")

    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 image, got shape {arr.shape}")
    if not rgb_input:
        arr = arr[..., ::-1]
    return arr.astype(np.float32, copy=False)


class StereoPreprocessor:
    def __init__(
        self,
        size: Iterable[int] | None = None,
        pad_mode: str = "right_top",
        divisible_by: int = 32,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        device: str | torch.device = "cuda",
        rgb_input: bool = True,
        normalize: bool = True,
    ):
        self.size = tuple(size) if size else None
        self.pad_mode = pad_mode.lower().replace("-", "_")
        self.divisible_by = divisible_by
        self.device = torch.device(device)
        self.rgb_input = rgb_input
        self.normalize = normalize
        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)

    def _target_size(self, h: int, w: int) -> tuple[int, int]:
        if self.size is not None:
            return int(self.size[0]), int(self.size[1])
        by = self.divisible_by
        return ((h + by - 1) // by) * by, ((w + by - 1) // by) * by

    def _pad(self, img: np.ndarray, target_h: int, target_w: int) -> tuple[np.ndarray, PadInfo]:
        h, w = img.shape[:2]
        if target_h < h or target_w < w:
            raise ValueError(f"Target size {(target_h, target_w)} is smaller than image {(h, w)}")
        pad_h = target_h - h
        pad_w = target_w - w
        if self.pad_mode == "right_top":
            top, bottom, left, right = pad_h, 0, 0, pad_w
        elif self.pad_mode == "right_bottom":
            top, bottom, left, right = 0, pad_h, 0, pad_w
        elif self.pad_mode == "center":
            top, bottom = pad_h // 2, pad_h - pad_h // 2
            left, right = pad_w // 2, pad_w - pad_w // 2
        elif self.pad_mode == "divisible":
            top, bottom, left, right = 0, pad_h, 0, pad_w
        else:
            raise ValueError(f"Unsupported pad_mode: {self.pad_mode}")
        padded = np.pad(img, ((top, bottom), (left, right), (0, 0)), mode="edge")
        return padded, PadInfo(top, bottom, left, right, (h, w))

    def prepare(self, left: Any, right: Any) -> tuple[dict[str, Any], PadInfo]:
        left_arr = load_image(left, rgb_input=self.rgb_input)
        right_arr = load_image(right, rgb_input=self.rgb_input)
        if left_arr.shape != right_arr.shape:
            raise ValueError(f"Left/right image shapes differ: {left_arr.shape} vs {right_arr.shape}")

        target_h, target_w = self._target_size(*left_arr.shape[:2])
        left_pad, pad = self._pad(left_arr, target_h, target_w)
        right_pad, _ = self._pad(right_arr, target_h, target_w)

        left_tensor = torch.from_numpy(left_pad).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        right_tensor = torch.from_numpy(right_pad).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        if self.normalize:
            mean = self.mean.to(left_tensor.device)
            std = self.std.to(left_tensor.device)
            left_tensor = (left_tensor - mean) / std
            right_tensor = (right_tensor - mean) / std
        left_tensor = left_tensor.to(self.device, non_blocking=True)
        right_tensor = right_tensor.to(self.device, non_blocking=True)
        return {"left": left_tensor, "right": right_tensor, "pad": [pad.top, pad.right]}, pad

    @staticmethod
    def crop_disparity(disparity: np.ndarray, pad: PadInfo) -> np.ndarray:
        h, w = pad.original_hw
        return disparity[pad.top:pad.top + h, pad.left:pad.left + w]
