from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import cv2
import numpy as np


_ADAPTIVE_KERNEL = np.ones((3, 3), dtype=np.uint8)
_UPSAMPLE_METHODS = frozenset({"nearest", "bilinear", "adaptive"})


def _validate_size(size: tuple[int, int], name: str) -> tuple[int, int]:
    if len(size) != 2:
        raise ValueError(f"{name} must be (height, width), got {size!r}")
    height, width = size
    if (
        isinstance(height, bool)
        or isinstance(width, bool)
        or not isinstance(height, int)
        or not isinstance(width, int)
        or height <= 0
        or width <= 0
    ):
        raise ValueError(f"{name} must contain positive integers, got {size!r}")
    return height, width


def _validate_bgr_image(image: np.ndarray, name: str) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{name} must have shape HxWx3, got {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"{name} must use uint8 BGR pixels, got {image.dtype}")


@dataclass(frozen=True)
class SpatialTransform:
    """Describe one source-to-model image transform.

    All size tuples and named dimensions use ``(height, width)`` ordering.
    The model receives a direct bilinear resize of the source image. Internal
    model alignment padding, when required, belongs to the model graph or the
    PyTorch preprocessor and is intentionally not represented here.
    """

    source_height: int
    source_width: int
    model_height: int
    model_width: int

    def __post_init__(self) -> None:
        _validate_size(self.source_size, "source_size")
        _validate_size(self.model_size, "model_size")

    @classmethod
    def from_sizes(
        cls,
        source_size: tuple[int, int],
        model_size: tuple[int, int],
    ) -> "SpatialTransform":
        source_height, source_width = _validate_size(source_size, "source_size")
        model_height, model_width = _validate_size(model_size, "model_size")
        return cls(source_height, source_width, model_height, model_width)

    @property
    def source_size(self) -> tuple[int, int]:
        return self.source_height, self.source_width

    @property
    def model_size(self) -> tuple[int, int]:
        return self.model_height, self.model_width

    @property
    def requires_resize(self) -> bool:
        return self.source_size != self.model_size

    @property
    def width_scale(self) -> float:
        """Convert disparity measured in model pixels to source pixels."""
        return self.source_width / self.model_width

    def prepare_image(self, image: np.ndarray, *, name: str = "image") -> np.ndarray:
        """Return an exact model-size contiguous uint8 BGR image.

        A camera capture path may already have fused rectification and resize.
        Such model-size input is validated and returned without another resize.
        Otherwise the image must have the declared source size and is resized
        exactly once.
        """
        _validate_bgr_image(image, name)
        actual_size = image.shape[:2]
        if actual_size == self.model_size:
            return np.ascontiguousarray(image)
        if actual_size != self.source_size:
            raise ValueError(
                f"{name} size must be source {self.source_width}x{self.source_height} "
                f"or model {self.model_width}x{self.model_height}, got "
                f"{actual_size[1]}x{actual_size[0]}"
            )
        return cv2.resize(
            image,
            (self.model_width, self.model_height),
            interpolation=cv2.INTER_LINEAR,
        )

    def prepare_stereo(
        self,
        frame: Mapping[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        try:
            left = frame["left"]
            right = frame["right"]
        except KeyError as exc:
            raise ValueError("Stereo frame must contain left and right images") from exc
        if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
            raise TypeError("Stereo frame images must be numpy arrays")
        if left.shape != right.shape:
            raise ValueError(
                f"Left/right image shapes differ: {left.shape} vs {right.shape}"
            )
        return {
            "left": self.prepare_image(left, name="left image"),
            "right": self.prepare_image(right, name="right image"),
        }


@dataclass(frozen=True)
class PostprocessedDisparity:
    disparity: np.ndarray
    effective_max_disp: float


class DisparityPostprocessor:
    """Restore one native model disparity to the declared source geometry."""

    def __init__(self, method: str = "adaptive", sigma: float = 2.0):
        method = str(method).lower()
        if method not in _UPSAMPLE_METHODS:
            supported = ", ".join(sorted(_UPSAMPLE_METHODS))
            raise ValueError(
                f"Unsupported disparity upsampling method {method!r}; "
                f"expected one of: {supported}"
            )
        sigma = float(sigma)
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("Disparity upsampling sigma must be positive and finite")
        self.method = method
        self.sigma = sigma

    def restore(
        self,
        native_disparity: np.ndarray,
        transform: SpatialTransform,
        max_disp: float,
    ) -> PostprocessedDisparity:
        native = np.asarray(native_disparity)
        if native.ndim != 2:
            raise ValueError(
                f"Native disparity must have shape HxW, got {native.shape}"
            )
        if native.shape != transform.model_size:
            raise ValueError(
                f"Native disparity shape {native.shape} does not match model size "
                f"{transform.model_size}"
            )
        max_disp = float(max_disp)
        if not np.isfinite(max_disp) or max_disp <= 0.0:
            raise ValueError("max_disp must be positive and finite")

        # This is the single cleanup and validity boundary for all backends.
        # Backends intentionally return unmodified native model output.
        source = native.astype(np.float32, copy=True)
        np.nan_to_num(source, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(source, 0.0, max_disp, out=source)

        width_scale = np.float32(transform.width_scale)
        source *= width_scale
        effective_max_disp = max_disp * transform.width_scale

        if not transform.requires_resize:
            return PostprocessedDisparity(source, effective_max_disp)

        output_size = (transform.source_width, transform.source_height)
        if self.method == "nearest":
            restored = cv2.resize(
                source, output_size, interpolation=cv2.INTER_NEAREST
            )
        elif self.method == "bilinear":
            restored = cv2.resize(
                source, output_size, interpolation=cv2.INTER_LINEAR
            )
        else:
            restored = self._adaptive_resize(source, output_size)
        return PostprocessedDisparity(
            np.asarray(restored, dtype=np.float32),
            effective_max_disp,
        )

    def _adaptive_resize(
        self,
        source: np.ndarray,
        output_size: tuple[int, int],
    ) -> np.ndarray:
        nearest = cv2.resize(
            source, output_size, interpolation=cv2.INTER_NEAREST
        )
        bilinear = cv2.resize(
            source, output_size, interpolation=cv2.INTER_LINEAR
        )

        # Estimate continuity at native resolution. Smooth regions retain the
        # bilinear result; depth discontinuities approach nearest-neighbour.
        max_difference = cv2.dilate(source, _ADAPTIVE_KERNEL)
        max_difference -= source
        negative_difference = cv2.erode(source, _ADAPTIVE_KERNEL)
        cv2.subtract(source, negative_difference, dst=negative_difference)
        np.maximum(max_difference, negative_difference, out=max_difference)

        exponent_scale = np.float32(-0.5 / (self.sigma * self.sigma))
        max_difference *= max_difference
        max_difference *= exponent_scale
        np.exp(max_difference, out=max_difference)
        confidence = cv2.resize(
            max_difference,
            output_size,
            interpolation=cv2.INTER_NEAREST,
        )

        bilinear -= nearest
        bilinear *= confidence
        nearest += bilinear
        return nearest
