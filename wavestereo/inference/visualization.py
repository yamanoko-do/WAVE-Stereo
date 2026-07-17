from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def check_and_fix_invalid(disp: np.ndarray, name: str = "disparity") -> np.ndarray:
    invalid = np.isnan(disp) | np.isinf(disp)
    if invalid.any():
        print(f"[WARN] {name}: fixing {int(invalid.sum())} invalid values")
        disp = disp.copy()
        disp[invalid] = 0.0
    return disp


def disparity_to_uint8(disp: np.ndarray, max_disp: float | None = None) -> np.ndarray:
    disp = check_and_fix_invalid(disp)
    if max_disp is None or max_disp <= 0:
        lo, hi = float(np.min(disp)), float(np.max(disp))
        if hi - lo < 1e-6:
            return np.zeros_like(disp, dtype=np.uint8)
        return ((disp - lo) / (hi - lo) * 255.0).clip(0, 255).astype(np.uint8)
    return (disp * (255.0 / max_disp)).clip(0, 255).astype(np.uint8)


def colorize_disparity(disp: np.ndarray, max_disp: float | None = None, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    return cv2.applyColorMap(disparity_to_uint8(disp, max_disp), colormap)


def save_outputs(prefix: str | Path, disparity: np.ndarray, max_disp: float | None = None, save_npy: bool = True) -> None:
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    if save_npy:
        np.save(prefix.with_suffix(".npy"), disparity.astype(np.float32))
    cv2.imwrite(str(prefix.with_suffix(".png")), colorize_disparity(disparity, max_disp))
