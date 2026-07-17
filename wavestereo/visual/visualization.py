"""
Visualization Module - Utility functions for visualization
"""
import os
from typing import Optional, Tuple, Dict, Any

import numpy as np
import cv2

# Set environment variables for Qt/Rerun
os.environ["XDG_SESSION_TYPE"] = "x11"
os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"


def create_disparity_colormap(disp_vis_cpu: np.ndarray,
                               colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    Create color disparity map from visualization disparity

    Args:
        disp_vis_cpu: Visualization disparity (uint8, 0-255)
        colormap: OpenCV colormap

    Returns:
        Color disparity image
    """
    return cv2.applyColorMap(disp_vis_cpu, colormap)


def resize_disparity(disp_vis_cpu: np.ndarray, disp_raw_cpu: np.ndarray,
                     target_size: Tuple[int, int],
                     resize_factor: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resize disparity maps to target size

    Args:
        disp_vis_cpu: Visualization disparity
        disp_raw_cpu: Raw disparity
        target_size: Target (width, height)
        resize_factor: Resize factor (for raw disparity scaling)

    Returns:
        (disp_vis_resized, disp_raw_resized)
    """
    orig_w, orig_h = target_size

    disp_vis_resized = cv2.resize(disp_vis_cpu, (orig_w, orig_h))

    disp_raw_resized = cv2.resize(disp_raw_cpu, (orig_w, orig_h),
                                   interpolation=cv2.INTER_LINEAR)

    if resize_factor != 1.0:
        disp_raw_resized /= resize_factor

    return disp_vis_resized, disp_raw_resized


def compute_gradient_vis(disp: np.ndarray, max_grad: float = 0.0,
                         gamma: float = 0.35) -> np.ndarray:
    """
    Compute disparity gradient magnitude visualization.

    与 _compute_gradient_loss 思路一致：
    - dx = |disp[:, 1:] - disp[:, :-1]|
    - dy = |disp[1:, :] - disp[:-1, :]|
    - mag = sqrt(dx^2 + dy^2)

    非线性映射 (gamma < 1)：压缩低梯度区域，将更多色阶分配给大梯度（边缘）。

    Args:
        disp: Raw disparity map (H, W) float32
        max_grad: Optional pre-computed max for consistent colormap across frames.
                  If 0, uses this frame's own max.
        gamma: Power-law exponent. < 1 → 更多颜色给大梯度（推荐 0.3~0.5）.
               = 1 → 线性映射. > 1 → 更多颜色给小梯度.

    Returns:
        grad_vis: uint8 gradient magnitude visualization (H, W)
        grad_mag: float32 gradient magnitude (H, W) — for thresholding / debug
    """
    # x-direction gradient (right - left), pad last column with 0
    dx = np.abs(np.diff(disp, axis=1))
    dx = np.pad(dx, ((0, 0), (0, 1)), mode='edge')

    # y-direction gradient (bottom - top), pad last row with 0
    dy = np.abs(np.diff(disp, axis=0))
    dy = np.pad(dy, ((0, 1), (0, 0)), mode='edge')

    # Gradient magnitude
    mag = np.sqrt(dx.astype(np.float64) ** 2 + dy.astype(np.float64) ** 2).astype(np.float32)

    # Normalize to [0, 255] with power-law curve
    m = max_grad if max_grad > 0 else mag.max()
    if m > 1e-6:
        # (mag/m)^gamma — gamma<1 拉伸高梯度区，压缩低梯度区
        vis = ((mag / m) ** gamma * 255).clip(0, 255).astype(np.uint8)
    else:
        vis = np.zeros_like(mag, dtype=np.uint8)

    return vis, mag


def check_and_fix_invalid(disp: np.ndarray, name: str = "disparity") -> np.ndarray:
    """
    Check for NaN/Inf values and fix them

    Args:
        disp: Disparity array
        name: Name for logging

    Returns:
        Fixed disparity array
    """
    has_nan = np.isnan(disp).any()
    has_inf = np.isinf(disp).any()

    if has_nan or has_inf:
        invalid_mask = np.isnan(disp) | np.isinf(disp)
        invalid_count = invalid_mask.sum()
        print(f"[WARN] {name}: detected invalid values! NaN: {has_nan}, Inf: {has_inf}, count: {invalid_count}")
        disp = disp.copy()
        disp[invalid_mask] = 0.0

    return disp
