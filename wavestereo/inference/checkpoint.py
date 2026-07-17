from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


_STATE_KEYS = ("model", "state_dict", "model_state", "model_state_dict", "net", "network")


def _looks_like_state_dict(obj: Any) -> bool:
    return isinstance(obj, dict) and obj and all(hasattr(v, "shape") for v in obj.values())


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if _looks_like_state_dict(checkpoint):
        return checkpoint
    if isinstance(checkpoint, dict):
        for key in _STATE_KEYS:
            value = checkpoint.get(key)
            if _looks_like_state_dict(value):
                return value
        for value in checkpoint.values():
            if _looks_like_state_dict(value):
                return value
    raise ValueError("Could not find a model state_dict in checkpoint")


def strip_prefix_if_present(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    if not any(k.startswith(prefix) for k in state_dict):
        return state_dict
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path, strict: bool = False, map_location="cpu"):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path:
        raise ValueError("checkpoint_path is empty")
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = extract_state_dict(checkpoint)
    for prefix in ("module.", "model."):
        state_dict = strip_prefix_if_present(state_dict, prefix)

    model_keys = set(model.state_dict().keys())
    matched = len(model_keys.intersection(state_dict.keys()))
    if matched == 0:
        raise RuntimeError(f"No checkpoint keys matched model keys: {checkpoint_path}")

    result = model.load_state_dict(state_dict, strict=strict)
    print(
        f"[INFO] Loaded checkpoint {checkpoint_path} "
        f"({matched}/{len(model_keys)} model keys matched, strict={strict})"
    )
    if result.missing_keys:
        print(f"[WARN] Missing keys: {len(result.missing_keys)}")
    if result.unexpected_keys:
        print(f"[WARN] Unexpected keys: {len(result.unexpected_keys)}")
    return result
