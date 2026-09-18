import math

import torch
import torch.nn.functional as F


CONTEXT_UPSAMPLE_MODES = frozenset({"standard", "consistency_guided"})
CONTEXT_UPSAMPLE_REFERENCE_MODES = frozenset({"argmax", "npu_reduce_max"})


def _unfold_disparity(disp_low):
    b, _, h, w = disp_low.shape
    candidates = F.unfold(disp_low, kernel_size=3, dilation=1, padding=1)
    return candidates.reshape(b, 9, h, w)


def _resize_candidates(candidates, height, width, scale_factor):
    return F.interpolate(
        candidates,
        (height * scale_factor, width * scale_factor),
        mode="nearest",
    )


def _standard_context_upsample(disp_low, up_weights, scale_factor):
    _, _, h, w = disp_low.shape
    candidates = _resize_candidates(
        _unfold_disparity(disp_low), h, w, scale_factor
    )
    return (candidates * up_weights).sum(dim=1)


def _consistency_guided_context_upsample(
    disp_low,
    up_weights,
    scale_factor,
    consistency_sigma,
    reference_mode,
):
    """Reweight learned candidates using full-resolution disparity consistency."""
    _, _, h, w = disp_low.shape
    candidates = _resize_candidates(
        _unfold_disparity(disp_low), h, w, scale_factor
    )

    if reference_mode == "argmax":
        reference_indices = up_weights.argmax(dim=1, keepdim=True)
        reference_disparity = torch.gather(candidates, 1, reference_indices)
    elif reference_mode == "npu_reduce_max":
        # Select the same first maximum without exporting the very slow NPU
        # ArgMax + GatherElements pair. Descending priorities preserve
        # argmax's first-index tie behavior exactly.
        maximum_weights = up_weights.amax(dim=1, keepdim=True)
        maximum_mask = up_weights == maximum_weights
        channel_priority = torch.arange(
            9,
            0,
            -1,
            dtype=up_weights.dtype,
            device=up_weights.device,
        ).reshape(1, 9, 1, 1)
        selected_priority = (
            maximum_mask.to(up_weights.dtype) * channel_priority
        ).amax(dim=1, keepdim=True)
        reference_mask = (channel_priority == selected_priority).to(
            up_weights.dtype
        )
        reference_disparity = (candidates * reference_mask).sum(dim=1, keepdim=True)
    else:
        supported = ", ".join(sorted(CONTEXT_UPSAMPLE_REFERENCE_MODES))
        raise ValueError(
            f"Unsupported context upsample reference mode {reference_mode!r}; "
            f"expected: {supported}"
        )
    difference = candidates - reference_disparity

    exponent_scale = -0.5 / (consistency_sigma * consistency_sigma)
    consistency = torch.exp(difference.square() * exponent_scale)

    clean_weights = up_weights * consistency
    numerator = (candidates * clean_weights).sum(dim=1)
    denominator = clean_weights.sum(dim=1) + 1e-6
    return numerator / denominator


def context_upsample(
    disp_low,
    up_weights,
    scale_factor=4,
    mode="standard",
    consistency_sigma=2.0,
    reference_mode="argmax",
):
    if not isinstance(scale_factor, int) or scale_factor <= 0:
        raise ValueError("context_upsample scale_factor must be a positive integer")
    if mode == "standard":
        return _standard_context_upsample(disp_low, up_weights, scale_factor)
    if mode == "consistency_guided":
        consistency_sigma = float(consistency_sigma)
        if not math.isfinite(consistency_sigma) or consistency_sigma <= 0.0:
            raise ValueError(
                "context_upsample consistency_sigma must be positive and finite"
            )
        return _consistency_guided_context_upsample(
            disp_low,
            up_weights,
            scale_factor,
            consistency_sigma,
            reference_mode,
        )
    supported = ", ".join(sorted(CONTEXT_UPSAMPLE_MODES))
    raise ValueError(
        f"Unsupported context upsample mode {mode!r}; expected: {supported}"
    )
