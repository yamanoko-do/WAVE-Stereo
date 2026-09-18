import pytest
import torch
import torch.nn.functional as F

from wavestereo.layers.disp_refinement import context_upsample


def _weights(height, width, scale_factor=4):
    logits = torch.randn(1, 9, height * scale_factor, width * scale_factor)
    return torch.softmax(logits, dim=1)


def test_standard_context_upsample_matches_legacy_implementation_exactly():
    torch.manual_seed(3)
    disparity = torch.rand(1, 1, 5, 7)
    weights = _weights(5, 7)

    candidates = F.unfold(disparity, kernel_size=3, padding=1).reshape(1, 9, 5, 7)
    candidates = F.interpolate(candidates, (20, 28), mode="nearest")
    expected = (candidates * weights).sum(dim=1)

    actual = context_upsample(disparity, weights, mode="standard")

    assert torch.equal(actual, expected)


def test_consistency_guided_matches_standard_in_constant_region():
    disparity = torch.full((1, 1, 5, 7), 12.0)
    weights = _weights(5, 7)

    standard = context_upsample(disparity, weights, mode="standard")
    guided = context_upsample(
        disparity,
        weights,
        mode="consistency_guided",
        consistency_sigma=2.0,
    )

    # Ignore the outer four output pixels because unfold intentionally uses
    # zero padding there; the interior candidates are all exactly equal.
    torch.testing.assert_close(guided[:, 4:-4, 4:-4], standard[:, 4:-4, 4:-4])


def test_consistency_guided_matches_full_resolution_reference_formula():
    torch.manual_seed(7)
    disparity = torch.rand(1, 1, 3, 5) * 64.0
    weights = _weights(3, 5)
    sigma = 2.0

    candidates = F.unfold(disparity, kernel_size=3, padding=1).reshape(1, 9, 3, 5)
    candidates = F.interpolate(candidates, (12, 20), mode="nearest")
    reference_indices = weights.argmax(dim=1, keepdim=True)
    reference = torch.gather(candidates, 1, reference_indices)
    consistency = torch.exp(-((candidates - reference) ** 2) / (2.0 * sigma ** 2))
    clean_weights = weights * consistency
    clean_weights = clean_weights / (clean_weights.sum(dim=1, keepdim=True) + 1e-6)
    expected = (candidates * clean_weights).sum(dim=1)

    actual = context_upsample(
        disparity,
        weights,
        mode="consistency_guided",
        consistency_sigma=sigma,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-5)


def test_consistency_guided_preserves_first_argmax_tie_behavior():
    disparity = torch.tensor(
        [[[[10.0, 50.0, 90.0], [10.0, 50.0, 90.0], [10.0, 50.0, 90.0]]]]
    )
    weights = torch.zeros(1, 9, 12, 12)
    weights[:, 4] = 0.5
    weights[:, 5] = 0.5

    candidates = F.unfold(disparity, kernel_size=3, padding=1).reshape(1, 9, 3, 3)
    candidates = F.interpolate(candidates, (12, 12), mode="nearest")
    reference_indices = weights.argmax(dim=1, keepdim=True)
    reference = torch.gather(candidates, 1, reference_indices)
    consistency = torch.exp(-((candidates - reference) ** 2) / 8.0)
    clean_weights = weights * consistency
    clean_weights = clean_weights / (clean_weights.sum(dim=1, keepdim=True) + 1e-6)
    expected = (candidates * clean_weights).sum(dim=1)

    actual = context_upsample(
        disparity,
        weights,
        mode="consistency_guided",
        consistency_sigma=2.0,
        reference_mode="npu_reduce_max",
    )

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-5)


def test_consistency_guided_suppresses_candidate_across_depth_edge():
    disparity = torch.tensor(
        [[[[10.0, 50.0, 50.0], [10.0, 50.0, 50.0], [10.0, 50.0, 50.0]]]]
    )
    weights = torch.zeros(1, 9, 12, 12)
    weights[:, 4] = 0.6  # The strongest candidate provides the reference.
    weights[:, 5] = 0.4  # This cross-edge candidate should be suppressed.

    standard = context_upsample(disparity, weights, mode="standard")
    guided = context_upsample(
        disparity,
        weights,
        mode="consistency_guided",
        consistency_sigma=2.0,
    )

    assert standard[0, 4, 0] == 26.0
    assert guided[0, 4, 0] < 11.0


def test_context_upsample_rejects_unknown_mode():
    disparity = torch.ones(1, 1, 2, 2)
    weights = torch.full((1, 9, 8, 8), 1.0 / 9.0)

    with pytest.raises(ValueError, match="Unsupported context upsample mode"):
        context_upsample(disparity, weights, mode="unknown")


def test_consistency_guided_rejects_invalid_sigma():
    disparity = torch.ones(1, 1, 2, 2)
    weights = torch.full((1, 9, 8, 8), 1.0 / 9.0)

    with pytest.raises(ValueError, match="consistency_sigma"):
        context_upsample(
            disparity,
            weights,
            mode="consistency_guided",
            consistency_sigma=0.0,
        )


def test_consistency_guided_rejects_unknown_reference_mode():
    disparity = torch.ones(1, 1, 2, 2)
    weights = torch.full((1, 9, 8, 8), 1.0 / 9.0)

    with pytest.raises(ValueError, match="reference mode"):
        context_upsample(
            disparity,
            weights,
            mode="consistency_guided",
            reference_mode="unknown",
        )
