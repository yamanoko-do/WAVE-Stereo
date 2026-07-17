import torch


def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=True)


def ste_peak_soft_argmax(prob, window_radius=3.0):
    b, d, h, w = prob.shape
    idx = torch.arange(d, device=prob.device, dtype=prob.dtype).view(1, d, 1, 1)
    peak = prob.argmax(dim=1).float().unsqueeze(1)
    inv_r_sq = 1.0 / (window_radius * window_radius + 1e-6)
    dist = idx - peak
    mask = torch.relu(1.0 - (dist * dist) * inv_r_sq)
    masked_prob = prob * mask
    numerator = (masked_prob * idx).sum(dim=1, keepdim=True)
    denominator = masked_prob.sum(dim=1, keepdim=True) + 1e-6
    return numerator / denominator
