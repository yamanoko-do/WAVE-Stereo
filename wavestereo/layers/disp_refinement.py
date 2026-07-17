import torch.nn.functional as F


def context_upsample(disp_low, up_weights, scale_factor=4):
    b, c, h, w = disp_low.shape
    disp_unfold = F.unfold(disp_low, kernel_size=3, dilation=1, padding=1)
    disp_unfold = disp_unfold.reshape(b, -1, h, w)
    disp_unfold = F.interpolate(disp_unfold, (h * scale_factor, w * scale_factor), mode='nearest')
    return (disp_unfold * up_weights).sum(1)
