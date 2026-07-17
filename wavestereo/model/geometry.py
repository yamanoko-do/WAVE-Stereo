import torch
import torch.nn.functional as F
from .utils import bilinear_sampler


class Geo_Encoding_Volume:
    """
    根据视差图在多级代价体(num_levels)的视差维度上索引半径radius内的特征。
    输出维度 num_levels*C*(2r+1)。

    TRT-friendly 实现: 使用 gather + 手动线性插值替代 grid_sample,
    避免 B*H*W reshape, 使整个 post_runner 可导出为 TensorRT 引擎。
    数学上与原始 grid_sample 实现完全等价。
    """
    def __init__(self, geo_volume, num_levels=2, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        # geo_volume: [B, C, D, H, W]
        # 沿 D 维度构建金字塔 (avg pool)
        self.geo_volume_pyramid = [geo_volume]
        vol = geo_volume
        for _ in range(num_levels - 1):
            vol = F.avg_pool3d(vol, kernel_size=(2, 1, 1), stride=(2, 1, 1))
            self.geo_volume_pyramid.append(vol)
        self.dx = torch.linspace(-radius, radius, 2 * radius + 1).to(geo_volume.device)

    def __call__(self, disp):
        # disp: [B, 1, H, W]
        b, _, h, w = disp.shape
        r = self.radius
        out_pyramid = []

        for i in range(self.num_levels):
            vol = self.geo_volume_pyramid[i]  # [B, C, D_i, H, W]
            D_i = vol.shape[2]
            C = vol.shape[1]

            # 采样位置: disp/2^i + dx[k]  ->  [B, 2r+1, H, W]
            sample_pos = disp.squeeze(1).unsqueeze(1) / (2 ** i) + self.dx.view(1, -1, 1, 1)

            # 1D 双线性插值 (等价于 grid_sample padding_mode='zeros', align_corners=True)
            pos_floor = torch.floor(sample_pos).long()
            pos_ceil = pos_floor + 1
            alpha = sample_pos - pos_floor.float()

            # 各索引独立判断是否在有效范围内 (越界贡献 0, 与 grid_sample zeros padding 一致)
            fl_valid = (pos_floor >= 0) & (pos_floor < D_i)  # [B, 2r+1, H, W]
            ce_valid = (pos_ceil >= 0) & (pos_ceil < D_i)

            # gather 需要合法索引, clamp 后再用 mask 置零
            idx_fl = pos_floor.clamp(0, D_i - 1).unsqueeze(1).expand(-1, C, -1, -1, -1)
            idx_ce = pos_ceil.clamp(0, D_i - 1).unsqueeze(1).expand(-1, C, -1, -1, -1)

            val_fl = torch.gather(vol, 2, idx_fl) * fl_valid.unsqueeze(1).float()
            val_ce = torch.gather(vol, 2, idx_ce) * ce_valid.unsqueeze(1).float()

            sampled = val_fl * (1 - alpha.unsqueeze(1)) + val_ce * alpha.unsqueeze(1)
            # [B, C, 2r+1, H, W] -> [B, C*(2r+1), H, W]
            sampled = sampled.reshape(b, C * (2 * r + 1), h, w)
            out_pyramid.append(sampled)

        return torch.cat(out_pyramid, dim=1).contiguous().float()  # [B, num_levels*C*(2r+1), H, W]
