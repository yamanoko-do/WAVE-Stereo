import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from wavestereo.layers.basic_block_2d import BasicConv2d
from wavestereo.layers.cost_volume import correlation_volume
from wavestereo.layers.disp_regression import disparity_regression, ste_peak_soft_argmax
from wavestereo.layers.disp_refinement import (
    CONTEXT_UPSAMPLE_MODES,
    context_upsample,
)
from .backbone import Backbone
from .aggregation import Aggregation
from .geometry import Geo_Encoding_Volume
from .update import TransformerUpdateBlock
from .submodule import BasicConv_IN, Conv2x


class WAVEStereo(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.args = cfgs
        self.max_disp = cfgs.MAX_DISP
        self.left_att = cfgs.LEFT_ATT
        self.hidden_dim = cfgs.HIDDEN_DIM
        self.corr_radius = cfgs.CORR_RADIUS
        self.corr_levels = cfgs.CORR_LEVELS
        self.enable_noc_mask = cfgs.get('ENABLE_NOC_MASK', False)
        self.init_mode = cfgs.get('INIT_MODE', 'softargmax')  # 'softargmax' | 'peak_softargmax'
        self.context_upsample_mode = str(
            cfgs.get('CONTEXT_UPSAMPLE_MODE', 'standard')
        ).lower()
        if self.context_upsample_mode not in CONTEXT_UPSAMPLE_MODES:
            supported = ", ".join(sorted(CONTEXT_UPSAMPLE_MODES))
            raise ValueError(
                f"Unsupported CONTEXT_UPSAMPLE_MODE "
                f"{self.context_upsample_mode!r}; expected: {supported}"
            )
        self.context_upsample_sigma = float(
            cfgs.get('CONTEXT_UPSAMPLE_SIGMA', 2.0)
        )
        self.context_upsample_reference_mode = 'argmax'
        if (
            not math.isfinite(self.context_upsample_sigma)
            or self.context_upsample_sigma <= 0.0
        ):
            raise ValueError("CONTEXT_UPSAMPLE_SIGMA must be a positive finite value")

        # backbone
        self.backbone = Backbone(cfgs.get('BACKBONE', 'MobileNetv2'), pretrained=cfgs.get('BACKBONE_PRETRAINED', False))

        # aggregation
        self.cost_agg = Aggregation(in_channels=48,
                                    left_att=self.left_att,
                                    blocks=cfgs.AGGREGATION_BLOCKS,
                                    expanse_ratio=cfgs.EXPANSE_RATIO,
                                    backbone_channels=self.backbone.output_channels)
        self.hnet = nn.Sequential(
            BasicConv2d(self.backbone.output_channels[0], self.hidden_dim*2, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU),
            BasicConv2d(self.hidden_dim*2, self.hidden_dim, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU))
        self.context_zqr_conv = nn.Sequential(
            BasicConv2d(self.backbone.output_channels[0], self.hidden_dim*6, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU),
            BasicConv2d(self.hidden_dim*6, self.hidden_dim*3, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU))
        self.update_block = TransformerUpdateBlock(self.args, hidden_dim=cfgs.HIDDEN_DIM)

        self.stem_2 = nn.Sequential(
            BasicConv_IN(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32), nn.ReLU()
        )

        self.stem_1 = BasicConv_IN(3, 16, kernel_size=3, stride=1, padding=1)

        self.spx_2_gru = Conv2x(32, 32, True, bn=False)
        self.spx = nn.Sequential(
            nn.Conv2d(2*32, 32, 3, 1, 1, bias=False),
            nn.ReLU())
        self.spx_1_gru = Conv2x(32, 16, True, bn=False, keep_concat=False)
        self.spx_gru = nn.Conv2d(16, 9, kernel_size=3, stride=1, padding=1)

    def upsample_disp(self, disp, mask_feat, stem_2x, stem_1x):
        xspx = self.spx_2_gru(mask_feat, stem_2x)
        xspx = self.spx(xspx)
        xspx = self.spx_1_gru(xspx, stem_1x)
        spx_pred = self.spx_gru(xspx)
        spx_pred = F.softmax(spx_pred, 1)
        up_disp = context_upsample(
            disp * 4.,
            spx_pred,
            mode=self.context_upsample_mode,
            consistency_sigma=self.context_upsample_sigma,
            reference_mode=self.context_upsample_reference_mode,
        ).unsqueeze(1)
        return up_disp.float()

    def forward(self, data):
        image1 = data['left']
        image2 = data['right']

        stem_2x_img = self.stem_2(image1)
        stem_1x_img = self.stem_1(image1)

        features_left = self.backbone(image1)
        features_right = self.backbone(image2)

        cost_volume = correlation_volume(features_left[0], features_right[0], self.max_disp // 4)
        encoding_volume = self.cost_agg(cost_volume, features_left)
        unsqueezed_encoding = encoding_volume[0].reshape(encoding_volume[0].size(0), -1, encoding_volume[0].size(1), encoding_volume[0].size(2), encoding_volume[0].size(3))
        prob = F.softmax(encoding_volume[0], dim=1)
        init_disp = disparity_regression(prob, self.max_disp // 4)
        init_disp_up = F.interpolate(init_disp * 4., scale_factor=4, mode='bilinear', align_corners=True)

        hidden = self.hnet(features_left[0])
        net = torch.tanh(hidden)
        context = list(self.context_zqr_conv(features_left[0]).split(split_size=self.hidden_dim, dim=1))

        geo_fn = Geo_Encoding_Volume(unsqueezed_encoding.float(), radius=self.corr_radius, num_levels=self.corr_levels)

        iters = self.args.VALID_ITERS if not self.training else self.args.TRAIN_ITERS
        disp = init_disp
        disp_preds = []

        for itr in range(iters):
            disp = disp.detach()
            corr = geo_fn(disp)
            net, delta_disp, mask_feat = self.update_block(
                net, context, feat_left=features_left[0],
                feat_right=features_right[0], disp=disp, corr=corr, itr=itr,
            )
            disp = disp + delta_disp
            if not self.training and itr < iters - 1:
                continue
            disp_up = self.upsample_disp(disp, mask_feat, stem_2x_img, stem_1x_img)
            disp_preds.append(disp_up)

        if not self.training:
            return {'disp_preds': disp_preds, 'disp_pred': disp_preds[-1]}

        init_disp = init_disp_up
        return {
            'init_disp': init_disp,
            'disp_preds': disp_preds,
        }

    def forward_debug(self, data):
        """Specialized forward for infercam debug. Returns prob, disp_low_res,
        init_disp_up alongside the usual outputs. Zero overhead over forward().
        """
        image1 = data['left']
        image2 = data['right']

        stem_2x_img = self.stem_2(image1)
        stem_1x_img = self.stem_1(image1)

        features_left = self.backbone(image1)
        features_right = self.backbone(image2)

        cost_volume = correlation_volume(features_left[0], features_right[0], self.max_disp // 4)
        encoding_volume = self.cost_agg(cost_volume, features_left)
        prob = F.softmax(encoding_volume[0], dim=1)
        init_disp = disparity_regression(prob, self.max_disp // 4)
        init_disp_up = F.interpolate(init_disp * 4., scale_factor=4, mode='bilinear', align_corners=True)

        hidden = self.hnet(features_left[0])
        net = torch.tanh(hidden)
        context = list(self.context_zqr_conv(features_left[0]).split(split_size=self.hidden_dim, dim=1))

        unsqueezed_encoding = encoding_volume[0].reshape(
            encoding_volume[0].size(0), -1, encoding_volume[0].size(1),
            encoding_volume[0].size(2), encoding_volume[0].size(3))
        geo_fn = Geo_Encoding_Volume(unsqueezed_encoding.float(), radius=self.corr_radius,
                                     num_levels=self.corr_levels)

        iters = self.args.VALID_ITERS if not self.training else self.args.TRAIN_ITERS
        disp = init_disp
        disp_preds = []

        for itr in range(iters):
            disp = disp.detach()
            corr = geo_fn(disp)
            net, delta_disp, mask_feat = self.update_block(
                net, context, feat_left=features_left[0],
                feat_right=features_right[0], disp=disp, corr=corr, itr=itr,
            )
            disp = disp + delta_disp
            if itr < iters - 1:
                continue
            disp_up = self.upsample_disp(disp, mask_feat, stem_2x_img, stem_1x_img)
            disp_preds.append(disp_up)

        # Handle iters=0: no GRU refinement, just upsample init_disp
        if len(disp_preds) == 0:
            disp_preds = [F.interpolate(init_disp * 4., scale_factor=4, mode='bilinear',
                                        align_corners=True)]
            disp_low_res = init_disp * 4.
        else:
            disp_low_res = disp * 4.

        return {
            'disp_preds': disp_preds,
            'disp_pred': disp_preds[-1],
            'disp_low_res': disp_low_res,     # H/4 raw post-GRU disp
            'init_disp_up': init_disp_up,     # pre-GRU init, bilinear upsampled
            'prob': prob,                     # [B,48,H/4,W/4] softmax cost volume
        }

    @staticmethod
    def _compute_gradient_loss(pred, gt, valid):
        """计算视差梯度 L1 loss，x/y 方向独立返回。

        x 方向 = 极线方向（水平视差跳变），y 方向 = 垂直方向（纵轴一致性）。
        两者物理含义不同，分开返回便于独立加权和分别监控。

        Returns:
            (loss_x, loss_y): x/y 方向未加权的 L1 loss
        """
        # x 方向梯度（水平相邻像素差分）
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        gt_dx = gt[:, :, :, 1:] - gt[:, :, :, :-1]
        valid_dx = valid[:, :, :, 1:] & valid[:, :, :, :-1]

        # y 方向梯度（垂直相邻像素差分）
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        gt_dy = gt[:, :, 1:, :] - gt[:, :, :-1, :]
        valid_dy = valid[:, :, 1:, :] & valid[:, :, :-1, :]

        loss_x = F.l1_loss(pred_dx[valid_dx], gt_dx[valid_dx]) if valid_dx.any() else torch.tensor(0.0, device=pred.device)
        loss_y = F.l1_loss(pred_dy[valid_dy], gt_dy[valid_dy]) if valid_dy.any() else torch.tensor(0.0, device=pred.device)

        return loss_x, loss_y

    def get_loss(self, model_pred, input_data):
        disp_gt = input_data["disp"]
        # 监督上限放宽到 999（原为 self.max_disp），仅过滤无效像素与极端离群
        loss_max_disp = 999
        disp_gt = disp_gt.unsqueeze(1)
        valid = ((disp_gt > 0) & (disp_gt < loss_max_disp))

        # 梯度 loss 权重
        edge_weight = self.args.get('EDGE_WEIGHT', 1.0)

        disp_init_pred = model_pred['init_disp']
        loss_disp = 1.0 * F.smooth_l1_loss(disp_init_pred[valid.bool()], disp_gt[valid.bool()], reduction='mean')
        edge_x, edge_y = self._compute_gradient_loss(disp_init_pred, disp_gt, valid)
        disp_loss = loss_disp + edge_weight * (edge_x + edge_y)

        loss_gamma = 0.9
        disp_preds = model_pred['disp_preds']
        n_predictions = len(disp_preds)
        for i in range(n_predictions):
            adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions - 1))
            i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)
            i_l1 = i_weight * (disp_preds[i] - disp_gt).abs()[valid.bool()].mean()
            i_ex, i_ey = self._compute_gradient_loss(disp_preds[i], disp_gt, valid)
            i_edge = i_weight * edge_weight * (i_ex + i_ey)
            loss_disp += i_l1
            disp_loss += i_l1 + i_edge

        loss_info = {
            'scalar/train/loss_disp_l1': loss_disp.item(),
            'scalar/train/loss_disp_edge_x': edge_x.item(),
            'scalar/train/loss_disp_edge_y': edge_y.item(),
        }

        loss_info['scalar/train/loss_disp'] = disp_loss.item()

        return disp_loss, loss_info
