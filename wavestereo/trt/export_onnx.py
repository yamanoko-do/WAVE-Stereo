"""
WAVEStereo 单 ONNX 导出脚本

WAVEStereo 使用纯 PyTorch 的 correlation_volume（无 Triton GWC），
因此可以将整个模型导出为单个 ONNX 文件，无需拆分引擎。

ONNX 模型期望预归一化的 float32 输入（ImageNet 归一化在预处理中完成）：
    normalized = (pixel_0_255 / 255.0 - mean) / std
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

输入:  left_image  [1, 3, H, W]  float32, ImageNet 归一化
       right_image [1, 3, H, W]  float32, ImageNet 归一化
输出:  disparity   [1, 1, H, W]  float32

Usage:
    python tools/export_onnx.py \
        --weights /path/to/checkpoint.pth \
        --config cfgs/wavestereo/wavestereo_sceneflow_eval.yaml \
        --save_path output_wavestereo/ \
        --height 480 --width 640

    # 构建 TRT 引擎:
    trtexec --onnx=outputs/onnx/wavestereo.onnx \
        --saveEngine=outputs/onnx/wavestereo.engine --fp16
"""
import os
import argparse
import logging

os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from wavestereo.config import load_config, save_config
from wavestereo.inference.checkpoint import load_checkpoint
from wavestereo.model import WAVEStereo


# ============================================
# ONNX 兼容的 correlation_volume（向量化 + FP32 累积）
# ============================================
def correlation_volume_onnx(left_feature, right_feature, max_disp):
    """
    ONNX 兼容的 correlation volume 构建。
    向量化实现：先 stack 所有移位的 right，再批量乘 + mean。
    使用 FP32 累积提高数值精度。
    Mathematically equivalent to wavestereo.layers.cost_volume.correlation_volume.
    """
    left_f = left_feature.float()
    right_f = right_feature.float()
    B, C, H, W = left_f.shape
    shifted = [
        F.pad(right_f, (d, 0, 0, 0), 'constant', 0.0)[:, :, :, :W]
        for d in range(max_disp)
    ]
    right_volume = torch.stack(shifted, dim=2)                  # [B, C, D, H, W]
    volume = (left_f.unsqueeze(2) * right_volume).mean(dim=1)   # [B, D, H, W]
    return volume.to(left_feature.dtype)


# ============================================
# ONNX 兼容的 disp_warp
# ============================================
def disp_warp_onnx(feature, disp, padding_mode='border'):
    """
    ONNX 兼容的视差 warp。
    使用 FP32 构建网格，非原地操作，F.grid_sample 在 opset 17 下可导出。
    """
    b, _, h, w = feature.size()
    x_range = torch.arange(0, w, device=feature.device, dtype=torch.float32).view(1, 1, w).expand(1, h, w)
    y_range = torch.arange(0, h, device=feature.device, dtype=torch.float32).view(1, h, 1).expand(1, h, w)
    grid = torch.cat((x_range, y_range), dim=0)
    grid = grid.unsqueeze(0).expand(b, 2, h, w)

    disp_f = disp.float()
    offset = torch.cat((-disp_f, torch.zeros_like(disp_f)), dim=1)
    sample_grid = grid + offset

    w_minus_1 = max(w - 1, 1)
    h_minus_1 = max(h - 1, 1)
    x_norm = 2.0 * sample_grid[:, 0:1, :, :] / w_minus_1 - 1.0
    y_norm = 2.0 * sample_grid[:, 1:2, :, :] / h_minus_1 - 1.0
    sample_grid = torch.cat([x_norm, y_norm], dim=1)
    sample_grid = sample_grid.permute(0, 2, 3, 1)

    return F.grid_sample(feature, sample_grid, mode='bilinear',
                         padding_mode=padding_mode, align_corners=True)


# ============================================
# ONNX 兼容的 DPTFusionBlock forward
# ============================================
def dpt_fusion_block_forward_onnx(self, main_feat, skip_feat=None, target_size=None):
    """
    移除 torch.amp.autocast，使用 .float() 直接转换。
    """
    out = main_feat
    if skip_feat is not None:
        out = out + self.res_conf_unit1(skip_feat)
    out = self.res_conf_unit2(out)
    if target_size is None:
        target_size = (main_feat.shape[2] * 2, main_feat.shape[3] * 2)
    out = F.interpolate(out.float(), size=target_size, mode='bilinear', align_corners=False).to(main_feat.dtype)
    return self.out_conv(out)


# ============================================
# ONNX 兼容的 WAVEStereo forward（数值敏感算子强制 FP32）
# ============================================
def wavestereo_forward_onnx(self, data):
    """
    WAVEStereo ONNX 兼容 forward。
    对 softmax、disparity_regression、context_upsample 等数值敏感算子强制 FP32，
    避免 FP16 下的精度损失。
    """
    image1 = data['left']
    image2 = data['right']

    stem_2x_img = self.stem_2(image1)
    stem_1x_img = self.stem_1(image1)

    features_left = self.backbone(image1)
    features_right = self.backbone(image2)

    cost_volume = correlation_volume_onnx(features_left[0], features_right[0], self.max_disp // 4)
    encoding_volume = self.cost_agg(cost_volume, features_left)

    # FP32 softmax + disparity_regression (FP16 softmax 在大 logits 下精度差)
    enc_fp32 = encoding_volume[0].float()
    unsqueezed_encoding = enc_fp32.reshape(
        enc_fp32.size(0), -1, enc_fp32.size(1), enc_fp32.size(2), enc_fp32.size(3))
    prob = F.softmax(enc_fp32, dim=1)
    init_disp = disparity_regression_onnx(prob, self.max_disp // 4).to(encoding_volume[0].dtype)
    init_disp_up = F.interpolate(init_disp.float() * 4., scale_factor=4, mode='bilinear', align_corners=True).to(init_disp.dtype)

    hidden = self.hnet(features_left[0])
    net = torch.tanh(hidden)
    context = list(self.context_zqr_conv(features_left[0]).split(split_size=self.hidden_dim, dim=1))

    geo_fn = Geo_Encoding_Volume_onnx(unsqueezed_encoding, radius=self.corr_radius, num_levels=self.corr_levels)

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


def disparity_regression_onnx(x, maxdisp):
    """FP32 disparity regression，避免 FP16 下求和精度损失"""
    disp_values = torch.arange(0, maxdisp, dtype=torch.float32, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=True)


# ============================================
# ONNX 兼容的 Geo_Encoding_Volume（register_buffer 替代动态创建）
# ============================================
class Geo_Encoding_Volume_onnx:
    """Geo_Encoding_Volume 的 ONNX 兼容版本，与原始实现数学等价"""

    def __init__(self, geo_volume, num_levels=2, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.geo_volume_pyramid = [geo_volume]
        vol = geo_volume
        for _ in range(num_levels - 1):
            vol = F.avg_pool3d(vol, kernel_size=(2, 1, 1), stride=(2, 1, 1))
            self.geo_volume_pyramid.append(vol)
        self.dx = torch.linspace(-radius, radius, 2 * radius + 1).to(geo_volume.device)

    def __call__(self, disp):
        b, _, h, w = disp.shape
        r = self.radius
        out_pyramid = []

        for i in range(self.num_levels):
            vol = self.geo_volume_pyramid[i]
            D_i = vol.shape[2]
            C = vol.shape[1]

            sample_pos = disp.squeeze(1).unsqueeze(1).float() / (2 ** i) + self.dx.view(1, -1, 1, 1)

            pos_floor = torch.floor(sample_pos).long()
            pos_ceil = pos_floor + 1
            alpha = sample_pos - pos_floor.float()

            fl_valid = (pos_floor >= 0) & (pos_floor < D_i)
            ce_valid = (pos_ceil >= 0) & (pos_ceil < D_i)

            idx_fl = pos_floor.clamp(0, D_i - 1).unsqueeze(1).expand(-1, C, -1, -1, -1)
            idx_ce = pos_ceil.clamp(0, D_i - 1).unsqueeze(1).expand(-1, C, -1, -1, -1)

            val_fl = torch.gather(vol, 2, idx_fl) * fl_valid.unsqueeze(1).float()
            val_ce = torch.gather(vol, 2, idx_ce) * ce_valid.unsqueeze(1).float()

            sampled = val_fl * (1 - alpha.unsqueeze(1)) + val_ce * alpha.unsqueeze(1)
            sampled = sampled.reshape(b, C * (2 * r + 1), h, w)
            out_pyramid.append(sampled)

        return torch.cat(out_pyramid, dim=1).contiguous().float()


# ============================================
# WAVEStereo ONNX 包装类
# ============================================
class WavestereoOnnx(nn.Module):
    """
    WAVEStereo 单 ONNX 包装类。
    输入: 预归一化的左右图像
    输出: 视差图
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def forward(self, left_image, right_image):
        data = {'left': left_image, 'right': right_image}
        result = self.model(data)
        return result['disp_pred']


def make_res_tag(height, width, valid_iters):
    """生成导出 tag，避免不同配置互相覆盖。"""
    return f'{height}x{width}_iter{valid_iters}'


# ============================================
# Monkey-patch 函数
# ============================================
_original_refs = {}


def apply_onnx_patches():
    """应用 ONNX 兼容性补丁，替换不可导出的算子"""
    global _original_refs

    # Patch correlation_volume
    import wavestereo.model.wavestereo as wavestereo_module
    _original_refs['correlation_volume'] = wavestereo_module.correlation_volume
    wavestereo_module.correlation_volume = correlation_volume_onnx

    # Patch disp_warp
    import wavestereo.model.update as update_module
    _original_refs['disp_warp'] = update_module.disp_warp
    update_module.disp_warp = disp_warp_onnx

    # Patch DPTFusionBlock.forward (移除 autocast)
    from wavestereo.model.update import DPTFusionBlock
    _original_refs['DPTFusionBlock.forward'] = DPTFusionBlock.forward
    DPTFusionBlock.forward = dpt_fusion_block_forward_onnx

    # Patch WAVEStereo.forward (softmax/softargmax FP32, Geo_Encoding_Volume ONNX 兼容)
    from wavestereo.model.wavestereo import WAVEStereo
    _original_refs['WAVEStereo.forward'] = WAVEStereo.forward
    WAVEStereo.forward = wavestereo_forward_onnx

def remove_onnx_patches():
    """恢复原始实现"""
    import wavestereo.model.wavestereo as wavestereo_module
    from wavestereo.model.update import DPTFusionBlock
    import wavestereo.model.update as update_module
    from wavestereo.model.wavestereo import WAVEStereo

    if 'correlation_volume' in _original_refs:
        wavestereo_module.correlation_volume = _original_refs['correlation_volume']
    if 'disp_warp' in _original_refs:
        update_module.disp_warp = _original_refs['disp_warp']
    if 'DPTFusionBlock.forward' in _original_refs:
        DPTFusionBlock.forward = _original_refs['DPTFusionBlock.forward']
    if 'WAVEStereo.forward' in _original_refs:
        WAVEStereo.forward = _original_refs['WAVEStereo.forward']
    _original_refs.clear()


# ============================================
# 主函数
# ============================================
def parse_args():
    parser = argparse.ArgumentParser(description='导出 WAVEStereo 为单个 ONNX 模型')
    parser.add_argument('--weights', type=str, default=None,
                        help='模型权重路径（可选，默认使用配置文件中的 PRETRAINED_MODEL）')
    parser.add_argument('--config', type=str, required=True,
                        default='cfgs/wavestereo/wavestereo_sceneflow_eval.yaml',
                        help='配置文件路径')
    parser.add_argument('--save_path', type=str, default='output_wavestereo/',
                        help='ONNX 保存路径')
    parser.add_argument('--height', type=int, default=480,
                        help='输入图像高度（必须能被 32 整除）')
    parser.add_argument('--width', type=int, default=640,
                        help='输入图像宽度（必须能被 32 整除）')
    parser.add_argument('--valid_iters', type=int, default=8,
                        help='GRU 迭代次数')
    parser.add_argument('--max_disp', type=int, default=192,
                        help='最大视差')
    parser.add_argument('--opset', type=int, default=17,
                        help='ONNX opset 版本')
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('make_onnx_wavestereo')

    os.makedirs(args.save_path, exist_ok=True)
    torch.autograd.set_grad_enabled(False)

    # Build model without the OpenStereo trainer framework.
    logger.info(f"Loading config: {args.config}")
    cfgs = load_config(args.config)

    cfgs.MODEL.MAX_DISP = args.max_disp
    cfgs.MODEL.VALID_ITERS = args.valid_iters

    if args.weights:
        cfgs.MODEL.PRETRAINED_MODEL = args.weights

    logger.info(f"Building model and loading weights: {cfgs.MODEL.PRETRAINED_MODEL}")
    model = WAVEStereo(cfgs.MODEL).cuda().eval()
    if cfgs.MODEL.PRETRAINED_MODEL:
        load_checkpoint(model, cfgs.MODEL.PRETRAINED_MODEL, strict=False, map_location='cpu')
    else:
        logger.warning("No weights were provided; exporting a randomly initialized model")

    # 应用 ONNX 兼容性补丁
    logger.info("应用 ONNX 兼容性补丁...")
    apply_onnx_patches()

    # 创建包装类
    wrapper = WavestereoOnnx(model)
    wrapper.cuda().eval()

    # 检查输入尺寸
    assert args.height % 32 == 0 and args.width % 32 == 0, \
        f"height 和 width 必须能被 32 整除, 当前 {args.height}x{args.width}"

    # 创建虚拟输入（预归一化）
    logger.info(f"创建虚拟输入: {args.height}x{args.width}")
    left_img = torch.randn(1, 3, args.height, args.width).cuda().float()
    right_img = torch.randn(1, 3, args.height, args.width).cuda().float()

    # 测试前向传播
    logger.info("测试前向传播...")
    with torch.no_grad():
        disp = wrapper(left_img, right_img)
    logger.info(f"  输出视差图形状: {disp.shape}")

    # 导出 ONNX（文件名带分辨率和迭代次数，避免不同配置互相覆盖）
    res_tag = make_res_tag(args.height, args.width, args.valid_iters)
    onnx_filename = f'wavestereo_{res_tag}.onnx'
    engine_filename = f'wavestereo_{res_tag}.engine'
    onnx_path = os.path.join(args.save_path, onnx_filename)
    logger.info(f"导出 ONNX → {onnx_path}")
    torch.onnx.export(
        wrapper,
        (left_img, right_img),
        onnx_path,
        opset_version=args.opset,
        input_names=['left_image', 'right_image'],
        output_names=['disparity'],
        do_constant_folding=True,
    )

    # 验证 ONNX 模型
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX 模型验证通过")
    except ImportError:
        logger.warning("onnx 未安装，跳过模型验证")
    except Exception as e:
        logger.error(f"ONNX 模型验证失败: {e}")

    # 获取归一化参数
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if hasattr(cfgs, 'DATA_CONFIG') and hasattr(cfgs.DATA_CONFIG, 'DATA_TRANSFORM'):
        transforms = cfgs.DATA_CONFIG.DATA_TRANSFORM.get('EVALUATING', [])
        for t in transforms:
            if isinstance(t, dict) and 'NormalizeImage' in str(t):
                mean = t.get('MEAN', mean)
                std = t.get('STD', std)

    # 保存配置
    cfg_dict = {
        'max_disp': args.max_disp,
        'valid_iters': args.valid_iters,
        'image_size': [args.height, args.width],
        'corr_radius': cfgs.MODEL.CORR_RADIUS,
        'corr_levels': cfgs.MODEL.CORR_LEVELS,
        'hidden_dim': cfgs.MODEL.HIDDEN_DIM,
        'normalize_mean': mean,
        'normalize_std': std,
        'model_type': 'WAVEStereo',
        'opset_version': args.opset,
        'onnx_filename': onnx_filename,
        'engine_filename': engine_filename,
    }
    cfg_path = os.path.join(args.save_path, 'onnx.yaml')
    save_config(cfg_dict, cfg_path)

    logger.info("=" * 50)
    logger.info("导出完成!")
    logger.info(f"  保存路径: {args.save_path}")
    logger.info(f"  - wavestereo_{res_tag}.onnx")
    logger.info(f"  - onnx.yaml")
    logger.info("=" * 50)
    engine_path = os.path.join(args.save_path, engine_filename)
    logger.info("下一步:")
    logger.info(f"  trtexec --onnx={onnx_path} "
                f"--saveEngine={engine_path} --fp16")


if __name__ == '__main__':
    main()
