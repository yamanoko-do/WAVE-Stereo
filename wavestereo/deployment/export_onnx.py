import os
import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from wavestereo.config import load_config, save_config
from wavestereo.artifacts import (
    BUNDLE_SCHEMA_VERSION,
    EXPORT_PROFILES,
    METADATA_FILENAME,
    ONNX_FILENAME,
    load_bundle_metadata,
    make_bundle_dir,
    sha256_file,
)
from wavestereo.inference.checkpoint import load_checkpoint
from wavestereo.inference.preprocess import normalization_from_config
from wavestereo.model import WAVEStereo


REPO_ROOT = Path(__file__).resolve().parents[2]


# ============================================
# ONNX 兼容的 correlation_volume（分块向量化 + 可选 NPU 归约）
# ============================================
_correlation_chunk_size = 0
_correlation_builder = 'padstack'
_correlation_reducer = 'mean'
_geometry_sampler = 'gather'
_batch_stereo_backbone = False
_hoist_warp_left = False
_late_image_stems = False
_geometry_pyramid_builder = 'pool'
_context_splitter = 'split'


class PolyphaseConvTranspose2d(nn.Module):
    """Exact stride-2 ConvTranspose2d decomposition for export experiments.

    The transposed convolution is written as a low-resolution Conv2d whose
    four output-channel phases are rearranged by PixelShuffle.  This keeps the
    learned weights unchanged and lets OpenVINO try its regular-convolution
    kernels.  Only the two layouts used by WAVEStereo are supported:

      * kernel=3, padding=1, output_padding=1
      * kernel=4, padding=1, output_padding=0
    """

    def __init__(self, deconv):
        super().__init__()
        if not isinstance(deconv, nn.ConvTranspose2d):
            raise TypeError("deconv must be nn.ConvTranspose2d")
        if deconv.groups != 1 or deconv.dilation != (1, 1) or deconv.stride != (2, 2):
            raise ValueError("polyphase rewrite requires groups=1, dilation=1, stride=2")

        kernel = deconv.kernel_size
        padding = deconv.padding
        output_padding = deconv.output_padding
        if (kernel, padding, output_padding) == ((3, 3), (1, 1), (1, 1)):
            self.mode = "k3"
            conv_kernel = 2
            offsets = (0, 1)
            offset_base = 0
        elif (kernel, padding, output_padding) == ((4, 4), (1, 1), (0, 0)):
            self.mode = "k4"
            conv_kernel = 3
            offsets = (-1, 0, 1)
            offset_base = 1
        else:
            raise ValueError(
                "unsupported ConvTranspose2d layout: "
                f"kernel={kernel}, padding={padding}, output_padding={output_padding}"
            )

        # ConvTranspose2d stores [Cin, Cout, Kh, Kw].  Conv2d needs
        # [Cout*4, Cin, Kh', Kw']; PixelShuffle channel order is
        # out_channel * 4 + row_phase * 2 + col_phase.
        src = deconv.weight.detach()
        cin, cout, _, _ = src.shape
        weight = src.new_zeros((cout * 4, cin, conv_kernel, conv_kernel))
        for out_ch in range(cout):
            for row_phase in range(2):
                for col_phase in range(2):
                    poly_ch = out_ch * 4 + row_phase * 2 + col_phase
                    for di in offsets:
                        for dj in offsets:
                            src_i = row_phase + 1 - 2 * di
                            src_j = col_phase + 1 - 2 * dj
                            if 0 <= src_i < kernel[0] and 0 <= src_j < kernel[1]:
                                weight[poly_ch, :, di + offset_base, dj + offset_base] = \
                                    src[:, out_ch, src_i, src_j]
        self.weight = nn.Parameter(weight, requires_grad=False)

        if deconv.bias is None:
            self.bias = None
        else:
            bias = deconv.bias.detach().repeat_interleave(4)
            self.bias = nn.Parameter(bias, requires_grad=False)

    def forward(self, x):
        if self.mode == "k3":
            # The lower/right phases read one sample beyond the input edge.
            x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0.0)
            x = F.conv2d(x, self.weight, self.bias)
        else:
            x = F.conv2d(x, self.weight, self.bias, padding=1)
        return F.pixel_shuffle(x, 2)


def replace_deconvs_with_polyphase(model, scope):
    """Replace selected inference-only deconvolutions in-place."""
    if scope == "none":
        return []

    replacements = []

    def replace(parent, attr, name):
        old = getattr(parent, attr)
        setattr(parent, attr, PolyphaseConvTranspose2d(old))
        replacements.append(name)

    if scope in ("aggregation", "all"):
        replace(model.cost_agg.conv5, "0", "cost_agg.conv5.0")
        replace(model.cost_agg.conv6, "0", "cost_agg.conv6.0")

    if scope in ("fpn", "all"):
        for level in (4, 3, 2):
            block = getattr(model.backbone, f"fpn_layer{level}").deconv.block
            replace(block, "0", f"backbone.fpn_layer{level}.deconv.block.0")

    if scope in ("spx", "all"):
        replace(model.spx_2_gru.conv1, "conv", "spx_2_gru.conv1.conv")
        replace(model.spx_1_gru.conv1, "conv", "spx_1_gru.conv1.conv")

    return replacements


def rewrite_stem_instance_norm_as_group_norm(onnx_path, model, scope="first"):
    """Use native ONNX GroupNormalization for selected image-stem norms.

    GroupNormalization(groups=C) is mathematically identical to
    InstanceNorm2d when running statistics are disabled.  Reusing the ONNX
    InstanceNormalization scale/bias inputs also preserves affine parameters.
    """
    targets = [
        ("stem_2/stem_2.0/IN/InstanceNormalization", model.stem_2[0].IN),
    ]
    if scope in ("stem2", "all"):
        targets.append(("stem_2/stem_2.2/InstanceNormalization", model.stem_2[2]))
    if scope == "all":
        targets.append(("stem_1/IN/InstanceNormalization", model.stem_1.IN))
    if scope not in ("first", "stem2", "all"):
        raise ValueError(f"unknown stem GroupNorm scope: {scope}")

    for _, norm in targets:
        if not isinstance(norm, nn.InstanceNorm2d):
            raise TypeError("selected image-stem norm is not InstanceNorm2d")
        if norm.track_running_stats:
            raise ValueError("GroupNormalization rewrite is invalid with running statistics")

    import onnx
    from onnx import helper, version_converter

    onnx_model = onnx.load(onnx_path)
    # GroupNormalization entered the default ONNX domain in opset 21.  Use
    # ONNX's converter so pre-21 attribute encodings (for example ReduceMean
    # axes) are upgraded together instead of only changing the opset number.
    default_opset = next(
        (opset.version for opset in onnx_model.opset_import
         if opset.domain in ("", "ai.onnx")),
        0,
    )
    if default_opset < 21:
        onnx_model = version_converter.convert_version(onnx_model, 21)
    for name_fragment, norm in targets:
        candidates = [
            node for node in onnx_model.graph.node
            if node.op_type == "InstanceNormalization" and name_fragment in node.name
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"expected one {name_fragment!r} InstanceNormalization, found {len(candidates)}"
            )

        node = candidates[0]
        epsilon = norm.eps
        for attr in node.attribute:
            if attr.name == "epsilon":
                epsilon = helper.get_attribute_value(attr)
                break
        del node.attribute[:]
        node.op_type = "GroupNormalization"
        node.attribute.extend([
            helper.make_attribute("epsilon", float(epsilon)),
            helper.make_attribute("num_groups", int(norm.num_features)),
        ])

    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, onnx_path)


def rewrite_dpt_instance_norm_as_group_norm(onnx_path, model):
    """Rewrite exported DPT residual InstanceNorm nodes as native GroupNorm."""
    import onnx
    from onnx import helper, version_converter

    dpt_norms = [
        module for module in model.update_block.transformer.dpt.modules()
        if isinstance(module, nn.InstanceNorm2d)
    ]
    num_features = {norm.num_features for norm in dpt_norms}
    if not dpt_norms or len(num_features) != 1:
        raise RuntimeError(
            "DPT GroupNorm rewrite requires one shared InstanceNorm channel count"
        )
    num_groups = num_features.pop()

    onnx_model = onnx.load(onnx_path)
    default_opset = next(
        (opset.version for opset in onnx_model.opset_import
         if opset.domain in ("", "ai.onnx")),
        0,
    )
    if default_opset < 21:
        onnx_model = version_converter.convert_version(onnx_model, 21)

    candidates = [
        node for node in onnx_model.graph.node
        if node.op_type == "InstanceNormalization"
        and "/transformer/dpt/" in node.name
        and "/down_proj/" not in node.name
    ]
    if not candidates:
        raise RuntimeError("found no exported DPT InstanceNormalization nodes")

    for node in candidates:
        epsilon = 1e-5
        for attr in node.attribute:
            if attr.name == "epsilon":
                epsilon = helper.get_attribute_value(attr)
                break
        del node.attribute[:]
        node.op_type = "GroupNormalization"
        node.attribute.extend([
            helper.make_attribute("epsilon", float(epsilon)),
            helper.make_attribute("num_groups", int(num_groups)),
        ])

    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, onnx_path)
    return len(candidates), num_groups


def correlation_volume_onnx(left_feature, right_feature, max_disp):
    """
    ONNX 兼容的 correlation volume 构建。
    分块向量化实现：每块 stack 一组移位的 right，批量乘并归约通道后再处理下一块。
    使用 FP32 累积提高数值精度。
    Mathematically equivalent to wavestereo.layers.cost_volume.correlation_volume.
    """
    left_f = left_feature.float()
    right_f = right_feature.float()
    B, C, H, W = left_f.shape
    chunk_size = _correlation_chunk_size or max_disp
    chunk_size = min(chunk_size, max_disp)

    def reduce_channels(right_volume):
        product = left_f.unsqueeze(2) * right_volume
        if _correlation_reducer == 'mean':
            return product.mean(dim=1)

        # This fixed pointwise convolution performs the same channel dot
        # product while mapping the reduction to the NPU's DPU path.
        # Export bundles have a fixed input specification, so freezing these
        # dimensions also gives ONNX enough channel information to lower Conv.
        batch, channels, disparities, height, width = (
            int(dim) for dim in product.shape
        )
        product = product.permute(0, 2, 1, 3, 4).reshape(
            batch * disparities, channels, height, width
        )
        weight = torch.full(
            (1, channels, 1, 1), 1.0 / channels,
            dtype=product.dtype, device=product.device,
        )
        reduced = F.conv2d(product, weight)
        return reduced.reshape(batch, disparities, height, width)

    volume_chunks = []
    if _correlation_builder == 'matmul':
        # For each image row, form the complete W(left) x W(right) matrix:
        #   [W, C] @ [C, W] -> [W, W]
        # Entry (x, y) is the channel dot product between left[x] and
        # right[y].  The stereo cost volume is the y = x - disparity band.
        # Padding the right-coordinate axis supplies exact zeros for x < d.
        left_rows = left_f.permute(0, 2, 3, 1)
        right_rows = right_f.permute(0, 2, 1, 3)
        all_pairs = torch.matmul(left_rows, right_rows)
        all_pairs = F.pad(
            all_pairs, (max_disp - 1, 0), mode='constant', value=0.0
        )
        x_index = torch.arange(W, device=right_f.device, dtype=torch.long)
        for start in range(0, max_disp, chunk_size):
            end = min(start + chunk_size, max_disp)
            disparities = torch.arange(
                start, end, device=right_f.device, dtype=torch.long
            )
            gather_index = (
                x_index.unsqueeze(1) + (max_disp - 1) - disparities.unsqueeze(0)
            )
            gather_index = gather_index.view(1, 1, W, end - start).expand(
                B, H, W, end - start
            )
            band = torch.gather(all_pairs, 3, gather_index)
            volume_chunks.append(band.permute(0, 3, 1, 2) / float(C))
    elif _correlation_builder == 'gather':
        # Pad right only once, then materialize each disparity tile with one
        # indexed read.  This removes the repeated Pad/Slice/Concat chain from
        # the graph while retaining the same BxCxDxHxW reduction order.
        right_padded = F.pad(right_f, (max_disp - 1, 0, 0, 0), 'constant', 0.0)
        x_index = torch.arange(W, device=right_f.device, dtype=torch.long)
        for start in range(0, max_disp, chunk_size):
            end = min(start + chunk_size, max_disp)
            disparities = torch.arange(start, end, device=right_f.device, dtype=torch.long)
            gather_index = (
                x_index.unsqueeze(1) + (max_disp - 1) - disparities.unsqueeze(0)
            )
            right_volume = torch.index_select(
                right_padded, 3, gather_index.reshape(-1)
            ).reshape(B, C, H, W, end - start).permute(0, 1, 4, 2, 3).contiguous()
            volume_chunks.append(reduce_channels(right_volume))
    else:
        for start in range(0, max_disp, chunk_size):
            end = min(start + chunk_size, max_disp)
            shifted = [
                F.pad(right_f, (d, 0, 0, 0), 'constant', 0.0)[:, :, :, :W]
                for d in range(start, end)
            ]
            # Restrict the large BxCxDxHxW temporary to one disparity tile.
            right_volume = torch.stack(shifted, dim=2)
            volume_chunks.append(reduce_channels(right_volume))
    volume = volume_chunks[0] if len(volume_chunks) == 1 else torch.cat(volume_chunks, dim=1)
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
    project_before_upsample = getattr(self, 'project_before_upsample', False)
    if project_before_upsample:
        out = self.out_conv(out)
    if target_size is None:
        target_size = (main_feat.shape[2] * 2, main_feat.shape[3] * 2)
    out = F.interpolate(out.float(), size=target_size, mode='bilinear', align_corners=False).to(main_feat.dtype)
    if project_before_upsample:
        # Resize's legacy ONNX symbolic drops the known channel dimension.
        # Restore it explicitly so a following InstanceNorm can be exported.
        out = out.reshape(out.shape[0], self.out_conv.out_channels, out.shape[2], out.shape[3])
    else:
        out = self.out_conv(out)
    return out


# ============================================
# Inference-only warp-fusion hoist
# ============================================
def warp_fusion_left_cache_onnx(encoder, feat_left):
    """Precompute the invariant left half of warp_fusion outside the GRU loop."""
    fusion = encoder.warp_fusion
    if fusion.groups != 1 or fusion.padding_mode != 'zeros':
        raise ValueError("warp-fusion hoist requires a dense zero-padded Conv2d")
    if fusion.in_channels % 2:
        raise ValueError("warp-fusion hoist requires an even input-channel count")
    split = fusion.in_channels // 2
    return F.conv2d(
        feat_left, fusion.weight[:, :split], None,
        fusion.stride, fusion.padding, fusion.dilation, fusion.groups,
    )


def update_block_forward_warp_hoisted(block, net, inp, feat_left, feat_right,
                                      disp, corr, itr, warp_left_cache):
    """Equivalent TransformerUpdateBlock forward with split warp_fusion weights.

    Conv(cat(left, warped), [W_left, W_right], bias) is decomposed into
    cached Conv(left, W_left, no_bias) plus per-iteration
    Conv(warped, W_right, bias).  ReLU and every subsequent operation retain
    their original order.  This helper is used only by the ONNX export path.
    """
    encoder = block.encoder

    cor = F.relu(encoder.convc1(corr))
    cor = F.relu(encoder.convc2(cor))

    disp_features = F.relu(encoder.convd1(disp))
    disp_features = F.relu(encoder.convd2(disp_features))

    warped = disp_warp_onnx(feat_right, disp)
    fusion = encoder.warp_fusion
    split = fusion.in_channels // 2
    warp_right = F.conv2d(
        warped, fusion.weight[:, split:], fusion.bias,
        fusion.stride, fusion.padding, fusion.dilation, fusion.groups,
    )
    warp = F.relu(warp_left_cache + warp_right)
    warp = F.relu(encoder.warp_conv(warp))

    motion = torch.cat([cor, disp_features, warp], dim=1)
    motion = F.relu(encoder.conv(motion))
    motion = F.relu(encoder.conv2(motion))
    motion = torch.cat([motion, disp], dim=1)

    net = block.gru(net, inp, motion)
    if block.refine_every > 0 and itr % block.refine_every == 0:
        global_feat = block.transformer(motion)
        net = net + block.tf_gate * global_feat

    delta_disp = block.disp_head(net)
    mask_feat_4 = block.mask_feat_4(net)
    return net, delta_disp, mask_feat_4


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

    late_image_stems = _late_image_stems and not self.training
    if late_image_stems:
        stem_2x_img = None
        stem_1x_img = None
    else:
        stem_2x_img = self.stem_2(image1)
        stem_1x_img = self.stem_1(image1)

    if _batch_stereo_backbone:
        # The two images use identical backbone weights.  Running them as one
        # batch preserves per-image normalization semantics while allowing the
        # GPU to launch each backbone kernel once for batch=2 instead of twice
        # for batch=1.  The feature tensors are views after the split.
        stereo_features = self.backbone(torch.cat([image1, image2], dim=0))
        batch_size = image1.shape[0]
        features_left = [feature[:batch_size] for feature in stereo_features]
        features_right = [feature[batch_size:] for feature in stereo_features]
    else:
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
    context_features = self.context_zqr_conv(features_left[0])
    if _context_splitter == 'slice':
        context = [
            context_features[:, index * self.hidden_dim:(index + 1) * self.hidden_dim]
            for index in range(3)
        ]
    else:
        context = list(context_features.split(split_size=self.hidden_dim, dim=1))

    geo_classes = {
        'gather': Geo_Encoding_Volume_onnx,
        'padded': Geo_Encoding_Volume_padded_onnx,
        'padded_fused': Geo_Encoding_Volume_padded_fused_onnx,
        'grid': Geo_Encoding_Volume_grid_onnx,
    }
    geo_cls = geo_classes[_geometry_sampler]
    geo_fn = geo_cls(unsqueezed_encoding, radius=self.corr_radius, num_levels=self.corr_levels)

    warp_left_cache = None
    if _hoist_warp_left:
        warp_left_cache = warp_fusion_left_cache_onnx(
            self.update_block.encoder, features_left[0]
        )

    iters = self.args.VALID_ITERS if not self.training else self.args.TRAIN_ITERS
    disp = init_disp
    disp_preds = []

    for itr in range(iters):
        disp = disp.detach()
        corr = geo_fn(disp)
        if _hoist_warp_left:
            net, delta_disp, mask_feat = update_block_forward_warp_hoisted(
                self.update_block, net, context,
                feat_left=features_left[0], feat_right=features_right[0],
                disp=disp, corr=corr, itr=itr,
                warp_left_cache=warp_left_cache,
            )
        else:
            net, delta_disp, mask_feat = self.update_block(
                net, context, feat_left=features_left[0],
                feat_right=features_right[0], disp=disp, corr=corr, itr=itr,
            )
        disp = disp + delta_disp
        if not self.training and itr < iters - 1:
            continue
        if late_image_stems:
            # These two high-resolution features are consumed only by the
            # final disparity upsampler during inference.  Tracing them here
            # shortens their live range across cost aggregation and the GRU
            # loop, reducing allocator pressure without changing arithmetic.
            stem_2x_img = self.stem_2(image1)
            stem_1x_img = self.stem_1(image1)
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
            if _geometry_pyramid_builder == 'slice':
                if int(vol.shape[2]) % 2:
                    raise ValueError("slice geometry pyramid requires even depth")
                vol = (vol[:, :, 0::2] + vol[:, :, 1::2]) * 0.5
            else:
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


class Geo_Encoding_Volume_grid_onnx(Geo_Encoding_Volume_onnx):
    """Equivalent geometry lookup using ONNX GridSample instead of gather trees."""

    def __call__(self, disp):
        b, _, h, w = disp.shape
        r = self.radius
        sample_count = 2 * r + 1
        out_pyramid = []

        for i, vol in enumerate(self.geo_volume_pyramid):
            depth = vol.shape[2]
            channels = vol.shape[1]
            sample_pos = (
                disp.squeeze(1).unsqueeze(1).float() / (2 ** i)
                + self.dx.view(1, -1, 1, 1)
            )

            # Treat every image location as an independent 1-D signal along D.
            lines = vol.permute(0, 3, 4, 1, 2).reshape(b * h * w, channels, 1, depth)
            x_grid = sample_pos.permute(0, 2, 3, 1).reshape(b * h * w, 1, sample_count)
            x_grid = 2.0 * x_grid / max(depth - 1, 1) - 1.0
            grid = torch.stack([x_grid, torch.zeros_like(x_grid)], dim=-1)
            sampled = F.grid_sample(
                lines, grid, mode='bilinear', padding_mode='zeros', align_corners=True
            )
            sampled = sampled.reshape(b, h, w, channels, sample_count)
            sampled = sampled.permute(0, 3, 4, 1, 2).reshape(
                b, channels * sample_count, h, w
            )
            out_pyramid.append(sampled)

        return torch.cat(out_pyramid, dim=1).contiguous().float()


class Geo_Encoding_Volume_padded_onnx(Geo_Encoding_Volume_onnx):
    """Exact gather lookup with zero sentinels instead of per-iteration masks."""

    def __init__(self, geo_volume, num_levels=2, radius=4):
        super().__init__(geo_volume, num_levels=num_levels, radius=radius)
        # One zero slice at each end reproduces GridSample's zero padding for
        # coordinates in and beyond the [-1, D] interpolation boundary.
        self.padded_pyramid = [
            F.pad(vol, (0, 0, 0, 0, 1, 1), mode='constant', value=0.0)
            for vol in self.geo_volume_pyramid
        ]

    def __call__(self, disp):
        b, _, h, w = disp.shape
        r = self.radius
        out_pyramid = []

        for i, vol in enumerate(self.padded_pyramid):
            depth = vol.shape[2] - 2
            channels = vol.shape[1]
            sample_pos = (
                disp.squeeze(1).unsqueeze(1).float() / (2 ** i)
                + self.dx.view(1, -1, 1, 1)
            )
            pos_floor = torch.floor(sample_pos)
            alpha = sample_pos - pos_floor
            floor_index = (pos_floor.long() + 1).clamp(0, depth + 1)
            ceil_index = (pos_floor.long() + 2).clamp(0, depth + 1)
            floor_index = floor_index.unsqueeze(1).expand(-1, channels, -1, -1, -1)
            ceil_index = ceil_index.unsqueeze(1).expand(-1, channels, -1, -1, -1)
            floor_value = torch.gather(vol, 2, floor_index)
            ceil_value = torch.gather(vol, 2, ceil_index)
            sampled = floor_value * (1 - alpha.unsqueeze(1)) + ceil_value * alpha.unsqueeze(1)
            out_pyramid.append(sampled.reshape(b, channels * (2 * r + 1), h, w))

        return torch.cat(out_pyramid, dim=1).contiguous().float()


class Geo_Encoding_Volume_padded_fused_onnx(Geo_Encoding_Volume_padded_onnx):
    """Exact padded lookup with one gather per pyramid level.

    The sampling offsets are integers, so one base floor is sufficient for
    every point in a level.  Concatenate the floor/ceil indices and fetch both
    sides with one gather before splitting them again.  Alpha is reconstructed
    from the rounded per-offset sample position to preserve PyTorch bitwise
    equivalence (sharing the base alpha changes a few FP32 low bits).
    """

    def __init__(self, geo_volume, num_levels=2, radius=4):
        super().__init__(geo_volume, num_levels=num_levels, radius=radius)
        self.integer_dx = torch.arange(
            -radius, radius + 1, dtype=torch.long, device=geo_volume.device
        )
        self.float_dx = self.integer_dx.float()

    def __call__(self, disp):
        b, _, h, w = disp.shape
        sample_count = 2 * self.radius + 1
        out_pyramid = []

        for i, vol in enumerate(self.padded_pyramid):
            depth = vol.shape[2] - 2
            channels = vol.shape[1]
            base_pos = disp.squeeze(1).float() / (2 ** i)
            base_floor = torch.floor(base_pos).long()
            integer_pos = (
                base_floor.unsqueeze(1)
                + self.integer_dx.view(1, -1, 1, 1)
            )
            sample_pos = (
                base_pos.unsqueeze(1)
                + self.float_dx.view(1, -1, 1, 1)
            )
            alpha = sample_pos - integer_pos.float()

            # +1 maps the original [0, depth-1] range into the padded volume.
            floor_index = integer_pos + 1
            gather_index = torch.cat((floor_index, floor_index + 1), dim=1)
            gather_index = gather_index.clamp(0, depth + 1)
            gather_index = gather_index.unsqueeze(1).expand(
                -1, channels, -1, -1, -1
            )

            values = torch.gather(vol, 2, gather_index)
            floor_value, ceil_value = values.split(sample_count, dim=2)
            alpha = alpha.unsqueeze(1)
            sampled = floor_value * (1 - alpha) + ceil_value * alpha
            out_pyramid.append(sampled.reshape(b, channels * sample_count, h, w))

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
    def __init__(self, model, input_height=None, input_width=None):
        super().__init__()
        self.model = model
        self.input_height = input_height
        self.input_width = input_width
        self.pad_h = (32 - input_height % 32) % 32 if input_height else 0
        self.pad_w = (32 - input_width % 32) % 32 if input_width else 0

    @torch.no_grad()
    def forward(self, left_image, right_image):
        if self.pad_h or self.pad_w:
            # Preserve the requested external ONNX shape while satisfying the
            # backbone's internal /32 feature-pyramid alignment requirement.
            # Pad the top and right using replicated edge values.
            padding = (0, self.pad_w, self.pad_h, 0)
            left_image = F.pad(left_image, padding, mode='replicate')
            right_image = F.pad(right_image, padding, mode='replicate')
        data = {'left': left_image, 'right': right_image}
        result = self.model(data)
        disparity = result['disp_pred']
        if self.pad_h or self.pad_w:
            disparity = disparity[
                :, :, self.pad_h:self.pad_h + self.input_height, :self.input_width
            ]
        return disparity


class BgrUint8OnnxInput(nn.Module):
    """Bake the schema-v2 BGR uint8/NHWC input contract into the graph."""

    def __init__(self, model, implementation, mean, std):
        super().__init__()
        if implementation not in ("reference", "lookup", "affine"):
            raise ValueError(f"Unknown ONNX preprocessing implementation: {implementation}")
        self.model = model
        self.implementation = implementation

        mean_tensor = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        std_tensor = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean_tensor)
        self.register_buffer("std", std_tensor)
        self.register_buffer("scale", 1.0 / (255.0 * std_tensor))
        self.register_buffer("bias", -mean_tensor / std_tensor)

        values = torch.arange(256, dtype=torch.float32).view(1, 256)
        lookup = (values / 255.0 - mean_tensor.view(3, 1)) / std_tensor.view(3, 1)
        self.register_buffer("lookup", lookup)

    def _preprocess(self, image):
        if image.dtype != torch.uint8:
            raise TypeError("schema-v2 ONNX input must be uint8")

        # Input is NHWC/BGR. Stacking in RGB order performs both the color and
        # layout conversion without an intermediate full-size transpose.
        if self.implementation == "lookup":
            return torch.stack(
                (
                    self.lookup[0][image[..., 2].long()],
                    self.lookup[1][image[..., 1].long()],
                    self.lookup[2][image[..., 0].long()],
                ),
                dim=1,
            )

        rgb = torch.stack(
            (image[..., 2], image[..., 1], image[..., 0]), dim=1
        ).float()
        if self.implementation == "affine":
            return rgb * self.scale + self.bias
        return (rgb / 255.0 - self.mean) / self.std

    @torch.no_grad()
    def forward(self, left_image, right_image):
        return self.model(
            self._preprocess(left_image), self._preprocess(right_image)
        )


# ============================================
# Monkey-patch 函数
# ============================================
_original_refs = {}


def apply_onnx_patches(correlation_chunk_size=0, geometry_sampler='gather',
                       batch_stereo_backbone=False, correlation_builder='padstack',
                       hoist_warp_left=False, late_image_stems=False,
                       correlation_reducer='mean', geometry_pyramid_builder='pool',
                       context_splitter='split'):
    """应用 ONNX 兼容性补丁，替换不可导出的算子"""
    global _original_refs, _correlation_chunk_size, _correlation_builder
    global _correlation_reducer
    global _geometry_sampler, _batch_stereo_backbone, _hoist_warp_left
    global _late_image_stems, _geometry_pyramid_builder, _context_splitter
    _correlation_chunk_size = correlation_chunk_size
    _correlation_builder = correlation_builder
    _correlation_reducer = correlation_reducer
    _geometry_sampler = geometry_sampler
    _batch_stereo_backbone = batch_stereo_backbone
    _hoist_warp_left = hoist_warp_left
    _late_image_stems = late_image_stems
    _geometry_pyramid_builder = geometry_pyramid_builder
    _context_splitter = context_splitter

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


_STANDARD_GRAPH_OPTIONS = {
    'correlation_chunk_size': 0,
    'correlation_builder': 'padstack',
    'correlation_reducer': 'mean',
    'geometry_sampler': 'gather',
    'geometry_pyramid_builder': 'pool',
    'context_splitter': 'split',
    'dpt_conv_before_upsample': False,
    'batch_stereo_backbone': False,
    'hoist_warp_left': False,
    'late_image_stems': False,
    'polyphase_deconv': 'none',
    'stem_groupnorm_scope': 'none',
    'dpt_groupnorm': False,
}

_INTEL_NPU_GRAPH_OPTIONS = _STANDARD_GRAPH_OPTIONS | {
    'correlation_reducer': 'conv',
    'geometry_sampler': 'grid',
    'geometry_pyramid_builder': 'slice',
    'context_splitter': 'slice',
}

_INTEL_GPU_GRAPH_OPTIONS = _STANDARD_GRAPH_OPTIONS | {
    'correlation_builder': 'matmul',
    'geometry_sampler': 'padded',
    'batch_stereo_backbone': True,
    'polyphase_deconv': 'all',
    'stem_groupnorm_scope': 'first',
    'dpt_groupnorm': True,
}

_PROFILE_GRAPH_OPTIONS = {
    'standard': _STANDARD_GRAPH_OPTIONS,
    'intel_gpu': _INTEL_GPU_GRAPH_OPTIONS,
    'intel_npu': _INTEL_NPU_GRAPH_OPTIONS,
}
_PROFILE_PREPROCESSING = {
    'standard': 'reference',
    'intel_gpu': 'lookup',
    'intel_npu': 'affine',
}
_PROFILE_ALLOWED_DEVICES = {
    'standard': ['*'],
    'intel_gpu': ['GPU'],
    'intel_npu': ['NPU'],
}


@dataclass(frozen=True)
class ExportSettings:
    path: Path
    path_relative: str
    profile: str
    model_config: Path
    model_config_relative: str
    output_root: Path
    height: int
    width: int
    valid_iters: int
    max_disp: int
    context_upsample_mode: str
    context_upsample_sigma: float
    opset: int
    preprocessing: str
    graph: dict[str, Any]


def _positive_config_int(value, field):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Export config {field} must be a positive integer")
    return value


def _positive_config_float(value, field):
    if isinstance(value, bool):
        raise ValueError(f"Export config {field} must be positive and finite")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Export config {field} must be positive and finite"
        ) from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"Export config {field} must be positive and finite")
    return value


def _resolve_repo_relative(value, field):
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f"Export config {field} must be a repo-relative path")
    resolved = (REPO_ROOT / value).resolve()
    try:
        relative = resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"Export config {field} must stay inside the repository"
        ) from exc
    return resolved, relative


def _resolve_export_config_path(path):
    path = Path(path)
    resolved = path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()
    try:
        relative = resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError as exc:
        raise ValueError("--export-config must point inside the repository") from exc
    return resolved, relative


def load_export_config(path):
    """Load and fully validate one controlled export profile."""
    config_path, config_relative = _resolve_export_config_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Export config not found: {config_path}")
    try:
        with config_path.open('r', encoding='utf-8') as stream:
            raw = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid export config: {config_path}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Export config must be a mapping: {config_path}")

    expected_sections = {
        'schema_version', 'profile', 'model_config', 'output_root',
        'input', 'model', 'onnx', 'graph',
    }
    if set(raw) != expected_sections:
        missing = sorted(expected_sections - set(raw))
        extra = sorted(set(raw) - expected_sections)
        raise ValueError(
            f"Invalid export config sections: missing={missing}, extra={extra}"
        )
    if raw['schema_version'] != 1 or isinstance(raw['schema_version'], bool):
        raise ValueError("Export config schema_version must be 1")
    profile = raw['profile']
    if profile not in EXPORT_PROFILES:
        raise ValueError(f"Unsupported export profile: {profile!r}")

    model_config, model_config_relative = _resolve_repo_relative(
        raw['model_config'], 'model_config'
    )
    output_root, _ = _resolve_repo_relative(raw['output_root'], 'output_root')
    if not model_config.is_file():
        raise FileNotFoundError(f"Model config not found: {model_config}")

    input_config = raw['input']
    if not isinstance(input_config, dict) or set(input_config) != {'height', 'width'}:
        raise ValueError("Export config input must contain only height and width")
    height = _positive_config_int(input_config['height'], 'input.height')
    width = _positive_config_int(input_config['width'], 'input.width')

    model = raw['model']
    expected_model = {
        'valid_iters', 'max_disp', 'context_upsample_mode',
        'context_upsample_sigma',
    }
    if not isinstance(model, dict) or set(model) != expected_model:
        raise ValueError(
            "Export config model must contain valid_iters, max_disp, "
            "context_upsample_mode, and context_upsample_sigma"
        )
    valid_iters = _positive_config_int(model['valid_iters'], 'model.valid_iters')
    max_disp = _positive_config_int(model['max_disp'], 'model.max_disp')
    mode = model['context_upsample_mode']
    if mode not in ('standard', 'consistency_guided'):
        raise ValueError(f"Unsupported context upsample mode: {mode!r}")
    sigma = _positive_config_float(
        model['context_upsample_sigma'], 'model.context_upsample_sigma'
    )

    onnx_config = raw['onnx']
    if not isinstance(onnx_config, dict) or set(onnx_config) != {
        'opset', 'preprocessing'
    }:
        raise ValueError("Export config onnx must contain only opset and preprocessing")
    opset = _positive_config_int(onnx_config['opset'], 'onnx.opset')
    preprocessing = onnx_config['preprocessing']
    expected_preprocessing = _PROFILE_PREPROCESSING[profile]
    if preprocessing != expected_preprocessing:
        raise ValueError(
            f"Profile {profile!r} requires ONNX preprocessing "
            f"{expected_preprocessing!r}, got {preprocessing!r}"
        )

    graph = raw['graph']
    expected_graph = _PROFILE_GRAPH_OPTIONS[profile]
    if not isinstance(graph, dict) or graph != expected_graph:
        raise ValueError(
            f"Profile {profile!r} graph must match its controlled preset: "
            f"expected {expected_graph!r}, got {graph!r}"
        )
    if graph['geometry_pyramid_builder'] == 'slice' and max_disp % 8:
        raise ValueError(
            "geometry_pyramid_builder=slice requires max_disp divisible by 8"
        )

    return ExportSettings(
        path=config_path,
        path_relative=config_relative,
        profile=profile,
        model_config=model_config,
        model_config_relative=model_config_relative,
        output_root=output_root,
        height=height,
        width=width,
        valid_iters=valid_iters,
        max_disp=max_disp,
        context_upsample_mode=mode,
        context_upsample_sigma=sigma,
        opset=opset,
        preprocessing=preprocessing,
        graph=dict(graph),
    )


def model_normalization(config):
    """Read the single authoritative normalization source from INFERENCE."""
    inference = config.get('INFERENCE')
    if not isinstance(inference, dict):
        raise ValueError("Model config must define INFERENCE.MEAN and INFERENCE.STD")
    return normalization_from_config(inference)


# ============================================
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Export a controlled WAVEStereo schema-v2 ONNX bundle'
    )
    parser.add_argument(
        '--export-config', required=True,
        help='Repo-relative cfgs/export/*.yaml deployment profile',
    )
    parser.add_argument('--weights', type=str, default=None,
                        help='Model checkpoint path (defaults to PRETRAINED_MODEL in the config)')
    return parser.parse_args(argv)


def main():
    cli_args = parse_args()
    settings = load_export_config(cli_args.export_config)
    args = argparse.Namespace(
        weights=cli_args.weights,
        config=str(settings.model_config),
        save_path=str(settings.output_root),
        height=settings.height,
        width=settings.width,
        valid_iters=settings.valid_iters,
        max_disp=settings.max_disp,
        context_upsample_mode=settings.context_upsample_mode,
        context_upsample_sigma=settings.context_upsample_sigma,
        opset=settings.opset,
        preprocessing=settings.preprocessing,
        **settings.graph,
    )

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('make_onnx_wavestereo')

    graph_options = dict(settings.graph)
    export_profile = settings.profile
    bundle_dir = make_bundle_dir(
        args.save_path,
        export_profile,
        args.height,
        args.width,
        args.valid_iters,
        args.max_disp,
    )
    bundle_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Export profile: %s", export_profile)
    logger.info("Export config: %s", settings.path_relative)
    logger.info("Bundle directory: %s", bundle_dir)
    torch.autograd.set_grad_enabled(False)

    # Build model without the OpenStereo trainer framework.
    logger.info(f"Loading config: {args.config}")
    cfgs = load_config(args.config)

    cfgs.MODEL.MAX_DISP = args.max_disp
    cfgs.MODEL.VALID_ITERS = args.valid_iters

    cfgs.MODEL.CONTEXT_UPSAMPLE_MODE = args.context_upsample_mode
    cfgs.MODEL.CONTEXT_UPSAMPLE_SIGMA = args.context_upsample_sigma
    mean, std = model_normalization(cfgs)

    if args.weights:
        weights_path = Path(args.weights)
        if not weights_path.is_absolute():
            weights_path = (Path.cwd() / weights_path).resolve()
        cfgs.MODEL.PRETRAINED_MODEL = str(weights_path)

    logger.info(f"Building model and loading weights: {cfgs.MODEL.PRETRAINED_MODEL}")
    model = WAVEStereo(cfgs.MODEL).cuda().eval()
    if (
        export_profile == 'intel_npu'
        and model.context_upsample_mode == 'consistency_guided'
    ):
        model.context_upsample_reference_mode = 'npu_reduce_max'
    logger.info("Context upsampling mode: %s", model.context_upsample_mode)
    if model.context_upsample_mode == 'consistency_guided':
        logger.info("Context upsampling sigma: %g", model.context_upsample_sigma)
        logger.info(
            "Context upsampling reference selector: %s",
            model.context_upsample_reference_mode,
        )
    if cfgs.MODEL.PRETRAINED_MODEL:
        load_checkpoint(model, cfgs.MODEL.PRETRAINED_MODEL, strict=False, map_location='cpu')
    else:
        logger.warning("No weights were provided; exporting a randomly initialized model")

    replaced_deconvs = replace_deconvs_with_polyphase(model, args.polyphase_deconv)
    if replaced_deconvs:
        logger.info("Polyphase ConvTranspose2d replacements (%s): %s",
                    args.polyphase_deconv, ", ".join(replaced_deconvs))

    if args.dpt_conv_before_upsample:
        from wavestereo.model.update import DPTFusionBlock
        dpt_blocks = [module for module in model.modules() if isinstance(module, DPTFusionBlock)]
        for module in dpt_blocks:
            module.project_before_upsample = True
        logger.info("Moved DPT 1x1 projection before upsampling in %d FusionBlocks", len(dpt_blocks))

    # 应用 ONNX 兼容性补丁
    logger.info("Applying ONNX compatibility patches...")
    apply_onnx_patches(args.correlation_chunk_size, args.geometry_sampler,
                       args.batch_stereo_backbone, args.correlation_builder,
                       args.hoist_warp_left, args.late_image_stems,
                       args.correlation_reducer, args.geometry_pyramid_builder,
                       args.context_splitter)
    if args.correlation_chunk_size:
        logger.info("Correlation volume chunk size: %d", args.correlation_chunk_size)
    else:
        logger.info("Correlation volume chunking: disabled (full disparity range)")
    logger.info("Correlation volume builder: %s", args.correlation_builder)
    logger.info("Correlation volume reducer: %s", args.correlation_reducer)
    logger.info("Geometry volume sampler: %s", args.geometry_sampler)
    logger.info("Geometry pyramid builder: %s", args.geometry_pyramid_builder)
    logger.info("Context channel splitter: %s", args.context_splitter)
    logger.info("Batched stereo backbone: %s", args.batch_stereo_backbone)
    logger.info("Hoisted warp_fusion left branch: %s", args.hoist_warp_left)
    logger.info("Late image stems: %s", args.late_image_stems)

    # The external contract is identical for every profile. Only the baked
    # normalization implementation differs by target.
    native_wrapper = WavestereoOnnx(model, args.height, args.width)
    wrapper = BgrUint8OnnxInput(
        native_wrapper, args.preprocessing, mean=mean, std=std
    )
    wrapper.cuda().eval()

    # Static schema-v2 input: BGR uint8/NHWC.
    logger.info(f"Creating dummy input: {args.height}x{args.width}")
    left_img = torch.randint(
        0, 256, (1, args.height, args.width, 3),
        device='cuda', dtype=torch.uint8,
    )
    right_img = torch.randint(
        0, 256, (1, args.height, args.width, 3),
        device='cuda', dtype=torch.uint8,
    )

    # 测试前向传播
    logger.info("Testing forward pass...")
    with torch.no_grad():
        disp = wrapper(left_img, right_img)
    logger.info(f"  Output disparity shape: {disp.shape}")

    # 每个固定输入规格使用独立 bundle，bundle 内文件名保持稳定。
    onnx_path = bundle_dir / ONNX_FILENAME
    staging_onnx_path = bundle_dir / ".model.exporting.onnx"
    if staging_onnx_path.is_file():
        staging_onnx_path.unlink()
    logger.info(f"Exporting ONNX -> {onnx_path}")
    try:
        torch.onnx.export(
            wrapper,
            (left_img, right_img),
            staging_onnx_path,
            opset_version=args.opset,
            input_names=['left_image', 'right_image'],
            output_names=['disparity'],
            do_constant_folding=True,
        )

        stem_groupnorm_scope = args.stem_groupnorm_scope
        if stem_groupnorm_scope != 'none':
            rewrite_stem_instance_norm_as_group_norm(
                staging_onnx_path, model, scope=stem_groupnorm_scope
            )
            logger.info(
                "Image-stem GroupNormalization scope: %s",
                stem_groupnorm_scope,
            )
        if args.dpt_groupnorm:
            rewritten, num_groups = rewrite_dpt_instance_norm_as_group_norm(
                staging_onnx_path, model
            )
            logger.info(
                "DPT GroupNormalization: %d nodes, groups=%d",
                rewritten, num_groups,
            )

        # 校验通过后再原子替换 canonical ONNX，失败时保留原 bundle。
        import onnx
        onnx_model = onnx.load(staging_onnx_path)
        output_dims = onnx_model.graph.output[0].type.tensor_type.shape.dim
        for dim, value in zip(output_dims, (1, 1, args.height, args.width)):
            dim.ClearField('dim_param')
            dim.dim_value = value
        onnx.save(onnx_model, staging_onnx_path)
        onnx.checker.check_model(onnx_model)
        staging_onnx_path.replace(onnx_path)
        logger.info("ONNX model validation passed")
    finally:
        if staging_onnx_path.is_file():
            staging_onnx_path.unlink()

    # Schema v2 contains only model.onnx. Remove a stale legacy engine if this
    # directory was previously exported with schema v1.
    legacy_engine_path = bundle_dir / 'model.engine'
    if legacy_engine_path.is_file():
        legacy_engine_path.unlink()
        logger.info("Removed legacy artifact: %s", legacy_engine_path)

    checkpoint = cfgs.MODEL.PRETRAINED_MODEL
    checkpoint_metadata = (
        {'filename': Path(str(checkpoint)).name} if checkpoint else None
    )
    effective_opset = (
        21 if (stem_groupnorm_scope != 'none' or args.dpt_groupnorm)
        else args.opset
    )

    # 保存严格、可移植的 bundle metadata；不记录本机 checkpoint 绝对路径。
    cfg_dict = {
        'schema_version': BUNDLE_SCHEMA_VERSION,
        'profile': export_profile,
        'allowed_devices': _PROFILE_ALLOWED_DEVICES[export_profile],
        'model': {
            'name': 'WAVEStereo',
            'max_disp': args.max_disp,
            'valid_iters': args.valid_iters,
            'corr_radius': cfgs.MODEL.CORR_RADIUS,
            'corr_levels': cfgs.MODEL.CORR_LEVELS,
            'hidden_dim': cfgs.MODEL.HIDDEN_DIM,
            'context_upsample_mode': model.context_upsample_mode,
            'context_upsample_sigma': model.context_upsample_sigma,
            'context_upsample_reference': model.context_upsample_reference_mode,
        },
        'input': {
            'names': ['left_image', 'right_image'],
            'shape': [1, args.height, args.width, 3],
            'dtype': 'uint8',
            'layout': 'NHWC',
            'color_order': 'BGR',
            'preprocessing': {
                'location': 'graph',
                'implementation': args.preprocessing,
                'mean': mean,
                'std': std,
            },
            'internal_padding': {
                'mode': 'top_right_replicate',
                'divisible_by': 32,
                'top': (-args.height) % 32,
                'right': (-args.width) % 32,
            },
        },
        'output': {
            'name': 'disparity',
            'shape': [1, 1, args.height, args.width],
            'dtype': 'float32',
            'units': 'model_input_pixels_x',
        },
        'artifacts': {
            'onnx': {
                'filename': ONNX_FILENAME,
                'sha256': sha256_file(onnx_path),
            },
        },
        'export': {
            'config': settings.path_relative,
            'model_config': settings.model_config_relative,
            'checkpoint': checkpoint_metadata,
            'requested_opset': args.opset,
            'effective_opset': effective_opset,
            'graph': graph_options,
        },
    }
    cfg_path = bundle_dir / METADATA_FILENAME
    save_config(cfg_dict, cfg_path)
    load_bundle_metadata(bundle_dir)

    logger.info("=" * 50)
    logger.info("Export complete!")
    logger.info(f"  Bundle: {bundle_dir}")
    logger.info(f"  - {ONNX_FILENAME}")
    logger.info(f"  - {METADATA_FILENAME}")
    logger.info("=" * 50)


if __name__ == '__main__':
    main()
