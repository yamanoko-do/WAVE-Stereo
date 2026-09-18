import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
#  Heads
# ═══════════════════════════════════════════════════════════════════════════════

class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=2):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class DispHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, output_dim=1):
        super(DispHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


# ═══════════════════════════════════════════════════════════════════════════════
#  ConvGRU  (proven — from original best)
# ═══════════════════════════════════════════════════════════════════════════════

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim, input_dim, kernel_size=3):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)

    def forward(self, h, c, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx) + c[0])
        r = torch.sigmoid(self.convr(hx) + c[1])
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)) + c[2])
        h = (1-z) * h + z * q
        return h


# ═══════════════════════════════════════════════════════════════════════════════
#  UnifiedMotionEncoder  — BasicMotionEncoder + warp branch
#
#  corr(18) ─→ convc1(1x1) ─→ convc2(3x3) ──────────────────────┐
#  disp(1)  ─→ convd1(7x7) ─→ convd2(3x3) ──────────────────────┤
#  warp(48) ─→ warp_fusion(3x3) ─→ warp_conv(3x3) ──────────────┤
#                                                                ├─ cat(192) ─→ conv(192→128) ─→ conv(128→63) ─→ cat+disp ─→ 64
#  Keeps BasicMotionEncoder structure, adds warp as a 3rd branch.
# ═══════════════════════════════════════════════════════════════════════════════

class UnifiedMotionEncoder(nn.Module):
    def __init__(self, args, feat_dim=24, hidden_dim=64):
        super().__init__()
        cor_planes = args.CORR_LEVELS * (2 * args.CORR_RADIUS + 1)

        # corr path (identical to BasicMotionEncoder)
        self.convc1 = nn.Conv2d(cor_planes, 64, 1, padding=0)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)

        # disp path (identical to BasicMotionEncoder)
        self.convd1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convd2 = nn.Conv2d(64, 64, 3, padding=1)

        # warp path (new — WAFT-style implicit matching)
        self.warp_fusion = nn.Conv2d(feat_dim * 2, 64, 3, padding=1)
        self.warp_conv   = nn.Conv2d(64, 64, 3, padding=1)

        # fusion: all 3 branches → hidden_dim
        self.conv  = nn.Conv2d(64 * 3, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, hidden_dim - 1, 3, padding=1)

    def forward(self, feat_left, feat_right, disp, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))

        disp_ = F.relu(self.convd1(disp))
        disp_ = F.relu(self.convd2(disp_))

        warped = disp_warp(feat_right, disp)
        warp  = F.relu(self.warp_fusion(torch.cat([feat_left, warped], dim=1)))
        warp  = F.relu(self.warp_conv(warp))

        out = torch.cat([cor, disp_, warp], dim=1)
        out = F.relu(self.conv(out))
        out = F.relu(self.conv2(out))
        return torch.cat([out, disp], dim=1)


# ═══════════════════════════════════════════════════════════════════════════════
#  Utility functions
# ═══════════════════════════════════════════════════════════════════════════════

def pool2x(x):
    return F.avg_pool2d(x, 3, stride=2, padding=1)

def pool4x(x):
    return F.avg_pool2d(x, 5, stride=4, padding=1)

def interp(x, dest):
    original_dtype = x.dtype
    x_fp32 = x.float()
    interp_args = {'mode': 'bilinear', 'align_corners': True}
    with torch.cuda.amp.autocast(enabled=False):
        output_fp32 = F.interpolate(x_fp32, dest.shape[2:], **interp_args)
    if original_dtype != torch.float32:
        output = output_fp32.to(original_dtype)
    else:
        output = output_fp32
    return output


# ═══════════════════════════════════════════════════════════════════════════════
#  WAFT — feature warping (auxiliary signal)
# ═══════════════════════════════════════════════════════════════════════════════

def disp_warp(feature, disp, padding_mode='border'):
    b, _, h, w = feature.size()
    x_range = torch.arange(0, w, device=feature.device).view(1, 1, w).expand(1, h, w).type_as(feature)
    y_range = torch.arange(0, h, device=feature.device).view(1, h, 1).expand(1, h, w).type_as(feature)
    grid = torch.cat((x_range, y_range), dim=0)                  # [2, H, W]
    grid = grid.unsqueeze(0).expand(b, 2, h, w)                  # [B, 2, H, W]

    offset = torch.cat((-disp, torch.zeros_like(disp)), dim=1)    # [B, 2, H, W]
    sample_grid = grid + offset

    sample_grid[:, 0, :, :] = 2.0 * sample_grid[:, 0, :, :] / (w - 1) - 1.0
    sample_grid[:, 1, :, :] = 2.0 * sample_grid[:, 1, :, :] / (h - 1) - 1.0
    sample_grid = sample_grid.permute(0, 2, 3, 1)                # [B, H, W, 2]

    return F.grid_sample(feature, sample_grid, mode='bilinear',
                         padding_mode=padding_mode, align_corners=True)




# ═══════════════════════════════════════════════════════════════════════════════
#  PGCG
# ═══════════════════════════════════════════════════════════════════════════════

class MultiheadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv  = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)

        attn = (q.float() @ k.float().transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1).to(x.dtype)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(out))


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=3, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = MultiheadSelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x.float()).to(x.dtype))
        x = x + self.mlp(self.norm2(x.float()).to(x.dtype))
        return x


class DPTResidualConv(nn.Module):
    def __init__(self, dim, k=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, dim, k, padding=k // 2, bias=True),
            nn.InstanceNorm2d(dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, dim, k, padding=k // 2, bias=True),
        )

    def forward(self, x):
        return self.conv(x) + x


class DPTFusionBlock(nn.Module):
    def __init__(self, dim, out_dim=None, project_before_upsample=False):
        super().__init__()
        out_dim = out_dim or dim
        self.project_before_upsample = project_before_upsample
        self.res_conf_unit1 = DPTResidualConv(dim)
        self.res_conf_unit2 = DPTResidualConv(dim)
        self.out_conv = nn.Conv2d(dim, out_dim, 1, bias=True)

    def forward(self, main_feat, skip_feat=None, target_size=None):
        out = main_feat
        if skip_feat is not None:
            out = out + self.res_conf_unit1(skip_feat)
        out = self.res_conf_unit2(out)

        # A pointwise affine projection commutes with bilinear interpolation.
        # Applying it at the lower resolution is therefore mathematically
        # equivalent while substantially reducing work when out_dim <= dim.
        if self.project_before_upsample:
            out = self.out_conv(out)

        if target_size is None:
            target_size = (main_feat.shape[2] * 2, main_feat.shape[3] * 2)

        with torch.amp.autocast('cuda', enabled=False):
            out = F.interpolate(
                out.float(), size=target_size, mode='bilinear', align_corners=False
            ).to(main_feat.dtype)
        if not self.project_before_upsample:
            out = self.out_conv(out)
        return out


class SimplifiedDPT(nn.Module):
    def __init__(self, dim, out_dim, num_levels=3, inner_dim=None):
        super().__init__()
        inner_dim = inner_dim or dim
        self.num_levels = num_levels

        self.scratch = nn.ModuleList([
            nn.Conv2d(dim, inner_dim, 3, padding=1, bias=False)
            for _ in range(num_levels)
        ])
        for m in self.scratch:
            nn.init.normal_(m.weight, std=0.01)

        self.refine = nn.ModuleList([
            DPTFusionBlock(inner_dim, inner_dim) for _ in range(num_levels - 1)
        ])

        self.final_refine = DPTFusionBlock(inner_dim, out_dim)
        nn.init.zeros_(self.final_refine.out_conv.weight)
        nn.init.zeros_(self.final_refine.out_conv.bias)

    def forward(self, feats, target_h, target_w):
        proj = [self.scratch[i](feats[i]) for i in range(self.num_levels)]
        out = proj[-1]
        for i in range(self.num_levels - 2, -1, -1):
            up_size = (proj[i].shape[2] * 2, proj[i].shape[3] * 2)
            out = self.refine[i](out, proj[i], target_size=up_size)
        out = self.final_refine(out, target_size=(target_h, target_w))
        return out


class TransformerPathway(nn.Module):
    def __init__(self, motion_dim=64, hidden_dim=64, tf_dim=128,
                 tf_layers=3, tf_heads=4, tf_mlp_ratio=3, dpt_levels=3):
        super().__init__()
        self.tf_dim = tf_dim
        self.tf_layers = tf_layers
        self.dpt_levels = min(dpt_levels, tf_layers)

        self.pool = nn.AvgPool2d(8, stride=8)
        self.down_proj = nn.Sequential(
            nn.Conv2d(motion_dim, tf_dim, 1, bias=False),
            nn.GroupNorm(min(8, tf_dim), tf_dim),
            nn.ReLU(inplace=True),
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(tf_dim, tf_heads, tf_mlp_ratio)
            for _ in range(tf_layers)
        ])

        if dpt_levels >= tf_layers:
            self.dpt_indices = list(range(tf_layers))
        else:
            step = tf_layers / dpt_levels
            self.dpt_indices = [int(step * (i + 1)) - 1 for i in range(dpt_levels)]

        self.dpt = SimplifiedDPT(tf_dim, hidden_dim, dpt_levels, inner_dim=tf_dim)

    def forward(self, motion_features):
        B, _, H, W = motion_features.shape
        x = self.pool(motion_features)
        x = self.down_proj(x)
        _, _, h, w = x.shape
        N = h * w

        tokens = x.flatten(2).transpose(1, 2)
        dpt_feats = []
        for i, blk in enumerate(self.blocks):
            tokens = blk(tokens)
            if i in self.dpt_indices:
                dpt_feats.append(tokens.transpose(1, 2).reshape(B, self.tf_dim, h, w))

        global_feat = self.dpt(dpt_feats, H, W)
        return global_feat


# ═══════════════════════════════════════════════════════════════════════════════
#  UpdateBlock  — UnifiedMotionEncoder + ConvGRU [+ optional Transformer]
#
#  Architecture:
#    motion = UnifiedMotionEncoder(feat_l, feat_r, disp, corr)
#           → corr + feat_l + warped_r + disp interact from layer 1
#    net = ConvGRU(net, context, motion)
#    net += tf_gate * TransformerPathway(motion)           ← periodic global
#    delta_disp = DispHead(net)
# ═══════════════════════════════════════════════════════════════════════════════

class TransformerUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=64):
        super().__init__()
        self.args = args
        self.hidden_dim = hidden_dim

        # ── Unified encoder (corr + warp interact natively) ──
        feat_dim = args.get('BACKBONE_DIM', 24)
        self.encoder = UnifiedMotionEncoder(args, feat_dim=feat_dim, hidden_dim=hidden_dim)

        # ── GRU (proven) ──
        self.gru = ConvGRU(hidden_dim, hidden_dim)

        # ── Transformer global context ──
        tf_dim       = args.get('TF_DIM', 128)
        tf_layers    = args.get('TF_LAYERS', 3)
        tf_heads     = args.get('TF_HEADS', 4)
        tf_mlp_ratio = args.get('TF_MLP_RATIO', 3)
        dpt_levels   = args.get('TF_DPT_LEVELS', 3)
        self.refine_every = args.get('TF_REFINE_EVERY', 4)

        self.transformer = TransformerPathway(
            motion_dim=hidden_dim, hidden_dim=hidden_dim,
            tf_dim=tf_dim, tf_layers=tf_layers, tf_heads=tf_heads,
            tf_mlp_ratio=tf_mlp_ratio, dpt_levels=dpt_levels,
        )
        self.tf_gate = nn.Parameter(torch.full([1], -0.1))

        # ── Heads ──
        self.disp_head = DispHead(hidden_dim, hidden_dim=128, output_dim=1)
        self.mask_feat_4 = nn.Sequential(
            nn.Conv2d(hidden_dim, 32, 3, padding=1),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, net, inp, feat_left=None, feat_right=None, disp=None, corr=None, itr=0):
        motion = self.encoder(feat_left, feat_right, disp, corr)
        net = self.gru(net, inp, motion)

        if self.refine_every > 0 and itr % self.refine_every == 0:
            global_feat = self.transformer(motion)
            net = net + self.tf_gate * global_feat

        delta_disp = self.disp_head(net)
        mask_feat_4 = self.mask_feat_4(net)
        return net, delta_disp, mask_feat_4
