# -*- coding: utf-8 -*-
# UNet-based Grasp Prediction with Optimized Residual Blocks, Attention Gates, Coord-Attention,
# and task-adaptive losses (Focal for Q, sin-cos periodic angle, Log-L1 for width).
#
# Drop-in replacement for your GenerativeResnet, keeping the same API.
# - Input : (N, C, H, W)  -> C can be 1 (D), 3 (RGB), or 4 (RGB-D)
# - Output: pos, cos, sin, width  (each: (N, 1, H, W))
# - Loss  : compute_loss(xc, yc) where yc = (y_pos, y_cos, y_sin, y_width)

import torch
import torch.nn as nn
import torch.nn.functional as F

from inference.models.grasp_model import GraspModel


# ---------------------- Building Blocks ----------------------

class DSConv(nn.Module):
    """Depthwise-Separable Conv -> BN -> SiLU"""

    def __init__(self, in_c, out_c, k=3, s=1, p=1, d=1):
        super().__init__()
        self.dw = nn.Conv2d(in_c, in_c, k, s, p, dilation=d, groups=in_c, bias=False)
        self.pw = nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return self.act(self.bn(x))


class ResidualBlockOptim(nn.Module):
    """
    Optimized residual block:
      - DSConv -> DSConv
      - Optional Squeeze-Excitation
      - Optional dilation on the first conv to enlarge RF
    """

    def __init__(self, c, dilation=1, use_se=True, r=16):
        super().__init__()
        self.conv1 = DSConv(c, c, k=3, p=dilation, d=dilation)
        self.conv2 = DSConv(c, c, k=3, p=1, d=1)
        self.use_se = use_se
        if use_se:
            red = max(c // r, 8)
            self.squeeze = nn.AdaptiveAvgPool2d(1)
            self.fc1 = nn.Conv2d(c, red, 1, bias=False)
            self.fc2 = nn.Conv2d(red, c, 1, bias=False)

    def forward(self, x):
        idn = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.use_se:
            w = self.squeeze(x)
            w = F.silu(self.fc1(w), inplace=True)
            w = torch.sigmoid(self.fc2(w))
            x = x * w
        return x + idn


class AttentionGate(nn.Module):
    """Attention U-Net style gate for skip connections"""

    def __init__(self, in_x, in_g, inter):
        super().__init__()
        self.theta_x = nn.Conv2d(in_x, inter, 1, bias=False)
        self.phi_g = nn.Conv2d(in_g, inter, 1, bias=False)
        self.psi = nn.Conv2d(inter, 1, 1, bias=False)

    def forward(self, x, g):
        # ensure spatial match
        if x.size()[2:] != g.size()[2:]:
            g = F.interpolate(g, size=x.size()[2:], mode='bilinear', align_corners=False)
        a = F.relu(self.theta_x(x) + self.phi_g(g), inplace=True)
        a = torch.sigmoid(self.psi(a))
        return x * a


class CoordAttn(nn.Module):
    """Coordinate Attention: lightweight, direction-aware, good for pixel-wise heads"""

    def __init__(self, c, r=32):
        super().__init__()
        m = max(8, c // r)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(c, m, 1, bias=False)
        self.act = nn.SiLU(inplace=True)
        self.conv_h = nn.Conv2d(m, c, 1, bias=False)
        self.conv_w = nn.Conv2d(m, c, 1, bias=False)

    def forward(self, x):
        n, c, h, w = x.shape
        x_h = self.pool_h(x)  # (n,c,h,1)
        x_w = self.pool_w(x).transpose(2, 3)  # (n,c,1,w)
        y = torch.cat([x_h, x_w], dim=2)  # (n,c,h+1,max(1,w))
        y = self.act(self.conv1(y))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.transpose(2, 3)
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        return x * a_h * a_w


class DepthGuidedGating(nn.Module):
    """
    Depth-guided feature gating to suppress invalid/noisy depth regions.
    Important: we resize the depth map to feat size inside forward.
    """

    def __init__(self, in_channels, reduction=8):
        super().__init__()
        mid = max(in_channels // reduction, 8)
        self.depth_proj = nn.Sequential(
            nn.Conv2d(1, mid, 3, padding=1, bias=False),
            nn.SiLU(True),
            nn.Conv2d(mid, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        # Background/low-confidence path: lightweight smoothing on feat
        self.bg_smooth = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)

    def forward(self, feat, depth_01):
        # 1) resize depth to feat spatial size
        if depth_01.size()[2:] != feat.size()[2:]:
            depth_01 = F.interpolate(depth_01, size=feat.size()[2:], mode='bilinear', align_corners=False)
        # 2) gating map
        g = self.sigmoid(self.depth_proj(depth_01))
        # 3) background path
        bg = self.bg_smooth(feat)
        # 4) gated fusion
        out = g * feat + (1.0 - g) * bg
        return out


# ---------------------- UNet Grasp ----------------------

class UNetGrasp(GraspModel):
    def __init__(self,
                 input_channels=4,
                 channel_size=32,
                 output_channels=1,
                 dropout=False,
                 prob=0.1,
                 use_ag=True,
                 use_coord_attn=True,
                 use_dgg=False):
        super().__init__()
        C1, C2, C3, C4 = channel_size, channel_size * 2, channel_size * 4, channel_size * 8
        self.channel_size = channel_size
        self.dropout = bool(dropout)
        self.prob = float(prob)

        # -------- Encoder --------
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels, C1, 3, padding=1, bias=False),
            nn.BatchNorm2d(C1), nn.SiLU(True),
            ResidualBlockOptim(C1, use_se=True)
        )
        self.down1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(C1, C2, 3, padding=1, bias=False),
            nn.BatchNorm2d(C2), nn.SiLU(True),
            ResidualBlockOptim(C2, use_se=True)
        )
        self.down2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(C2, C3, 3, padding=1, bias=False),
            nn.BatchNorm2d(C3), nn.SiLU(True),
            ResidualBlockOptim(C3, dilation=2, use_se=True)
        )
        self.down3 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(C3, C4, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(C4), nn.SiLU(True),
            ResidualBlockOptim(C4, dilation=2, use_se=True)
        )

        # Optional depth-guided gating at mid-level (after enc3)
        self.use_dgg = use_dgg
        if use_dgg:
            self.dgg3 = DepthGuidedGating(C3, reduction=8)

        # Attention gates for skips
        self.use_ag = use_ag
        if use_ag:
            # in_g must match the upsampled channel!
            self.ag3 = AttentionGate(C3, C3, C3 // 2)
            self.ag2 = AttentionGate(C2, C2, C2 // 2)
            self.ag1 = AttentionGate(C1, C1, C1 // 2)

        # -------- Decoder (Resize-Conv: interpolate + 1x1 conv) --------
        # Use 1x1 to project channels; interpolation handles exact spatial alignment.
        self.up3_conv = nn.Conv2d(C4, C3, 1, bias=False)
        self.dec3 = nn.Sequential(
            nn.Conv2d(C3 + C3, C3, 3, padding=1, bias=False),
            nn.BatchNorm2d(C3), nn.SiLU(True),
            ResidualBlockOptim(C3, use_se=True)
        )

        self.up2_conv = nn.Conv2d(C3, C2, 1, bias=False)
        self.dec2 = nn.Sequential(
            nn.Conv2d(C2 + C2, C2, 3, padding=1, bias=False),
            nn.BatchNorm2d(C2), nn.SiLU(True),
            ResidualBlockOptim(C2, use_se=True)
        )

        self.up1_conv = nn.Conv2d(C2, C1, 1, bias=False)
        self.dec1 = nn.Sequential(
            nn.Conv2d(C1 + C1, C1, 3, padding=1, bias=False),
            nn.BatchNorm2d(C1), nn.SiLU(True),
            ResidualBlockOptim(C1, use_se=True)
        )

        self.use_coord_attn = use_coord_attn
        if use_coord_attn:
            self.ca2 = CoordAttn(C2)
            self.ca1 = CoordAttn(C1)

        # Dropout modules
        if self.dropout:
            self.do_e1 = nn.Dropout2d(self.prob)
            self.do_e2 = nn.Dropout2d(self.prob)
            self.do_e3 = nn.Dropout2d(self.prob)
            self.do_b = nn.Dropout2d(self.prob)
            self.do_d3 = nn.Dropout2d(self.prob)
            self.do_d2 = nn.Dropout2d(self.prob)
            self.do_d1 = nn.Dropout2d(self.prob)
            # head-wise dropout
            self.do_pos = nn.Dropout2d(self.prob)
            self.do_cos = nn.Dropout2d(self.prob)
            self.do_sin = nn.Dropout2d(self.prob)
            self.do_wid = nn.Dropout2d(self.prob)

        # 1x1 heads
        self.pos_head = nn.Conv2d(C1, output_channels, 1)
        self.cos_head = nn.Conv2d(C1, output_channels, 1)
        self.sin_head = nn.Conv2d(C1, output_channels, 1)
        self.width_head = nn.Conv2d(C1, output_channels, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        支持 1/3/4 通道输入：
          - 4 通道：RGB-D（D 在第 4 通道）
          - 3 通道：RGB
          - 1 通道：Depth-only
        当 use_dgg=True 且没有深度通道时自动跳过 DGG。
        """
        n, c, h, w = x.shape

        # parse depth if exists
        depth = None
        if c >= 4:
            depth = x[:, 3:4, :, :]
        elif c == 1:
            depth = x

        # Encoder
        e1 = self.enc1(x)  # H
        if self.dropout: e1 = self.do_e1(e1)

        e2 = self.enc2(self.down1(e1))  # H/2
        if self.dropout: e2 = self.do_e2(e2)

        e3 = self.enc3(self.down2(e2))  # H/4

        # Optional DGG
        if self.use_dgg and (depth is not None):
            # normalize depth to [0,1]
            d_min = depth.amin(dim=(2, 3), keepdim=True)
            d_max = depth.amax(dim=(2, 3), keepdim=True)
            d_norm = torch.clamp((depth - d_min) / (d_max - d_min + 1e-6), 0., 1.)
            e3 = self.dgg3(e3, d_norm)
        if self.dropout: e3 = self.do_e3(e3)

        # Bottleneck
        b = self.bottleneck(self.down3(e3))  # H/8 (floor for odd sizes)
        if self.dropout: b = self.do_b(b)

        # Decoder (Resize-Conv to align spatial size with skip features)
        # up to e3 size
        u3 = F.interpolate(b, size=e3.shape[2:], mode='bilinear', align_corners=False)
        u3 = self.up3_conv(u3)
        s3 = self.ag3(e3, u3) if self.use_ag else e3
        d3 = self.dec3(torch.cat([u3, s3], dim=1))
        if self.dropout: d3 = self.do_d3(d3)

        # up to e2 size
        u2 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        u2 = self.up2_conv(u2)
        s2 = self.ag2(e2, u2) if self.use_ag else e2
        d2 = self.dec2(torch.cat([u2, s2], dim=1))
        if self.use_coord_attn: d2 = self.ca2(d2)
        if self.dropout: d2 = self.do_d2(d2)

        # up to e1 size
        u1 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        u1 = self.up1_conv(u1)
        s1 = self.ag1(e1, u1) if self.use_ag else e1
        d1 = self.dec1(torch.cat([u1, s1], dim=1))
        if self.use_coord_attn: d1 = self.ca1(d1)
        if self.dropout: d1 = self.do_d1(d1)

        # Heads (apply head-wise dropout if enabled)
        if self.dropout:
            pos = self.pos_head(self.do_pos(d1))
            cos = self.cos_head(self.do_cos(d1))
            sin = self.sin_head(self.do_sin(d1))
            width = self.width_head(self.do_wid(d1))
        else:
            pos = self.pos_head(d1)
            cos = self.cos_head(d1)
            sin = self.sin_head(d1)
            width = self.width_head(d1)

        return pos, cos, sin, width

    # ---------------------- Losses ----------------------
    @staticmethod
    def _focal_bce(prob, target, alpha=0.25, gamma=2.0, eps=1e-6):
        # prob in [0,1], target in [0,1]
        p = torch.clamp(prob, eps, 1 - eps)
        pos_term = -alpha * target * ((1 - p) ** gamma) * torch.log(p)
        neg_term = -(1 - alpha) * (1 - target) * (p ** gamma) * torch.log(1 - p)
        return (pos_term + neg_term).mean()

    @staticmethod
    def _log_l1(pred_pos, target_pos, eps=1e-3):
        # pred/target expected >=0 (apply relu before)
        return F.l1_loss(torch.log(torch.clamp(pred_pos, min=eps)),
                         torch.log(torch.clamp(target_pos, min=eps)))

    def get_config_name(self):
        config_name = f"UNetGrasp_{self.channel_size}"
        if self.use_ag:
            config_name += "_AG"
        if self.use_coord_attn:
            config_name += "_CA"
        if self.use_dgg:
            config_name += "_DGG"
        return config_name

    def compute_loss(self, xc, yc):
        """
        极简损失：BCE(Logits) + 掩膜 SmoothL1（角度与宽度）
        - 返回结构保持不变
        - y_pos ∈ [0,1]；建议 y_cos,y_sin = cos(2θ), sin(2θ)；y_width 为像素宽度（已随几何增强同步缩放）
        """
        y_pos, y_cos, y_sin, y_width = yc
        pos_logits, cos_pred, sin_pred, width_pred = self(xc)  # pos_logits 是logits

        eps = 1e-6
        M = torch.clamp(y_pos, 0.0, 1.0)  # 可抓掩膜

        # ---- 1) Q/pos: BCEWithLogits + 批内自动平衡 ----
        with torch.no_grad():
            pos_sum = y_pos.sum()
            neg_sum = (1.0 - y_pos).sum()
            pos_weight = (neg_sum / (pos_sum + eps)).clamp(0.5, 10.0)
        p_loss = torch.nn.functional.binary_cross_entropy_with_logits(pos_logits, y_pos, pos_weight=pos_weight)

        # ---- 2) 角度：掩膜 SmoothL1（最简）----
        l1_cos = torch.nn.functional.smooth_l1_loss(cos_pred, y_cos, reduction='none')
        l1_sin = torch.nn.functional.smooth_l1_loss(sin_pred, y_sin, reduction='none')
        cos_loss = (l1_cos * M).sum() / (M.sum() + eps)
        sin_loss = (l1_sin * M).sum() / (M.sum() + eps)

        # ---- 3) 宽度：掩膜 SmoothL1（像素域，最简）----
        width_loss_map = torch.nn.functional.smooth_l1_loss(width_pred, y_width, reduction='none')
        width_loss = (width_loss_map * M).sum() / (M.sum() + eps)

        total = p_loss + cos_loss + sin_loss + width_loss

        return {
            'loss': total,
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss
            },
            'pred': {
                'pos': torch.sigmoid(pos_logits),  # 概率图 [0,1]
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred
            }
        }


# ---------------------- Quick Self Test ----------------------
if __name__ == '__main__':
    print("测试 UNetGrasp 模型")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 测试不同配置
    cfgs = {
        'AG_CA': {'use_ag': True, 'use_coord_attn': True, 'use_dgg': False},
        'AG_CA_DGG': {'use_ag': True, 'use_coord_attn': True, 'use_dgg': True},
    }

    for name, cfg in cfgs.items():
        model = UNetGrasp(
            input_channels=4,
            channel_size=32,
            dropout=False,
            **cfg
        ).to(device)
        print(f"配置名称: {model.get_config_name()}")
        print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

        # 224x224 测试
        with torch.no_grad():
            x = torch.randn(2, 4, 224, 224).to(device)
            outputs = model(x)
            print(f"224 输出形状: {[o.shape for o in outputs]}")
            y = [torch.rand_like(outputs[0]).to(device),  # y_pos in [0,1]
                 torch.randn_like(outputs[1]).to(device),  # y_cos
                 torch.randn_like(outputs[2]).to(device),  # y_sin
                 torch.rand_like(outputs[3]).abs().to(device) * 80.0]  # y_width >=0
            loss_dict = model.compute_loss(x, y)
            print(f" 224 损失: {loss_dict['loss'].item():.4f}")

        # 300x300（奇数尺寸处理验证）
        with torch.no_grad():
            x = torch.randn(2, 4, 300, 300).to(device)
            outputs = model(x)
            print(f"300 输出形状: {[o.shape for o in outputs]}")
        print("\n" + "=" * 50 + "\n")
