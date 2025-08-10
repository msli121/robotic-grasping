# -*- coding: utf-8 -*-
# @Time       : 2025/8/11 0:34
# @File       : grconvnet_mfa.py.py
# @Description:
# -*- coding: utf-8 -*-
# @File       : grconvnet_mas_optimized.py
# @Description: 优化版grconvnet_mas - 改进上采样/特征融合/残差块设计

import torch
import torch.nn as nn
import torch.nn.functional as F

from inference.models.grasp_model import GraspModel, ResidualBlock
from inference.model_utils.fpn import FPN
from inference.model_utils.spdconv import SPDConv
from inference.model_utils.cbam import CBAM


class FusionBlock(nn.Module):
    """特征融合模块 - 使用concat+conv代替简单加法"""

    def __init__(self, in_channels1, in_channels2, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels1 + in_channels2, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        # 确保尺寸匹配
        if x1.size()[-2:] != x2.size()[-2:]:
            x2 = F.interpolate(x2, size=x1.shape[-2:], mode='nearest')
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)


class UpsampleBlock(nn.Module):
    """上采样模块 - 双线性上采样+卷积代替转置卷积"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gn = nn.GroupNorm(8, out_channels)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return F.relu(self.gn(self.conv(x)))


class EnhancedHead(nn.Module):
    """增强型输出头 - 增加非线性能力"""

    def __init__(self, in_channels):
        super().__init__()
        mid_channels = max(in_channels // 2, 16)  # 确保最小通道数
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.gn = nn.GroupNorm(8, mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, 1, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.gn(self.conv1(x)))
        return self.conv2(x)


class GRConvNetMFA(GraspModel):
    """
        GRConvNetMFA: 多尺度融合与注意力增强的抓取检测网络
        M - Multi-scale Fusion (FPN特征金字塔)
        F - Feature Refinement (特征精炼融合模块)
        A - Attention Mechanism (CBAM注意力机制)

        核心特性:
          1. 多尺度特征融合(FPN)
          2. 特征精炼融合模块(concat+conv)
          3. 通道与空间双重注意力(CBAM)
          4. 空间结构保持(SPD-Conv)
          5. 渐进式上采样解码器
    """

    def __init__(self, input_channels=4, channel_size=32,
                 use_fpn=True, use_cbam=True, use_spd=True, spd_scale=2,
                 dropout=False, prob=0.0):
        super().__init__()
        cs = channel_size

        # ========= 编码器 =========
        self.conv1 = nn.Conv2d(input_channels, cs, kernel_size=9, stride=1, padding=4)
        self.gn1 = nn.GroupNorm(8, cs)

        self.conv2 = nn.Conv2d(cs, cs * 2, kernel_size=4, stride=2, padding=1)
        self.gn2 = nn.GroupNorm(8, cs * 2)

        self.conv3 = nn.Conv2d(cs * 2, cs * 4, kernel_size=4, stride=2, padding=1)
        self.gn3 = nn.GroupNorm(8, cs * 4)

        # 多样化残差块
        self.res_blocks = nn.Sequential(
            ResidualBlock(cs * 4, cs * 4),
            ResidualBlock(cs * 4, cs * 4),
            ResidualBlock(cs * 4, cs * 4),
            ResidualBlock(cs * 4, cs * 4),
            ResidualBlock(cs * 4, cs * 4)
        )

        # ========= 选项 =========
        self.use_fpn = use_fpn
        self.use_cbam = use_cbam
        self.use_spd = use_spd

        if use_spd:
            self.spd = SPDConv(cs, scale=spd_scale, out_ch=cs)

        # ========= 解码器 =========
        self.up1 = UpsampleBlock(cs * 4, cs * 2)  # 56->112
        self.up2 = UpsampleBlock(cs * 2, cs)  # 112->224

        # 特征细化
        self.refine = nn.Sequential(
            nn.Conv2d(cs, cs, kernel_size=3, padding=1),
            nn.GroupNorm(8, cs),
            nn.ReLU(inplace=True),
            nn.Conv2d(cs, cs, kernel_size=3, padding=1),
            nn.GroupNorm(8, cs),
            nn.ReLU(inplace=True)
        )

        # ========= FPN与融合模块 =========
        if use_fpn:
            self.fpn = FPN([cs, cs * 2, cs * 4], cs)
            # 融合模块
            self.fusion1 = FusionBlock(cs * 2, cs, cs * 2)  # 用于up1输出和p3融合
            self.fusion2 = FusionBlock(cs, cs, cs)  # 用于up2输出和p2融合

        # ========= CBAM注意力 =========
        self.cbam1 = CBAM(cs * 2) if use_cbam else nn.Identity()
        self.cbam2 = CBAM(cs) if use_cbam else nn.Identity()

        # ========= 输出头 =========
        self.pos_output = EnhancedHead(cs)
        self.cos_output = EnhancedHead(cs)
        self.sin_output = EnhancedHead(cs)
        self.width_output = EnhancedHead(cs)

        # Dropout
        self.dropout = dropout
        self.dp_pos = nn.Dropout(p=prob) if dropout else nn.Identity()
        self.dp_cos = nn.Dropout(p=prob) if dropout else nn.Identity()
        self.dp_sin = nn.Dropout(p=prob) if dropout else nn.Identity()
        self.dp_wid = nn.Dropout(p=prob) if dropout else nn.Identity()

        # 初始化
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _encode(self, x):
        """编码器前向传播"""
        c1 = F.relu(self.gn1(self.conv1(x)))

        # 在编码阶段融入SPD特征
        if self.use_spd:
            spd_feat = self.spd(c1)
            spd_feat = F.interpolate(spd_feat, size=c1.shape[-2:], mode='bilinear', align_corners=True)
            c1 = c1 + spd_feat

        c2 = F.relu(self.gn2(self.conv2(c1)))
        c3 = F.relu(self.gn3(self.conv3(c2)))
        c4 = self.res_blocks(c3)
        return c1, c2, c4

    def forward(self, x_in):
        c1, c2, c4 = self._encode(x_in)

        # FPN特征金字塔
        if self.use_fpn:
            p2, p3, _ = self.fpn(c1, c2, c4)

        # 第一次上采样：56->112
        x = self.up1(c4)

        # 与FPN的p3特征融合
        if self.use_fpn:
            x = self.fusion1(x, p3)  # 融合

        x = self.cbam1(x)

        # 第二次上采样：112->224
        x = self.up2(x)

        # 与FPN的p2特征融合
        if self.use_fpn:
            x = self.fusion2(x, p2)  # 融合

        x = self.cbam2(x)
        x = self.refine(x)

        # 输出头
        pos = self.pos_output(self.dp_pos(x))
        cos = self.cos_output(self.dp_cos(x))
        sin = self.sin_output(self.dp_sin(x))
        wid = self.width_output(self.dp_wid(x))

        return pos, cos, sin, wid


# ====================== 自检测试 验证模型结构通道数 ======================
if __name__ == "__main__":
    torch.set_printoptions(sci_mode=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def count_params(m):
        return sum(p.numel() for p in m.parameters())


    # 构造一条假输入（与训练一致：RGB-D → 4通道，224x224）
    x = torch.randn(2, 4, 224, 224).to(device)

    # 测试 1：全开（FPN+CBAM+SPD）
    print("\n[Self-check] Test-1: use_fpn=1, use_cbam=1, use_spd=1")
    net1 = GRConvNetMFA(input_channels=4, channel_size=32, use_fpn=True, use_cbam=True, use_spd=True, spd_scale=2,
                        dropout=False).to(device)
    with torch.no_grad():
        pos, cos, sin, wid = net1(x)
    print(f"  input: {tuple(x.shape)}")
    print(f"  output: pos={tuple(pos.shape)}, cos={tuple(cos.shape)}, sin={tuple(sin.shape)}, width={tuple(wid.shape)}")
    print(f"  params: {count_params(net1):,}")

    # 测试 2：仅 FPN
    print("\n[Self-check] Test-2: use_fpn=1, use_cbam=0, use_spd=0")
    net2 = GRConvNetMFA(input_channels=4, channel_size=32, use_fpn=True, use_cbam=False, use_spd=False,
                        dropout=False).to(device)
    with torch.no_grad():
        pos, cos, sin, wid = net2(x)
    print(f"  output: pos={tuple(pos.shape)}, cos={tuple(cos.shape)}, sin={tuple(sin.shape)}, width={tuple(wid.shape)}")
    print(f"  params: {count_params(net2):,}")

    # 测试 3：仅 SPD
    print("\n[Self-check] Test-3: use_fpn=0, use_cbam=0, use_spd=1")
    net3 = GRConvNetMFA(input_channels=4, channel_size=32, use_fpn=False, use_cbam=False, use_spd=True, spd_scale=2,
                        dropout=False).to(device)
    with torch.no_grad():
        pos, cos, sin, wid = net3(x)
    print(f"  output: pos={tuple(pos.shape)}, cos={tuple(cos.shape)}, sin={tuple(sin.shape)}, width={tuple(wid.shape)}")
    print(f"  params: {count_params(net3):,}")

    # 额外检查：通道与尺寸在两个融合点的预期（需要临时暴露内部形状的话，可把 forward 中的关键张量打印出来）
    print("\n[Self-check] Done. 如果上面输出均成功，说明通道与尺寸在 FPN/CBAM/SPD 融合处已正确对齐。")
