# -*- coding: utf-8 -*-
# @File       : grconvnet_mas.py
# @Description: 改进版 GR-ConvNet (grconvnet_mas) —— FPN/CBAM/SPD-Conv，可与原训练/评测流程兼容

import torch
import torch.nn as nn
import torch.nn.functional as F

from inference.models.grasp_model import GraspModel, ResidualBlock
from inference.model_utils.fpn import FPN
from inference.model_utils.spdconv import SPDConv
from inference.model_utils.cbam import CBAM


class GRConvNetMAS(GraspModel):
    """
    GRConvNet-MAS: 多尺度融合+注意力+空间增强的抓取检测网络
    M - Multi-scale Fusion (FPN特征金字塔)
    A - Attention Mechanism (CBAM注意力机制)
    S - Spatial Enhancement (SPD空间结构增强)

    核心特性:
      1. 多尺度特征融合(FPN)
      2. 通道与空间双重注意力(CBAM)
      3. 空间结构保持(SPD-Conv)
      4. 严格上采样解码器
    """

    def __init__(self, input_channels=4, channel_size=32,
                 use_fpn=True, use_cbam=True, use_spd=True, spd_scale=2,
                 dropout=False, prob=0.0):
        super().__init__()
        cs = channel_size

        # ========= 编码器 =========
        # 224x224 -> 224x224
        self.conv1 = nn.Conv2d(input_channels, cs, kernel_size=9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm2d(cs)
        # 224 -> 112
        self.conv2 = nn.Conv2d(cs, cs * 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(cs * 2)
        # 112 -> 56
        self.conv3 = nn.Conv2d(cs * 2, cs * 4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(cs * 4)

        # 5 个残差块（保持通道 4*cs）
        self.res1 = ResidualBlock(cs * 4, cs * 4)
        self.res2 = ResidualBlock(cs * 4, cs * 4)
        self.res3 = ResidualBlock(cs * 4, cs * 4)
        self.res4 = ResidualBlock(cs * 4, cs * 4)
        self.res5 = ResidualBlock(cs * 4, cs * 4)

        # ========= 选项 =========
        self.use_fpn = use_fpn
        self.use_cbam = use_cbam
        self.use_spd = use_spd
        if use_spd:
            # 对浅层 c1(224x224) 做 space-to-depth_full 增强
            self.spd = SPDConv(cs, scale=spd_scale, out_ch=cs)

        # ========= 解码器（严格 2× 上采样，确保 56->112->224） =========
        # 56 -> 112，通道 4*cs -> 2*cs
        self.up1 = nn.ConvTranspose2d(cs * 4, cs * 2, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.bn4 = nn.BatchNorm2d(cs * 2)
        # 112 -> 224，通道 2*cs -> cs
        self.up2 = nn.ConvTranspose2d(cs * 2, cs, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.bn5 = nn.BatchNorm2d(cs)

        # 细化卷积
        self.refine = nn.ConvTranspose2d(cs, cs, kernel_size=9, stride=1, padding=4)

        # ========= FPN 与通道对齐 =========
        if use_fpn:
            # FPN 输入：[c1(cs), c2(2cs), c4(4cs)]，输出统一为 cs
            self.fpn = FPN([cs, cs * 2, cs * 4], cs)
            # 1/2 尺度融合时，x 的通道 2*cs，而 p3 为 cs，需要 1×1 投影
            self.proj_p3 = nn.Conv2d(cs, cs * 2, kernel_size=1)
        else:
            self.proj_p3 = nn.Identity()

        # ========= CBAM 注意力 =========
        self.cbam1 = CBAM(cs * 2) if use_cbam else nn.Identity()  # 1/2 尺度通道 2*cs
        self.cbam2 = CBAM(cs) if use_cbam else nn.Identity()  # 1/1 尺度通道 cs

        # ========= 输出头：改为 1×1，保证输出 224×224 =========
        self.pos_output = nn.Conv2d(cs, 1, kernel_size=1)
        self.cos_output = nn.Conv2d(cs, 1, kernel_size=1)
        self.sin_output = nn.Conv2d(cs, 1, kernel_size=1)
        self.width_output = nn.Conv2d(cs, 1, kernel_size=1)

        # Dropout（保持与原框架兼容）
        self.dropout = dropout
        self.dp_pos = nn.Dropout(p=prob) if dropout else nn.Identity()
        self.dp_cos = nn.Dropout(p=prob) if dropout else nn.Identity()
        self.dp_sin = nn.Dropout(p=prob) if dropout else nn.Identity()
        self.dp_wid = nn.Dropout(p=prob) if dropout else nn.Identity()

        # 初始化
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    # =============== 编码 ===============
    def _encode(self, x):
        c1 = F.relu(self.bn1(self.conv1(x)))  # 224x224, cs
        c2 = F.relu(self.bn2(self.conv2(c1)))  # 112x112, 2cs
        c3 = F.relu(self.bn3(self.conv3(c2)))  # 56x56,  4cs
        x = self.res1(c3);
        x = self.res2(x);
        x = self.res3(x);
        x = self.res4(x);
        x = self.res5(x)  # 56x56, 4cs
        return c1, c2, x  # 返回 c1(224,cs), c2(112,2cs), c4(56,4cs)

    # =============== 前向 ===============
    def forward(self, x_in):
        c1, c2, c4 = self._encode(x_in)

        # 只算一次 FPN，得到三个尺度的特征 (与 c1/c2/c4 对应)
        if self.use_fpn:
            p2, p3, _ = self.fpn(c1, c2, c4)  # p2≈224x224 (cs), p3≈112x112 (cs)

        # → 1/2 (112x112) 融合
        x = F.relu(self.bn4(self.up1(c4)))  # 56->112, 通道 2*cs
        if self.use_fpn:
            p3 = F.interpolate(p3, size=x.shape[-2:], mode='nearest')  # 保底尺寸对齐
            p3 = self.proj_p3(p3)  # cs -> 2*cs，通道对齐
            x = x + p3
        x = self.cbam1(x)

        # → 1/1 (224x224) 融合
        x = F.relu(self.bn5(self.up2(x)))  # 112->224, 通道 cs
        if self.use_fpn:
            p2 = F.interpolate(p2, size=x.shape[-2:], mode='nearest')  # 保底尺寸对齐
            x = x + p2

        # SPD 小目标增强（来自 c1 的浅层高频）
        if self.use_spd:
            x = x + F.interpolate(self.spd(c1), size=x.shape[-2:], mode='nearest')

        x = self.cbam2(x)
        x = self.refine(x)

        # 输出头（224x224）
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
    net1 = GRConvNetMAS(input_channels=4, channel_size=32, use_fpn=True, use_cbam=True, use_spd=True, spd_scale=2,
                        dropout=False).to(device)
    with torch.no_grad():
        pos, cos, sin, wid = net1(x)
    print(f"  input: {tuple(x.shape)}")
    print(f"  output: pos={tuple(pos.shape)}, cos={tuple(cos.shape)}, sin={tuple(sin.shape)}, width={tuple(wid.shape)}")
    print(f"  params: {count_params(net1):,}")

    # 测试 2：仅 FPN
    print("\n[Self-check] Test-2: use_fpn=1, use_cbam=0, use_spd=0")
    net2 = GRConvNetMAS(input_channels=4, channel_size=32, use_fpn=True, use_cbam=False, use_spd=False,
                        dropout=False).to(device)
    with torch.no_grad():
        pos, cos, sin, wid = net2(x)
    print(f"  output: pos={tuple(pos.shape)}, cos={tuple(cos.shape)}, sin={tuple(sin.shape)}, width={tuple(wid.shape)}")
    print(f"  params: {count_params(net2):,}")

    # 测试 3：仅 SPD
    print("\n[Self-check] Test-3: use_fpn=0, use_cbam=0, use_spd=1")
    net3 = GRConvNetMAS(input_channels=4, channel_size=32, use_fpn=False, use_cbam=False, use_spd=True, spd_scale=2,
                        dropout=False).to(device)
    with torch.no_grad():
        pos, cos, sin, wid = net3(x)
    print(f"  output: pos={tuple(pos.shape)}, cos={tuple(cos.shape)}, sin={tuple(sin.shape)}, width={tuple(wid.shape)}")
    print(f"  params: {count_params(net3):,}")

    # 额外检查：通道与尺寸在两个融合点的预期（需要临时暴露内部形状的话，可把 forward 中的关键张量打印出来）
    print("\n[Self-check] Done. 如果上面输出均成功，说明通道与尺寸在 FPN/CBAM/SPD 融合处已正确对齐。")
