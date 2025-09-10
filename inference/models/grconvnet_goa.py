# -*- coding: utf-8 -*-
"""
grconvnet_goa.py - 基于GOA注意力机制的改进GR-ConvNet

完全兼容原框架，在原有代码基础上集成改进模块：
1. 保持与原始 GR-ConvNet 完全兼容
2. 支持所有原有的训练/测试流程
3. 集成 GOA (抓取导向注意力) 和 AFF (自适应特征融合)
4. 支持模块：FPN/GOA/AFF/CBAM/SPD-Conv，模块可可选，支持灵活的消融实验配置

Author: [lms]
Date: 2025.8
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from inference.models.grasp_model import GraspModel, ResidualBlock


# ============================================================================
# 辅助模块定义
# ============================================================================

class FPN(nn.Module):
    """特征金字塔网络 - 多尺度特征融合"""

    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels_list
        ])

    def forward(self, c1, c2, c4):
        # c1: 224, c2: 112, c4: 56
        p4 = self.lateral_convs[2](c4)
        p3 = self.lateral_convs[1](c2) + F.interpolate(p4, size=c2.shape[-2:], mode='nearest')
        p2 = self.lateral_convs[0](c1) + F.interpolate(p3, size=c1.shape[-2:], mode='nearest')
        return p2, p3, p4


class SPDConv(nn.Module):
    """空间到深度卷积 - 空间结构增强"""

    def __init__(self, in_channels, scale=2, out_channels=None):
        super(SPDConv, self).__init__()
        self.scale = scale
        if out_channels is None:
            out_channels = in_channels

        expanded_channels = in_channels * (scale ** 2)
        self.conv = nn.Conv2d(expanded_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # Space-to-depth操作
        N, C, H, W = x.shape
        x = x.view(N, C, H // self.scale, self.scale, W // self.scale, self.scale)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(N, C * (self.scale ** 2), H // self.scale, W // self.scale)

        x = F.relu(self.bn(self.conv(x)))
        return x


class CBAM(nn.Module):
    """传统CBAM注意力机制 - 用于对比实验"""

    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()

        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        ca = self.channel_attention(x)
        x = x * ca

        # 空间注意力
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        sa_input = torch.cat([max_pool, avg_pool], dim=1)
        sa = self.spatial_attention(sa_input)
        x = x * sa

        return x


class GraspOrientedAttention(nn.Module):
    """抓取导向注意力机制 (GOA) - 支持 BN / GN 切换"""

    def __init__(self, channels, reduction=8, norm_type='BN', num_groups=8):
        """
        Args:
            channels (int): 输入通道数
            reduction (int): 通道缩减比
            norm_type (str): 归一化类型，可选 'BN' 或 'GN'
            num_groups (int): 当 norm_type='GN' 时，GN 的分组数
        """
        super(GraspOrientedAttention, self).__init__()
        self.channels = channels
        self.reduction = reduction

        # 根据 norm_type 选择归一化层
        def NormLayer(ch):
            if norm_type.upper() == 'GN':
                return nn.GroupNorm(num_groups, ch)
            elif norm_type.upper() == 'BN':
                return nn.BatchNorm2d(ch)
            else:
                raise ValueError("norm_type 必须是 'BN' 或 'GN'")

        # === 抓取三要素分支 ===
        # 位置分支：全局分布
        self.position_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            NormLayer(channels // reduction)
        )

        # 角度分支：局部方向
        self.angle_branch = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            NormLayer(channels // reduction)
        )

        # 宽度分支：空间结构
        self.width_branch = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            NormLayer(channels // reduction)
        )

        # 三要素融合（通道注意力）
        self.channel_fusion = nn.Sequential(
            nn.Conv2d(3 * (channels // reduction), channels, 1),
            nn.Sigmoid()
        )

        # 空间注意力（含标准差池化）
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=7, padding=3),
            NormLayer(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # === 通道注意力：抓取三要素 ===
        pos_feat = self.position_branch(x)  # 抓取位置特征
        angle_feat = self.angle_branch(x)  # 抓取角度特征
        width_feat = self.width_branch(x)  # 抓取宽度特征

        # 融合三要素特征
        combined_feat = torch.cat([pos_feat, angle_feat, width_feat], dim=1)
        channel_attention = self.channel_fusion(combined_feat)
        x_channel = x * channel_attention

        # === 空间注意力：含标准差池化 ===
        max_pool = torch.max(x_channel, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x_channel, dim=1, keepdim=True)
        std_pool = torch.std(x_channel, dim=1, keepdim=True)  # 创新点

        spatial_input = torch.cat([max_pool, avg_pool, std_pool], dim=1)
        spatial_attention = self.spatial_attention(spatial_input)

        # 输出
        output = x_channel * spatial_attention
        return output


class AdaptiveFeatureFusion(nn.Module):
    """自适应特征融合模块 (AFF) - 核心创新模块"""

    def __init__(self, channels):
        super(AdaptiveFeatureFusion, self).__init__()
        self.channels = channels

        # 特征重要性评估网络
        self.importance_evaluator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 2, 1),
            nn.Softmax(dim=1)
        )

        # 特征增强网络
        self.feature_enhancer = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels)
        )

        # 可学习的残差权重
        self.residual_weight = nn.Parameter(torch.ones(1) * 0.3)

    def forward(self, x1, x2):
        """
        Args:
            x1: 主特征流
            x2: 辅助特征流
        """
        # 尺寸对齐
        if x2.shape[-2:] != x1.shape[-2:]:
            x2 = F.interpolate(x2, size=x1.shape[-2:], mode='bilinear', align_corners=False)

        # 特征重要性评估
        combined_features = torch.cat([x1, x2], dim=1)
        importance_weights = self.importance_evaluator(combined_features)

        w1 = importance_weights[:, 0:1, :, :]
        w2 = importance_weights[:, 1:2, :, :]

        # 加权融合
        fused_features = w1 * x1 + w2 * x2

        # 特征增强
        enhanced_features = self.feature_enhancer(fused_features)

        # 残差连接
        output = enhanced_features + self.residual_weight * x1

        return output


# ============================================================================
# 主网络模型
# ============================================================================

class GenerativeResnet(GraspModel):
    """
    改进版 GR-ConvNet，集成GOA和AFF模块
    支持 FPN 模式和 U-Net 模式的跳跃连接切换
    支持 CBAM 模式和 GOA 模式的注意力机制切换
    完全兼容原框架的训练/测试流程，支持以下配置：
    - baseline: 原始结构
    - +UNet: 添加跳跃连接
    - +FPN: 添加多尺度特征
    - +SPD: 添加空间增强
    - +CBAM: 添加传统注意力
    - +GOA: 使用抓取导向注意力（替代CBAM）
    - +AFF: 使用自适应特征融合

    关键改进点：
    1. GOA: 针对抓取三要素的专门注意力机制
    2. AFF: 自适应特征融合替代简单相加
    3. 模块化设计，支持灵活的消融实验
    """

    def __init__(self,
                 input_channels=4,
                 output_channels=1,
                 channel_size=32,
                 dropout=False,
                 prob=0.0,
                 # 新增参数：模块开关
                 use_upconv=False,  # 控制上采样方式
                 use_unet=False,  # 控制U-Net跳跃连接
                 use_fpn=False,  # 控制FPN多尺度特征
                 use_cbam=False,  # 控制CBAM传统注意力
                 use_goa=False,  # 控制GOA抓取导向注意力
                 use_aff=False,  # 控制AFF自适应特征融合
                 use_spd=False,  # 控制SPD空间增强
                 spd_scale=2):  # 新增参数：SPD缩放因子
        super(GenerativeResnet, self).__init__()

        # 保存配置
        self.config = {
            'use_upconv': use_upconv,
            'use_unet': use_unet,
            'use_fpn': use_fpn,
            'use_spd': use_spd,
            'use_cbam': use_cbam,
            'use_goa': use_goa,
            'use_aff': use_aff,
            'spd_scale': spd_scale
        }
        self.use_cbam, self.use_goa, self.use_aff = use_cbam, use_goa, use_aff
        self.use_upconv, self.use_unet, self.use_fpn, self.use_spd = use_upconv, use_unet, use_fpn, use_spd

        cs = channel_size

        # 互斥检查
        if use_cbam and use_goa:
            raise ValueError("CBAM和GOA不能同时使用")
        if use_unet and use_fpn:
            raise ValueError("U-Net和FPN是两种不同的跳跃连接策略, 不能同时使用")
        if use_spd and not use_fpn:
            print("Warning: SPD-Conv通常与FPN一起使用以获得最佳效果")

        # === 1.编码器 (与原网络完全一致) ===
        self.conv1 = nn.Conv2d(input_channels, cs, kernel_size=9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm2d(cs)
        self.conv2 = nn.Conv2d(cs, cs * 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(cs * 2)
        self.conv3 = nn.Conv2d(cs * 2, cs * 4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(cs * 4)
        # 残差块
        self.res1 = ResidualBlock(cs * 4, cs * 4)
        self.res2 = ResidualBlock(cs * 4, cs * 4)
        self.res3 = ResidualBlock(cs * 4, cs * 4)
        self.res4 = ResidualBlock(cs * 4, cs * 4)
        self.res5 = ResidualBlock(cs * 4, cs * 4)

        # === 2.解码器 (保持原网络输出层逻辑) ===
        if self.use_upconv:
            # 使用 "上采样 + 卷积" 方式
            self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv4 = nn.Conv2d(cs * 4, cs * 2, 3, padding=1)
            self.bn4 = nn.BatchNorm2d(cs * 2)

            self.up5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv5 = nn.Conv2d(cs * 2, cs, 3, padding=1)
            self.bn5 = nn.BatchNorm2d(cs)

            self.conv6 = nn.Conv2d(cs, cs, 9, padding=4)
        else:
            self.conv4 = nn.ConvTranspose2d(cs * 4, cs * 2, kernel_size=4, stride=2, padding=1, output_padding=1)
            self.bn4 = nn.BatchNorm2d(cs * 2)

            self.conv5 = nn.ConvTranspose2d(cs * 2, cs, kernel_size=4, stride=2, padding=2, output_padding=1)
            self.bn5 = nn.BatchNorm2d(cs)

            self.conv6 = nn.ConvTranspose2d(cs, cs, kernel_size=9, stride=1, padding=4)

        # === 3. 跳跃连接/特征增强模块 可选增强模块 ===
        if use_fpn:
            self.fpn = FPN([cs, cs * 2, cs * 4], cs)
            self.fpn_proj1 = nn.Conv2d(cs, cs * 2, kernel_size=1)  # p3(cs) -> x1(cs*2)
            self.fpn_proj2 = nn.Conv2d(cs, cs, kernel_size=1)  # p2(cs) -> x2(cs) (通道数相同，可选)
        elif use_unet:
            # === U-Net 模式下的专用模块 (用于对齐跳跃连接的通道) ===
            self.unet_proj1 = nn.Conv2d(cs * 2, cs * 2, kernel_size=1)  # c2(cs*2) -> x1(cs*2)
            self.unet_proj2 = nn.Conv2d(cs, cs, kernel_size=1)  # c1(cs)   -> x2(cs)
        # === 可选的 SPD 模块 (仅在 FPN 模式下考虑) ===
        if use_spd:
            self.spd = SPDConv(cs, scale=spd_scale, out_channels=cs)

        # === 4. 注意力模块 (按需创建) ===
        if use_cbam:
            self.attention1 = CBAM(cs * 2)
            self.attention2 = CBAM(cs)
        elif use_goa:
            self.attention1 = GraspOrientedAttention(cs * 2)
            self.attention2 = GraspOrientedAttention(cs)
        else:
            self.attention1 = nn.Identity()
            self.attention2 = nn.Identity()

        # ===  5. 特征融合模块 (按需创建) ===
        if use_unet or use_fpn:
            if use_aff:
                self.fusion1 = AdaptiveFeatureFusion(cs * 2)
                self.fusion2 = AdaptiveFeatureFusion(cs)
            else:
                self.fusion1 = self._simple_fusion
                self.fusion2 = self._simple_fusion

        # === 输出头 (与原网络保持一致) ===
        if use_upconv:
            self.pos_output = nn.Conv2d(in_channels=cs, out_channels=output_channels, kernel_size=1)
            self.cos_output = nn.Conv2d(in_channels=cs, out_channels=output_channels, kernel_size=1)
            self.sin_output = nn.Conv2d(in_channels=cs, out_channels=output_channels, kernel_size=1)
            self.width_output = nn.Conv2d(in_channels=cs, out_channels=output_channels, kernel_size=1)
        else:
            self.pos_output = nn.Conv2d(in_channels=cs, out_channels=output_channels, kernel_size=2)
            self.cos_output = nn.Conv2d(in_channels=cs, out_channels=output_channels, kernel_size=2)
            self.sin_output = nn.Conv2d(in_channels=cs, out_channels=output_channels, kernel_size=2)
            self.width_output = nn.Conv2d(in_channels=cs, out_channels=output_channels, kernel_size=2)

        # === Dropout (与原网络保持一致) ===
        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)
        self.dropout_sin = nn.Dropout(p=prob)
        self.dropout_wid = nn.Dropout(p=prob)

        # === 权重初始化 (与原网络保持一致) ===
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def _encode(self, x):
        """编码器，提取多层特征"""
        c1 = F.relu(self.bn1(self.conv1(x)))  # 224x224, 32
        c2 = F.relu(self.bn2(self.conv2(c1)))  # 112x112, 64
        c3 = F.relu(self.bn3(self.conv3(c2)))  # 56x56, 128

        # 残差块处理
        c4 = c3
        c4 = self.res1(c4)
        c4 = self.res2(c4)
        c4 = self.res3(c4)
        c4 = self.res4(c4)
        c4 = self.res5(c4)  # 56x56, 128

        return c1, c2, c4

    def _simple_fusion(self, x1, x2):
        """U-Net 风格的简单相加融合, 确保尺寸一致"""
        if x2.shape[-2:] != x1.shape[-2:]:
            x2 = F.interpolate(x2, size=x1.shape[-2:], mode='bilinear', align_corners=True)
        return x1 + x2

    def forward(self, x_in):
        """
        前向传播函数
        支持 FPN 模式和 U-Net 模式的动态切换
        """
        # 1. 编码阶段: 提取多尺度特征
        # c1: (cs, 224, 224), c2: (cs*2, 112, 112), c4: (cs*4, 56, 56)
        c1, c2, c4 = self._encode(x_in)

        # 2. 解码阶段 x: (cs*2, 112, 112) 第一次上采样
        if self.use_upconv:
            x = self.up4(c4)
            x = F.relu(self.bn4(self.conv4(x)))
        else:
            x = F.relu(self.bn4(self.conv4(c4)))

        # 3. 第一次可选的跳跃连接、融合与注意力
        # 如果 use_fpn 和 use_unet 都为 False, 则 x 保持不变 (Baseline)
        if self.use_fpn:
            # FPN 模式
            p2, p3, _ = self.fpn(c1, c2, c4)  # 先计算特征金字塔
            p3_proj = self.fpn_proj1(p3)
            # 特征融合
            x = self.fusion1(x, p3_proj)
        elif self.use_unet:
            # U-Net 模式
            c2_proj = self.unet_proj1(c2)
            # 特征融合
            x = self.fusion1(x, c2_proj)

        # 4.解码阶段 第二次上采样 x: (cs, 224, 224)
        if self.use_upconv:
            x = self.up5(x)
            x = F.relu(self.bn5(self.conv5(x)))
        else:
            x = F.relu(self.bn5(self.conv5(x)))

        # 5. 第二次可选的跳跃连接、融合与注意力
        if self.use_fpn:
            # FPN 模式
            # 注意: p2, p3 已经在前面计算过，这里直接用 p2
            p2_proj = self.fpn_proj2(p2)
            x = self.fusion2(x, p2_proj)
            # 在 FPN 模式下，才考虑融合 SPD 特征
            if self.use_spd:
                spd_features = self.spd(c1)
                spd_up = F.interpolate(spd_features, size=x.shape[-2:], mode='bilinear', align_corners=True)
                # 再次融合
                x = self.fusion2(x, spd_up)
        elif self.use_unet:
            # U-Net 模式
            c1_proj = self.unet_proj2(c1)
            x = self.fusion2(x, c1_proj)

        # 6. 应用注意力模块
        x = self.attention2(x)

        # 7. 最终卷积层，细化特征
        x = self.conv6(x)

        # 8. 输出头，生成最终的抓取图
        if self.dropout:
            pos_output = self.pos_output(self.dropout_pos(x))
            cos_output = self.cos_output(self.dropout_cos(x))
            sin_output = self.sin_output(self.dropout_sin(x))
            width_output = self.width_output(self.dropout_wid(x))
        else:
            pos_output = self.pos_output(x)
            cos_output = self.cos_output(x)
            sin_output = self.sin_output(x)
            width_output = self.width_output(x)

        return pos_output, cos_output, sin_output, width_output

    def get_config_name(self):
        """获取配置名称 - 便于实验管理"""
        config = self.config
        name_parts = ['goanet']
        additions = []
        if config['use_upconv']:
            additions.append('upconv')
        if config['use_unet']:
            additions.append('unet')
        if config['use_fpn']:
            additions.append('fpn')
        if config['use_spd']:
            additions.append('spd')
        if config['use_cbam']:
            additions.append('cbam')
        if config['use_goa']:
            additions.append('goa')
        if config['use_aff']:
            additions.append('aff')

        if additions:
            name_parts.append('+'.join(additions))
        else:
            name_parts.append('baseline')

        return '_'.join(name_parts)


if __name__ == "__main__":
    print("🧪 测试改进版 GR-ConvNet 模型")

    # 测试不同配置
    configs = {
        'baseline': {},
        'unet_only': {'use_unet': True},
        'upconv_only': {'use_upconv': True},
        'goa_only': {'use_goa': True},
        'aff_only': {'use_aff': True},
        'fpn_only': {'use_fpn': True},
        'cbam_only': {'use_cbam': True},
        'goa_aff': {'use_goa': True, 'use_aff': True},
        'fpn_spd': {'use_fpn': True, 'use_spd': True},
        'fpn_spd_cam': {'use_fpn': True, 'use_spd': True, 'use_cbam': True},
        'fpn_goa': {'use_fpn': True, 'use_goa': True},
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for name, config in configs.items():
        print(f"\n测试配置: {name}")

        # 创建模型
        model = GenerativeResnet(
            input_channels=4,
            channel_size=32,
            dropout=False,
            **config
        ).to(device)

        print(f"配置名称: {model.get_config_name()}")
        print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

        # 测试前向传播
        with torch.no_grad():
            x = torch.randn(2, 4, 300, 300).to(device)
            outputs = model(x)

            print(f"输出形状: {[out.shape for out in outputs]}")

        # 测试损失计算（兼容性测试）
        try:
            y = [torch.randn_like(out) for out in outputs]
            loss_dict = model.compute_loss(x, y)
            print(f"损失计算成功: {loss_dict['loss'].item():.4f}")
        except Exception as e:
            print(f"损失计算失败: {e}")

    print("\n✅ 模型基础测试完成！")
