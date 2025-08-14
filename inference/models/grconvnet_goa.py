# -*- coding: utf-8 -*-
"""
grconvnet_goa.py.py - 集成GOA和AFF的改进GR-ConvNet

完全兼容原框架，在原有代码基础上集成改进模块：
1. 保持与原始 GraspModel 的完全兼容
2. 支持所有原有的训练/测试流程
3. 支持多尺度特征融合
3. 集成 GOA (抓取导向注意力) 和 AFF (自适应特征融合)
4. 支持灵活的消融实验配置

Author: [lms]
Date: 2025
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
# 主网络模型 - 继承GraspModel保证兼容性
# ============================================================================

class GenerativeResnet(GraspModel):
    """
    改进版 GR-ConvNet，集成GOA和AFF模块

    完全兼容原框架的训练/测试流程，支持以下配置：
    - baseline: 原始结构
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
                 use_fpn=False,
                 use_spd=False,
                 use_cbam=False,
                 use_goa=False,
                 use_aff=False,
                 spd_scale=2):
        super(GenerativeResnet, self).__init__()

        # 保存配置
        self.config = {
            'use_fpn': use_fpn,
            'use_spd': use_spd,
            'use_cbam': use_cbam,
            'use_goa': use_goa,
            'use_aff': use_aff,
            'spd_scale': spd_scale
        }
        self.use_fpn, self.use_spd, self.use_goa, self.use_aff = use_fpn, use_spd, use_goa, use_aff
        cs = channel_size

        # 互斥检查
        if use_cbam and use_goa:
            raise ValueError("CBAM和GOA不能同时使用")

        # === 编码器 (与原始完全一致) ===
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

        # === 可选增强模块 ===
        if use_fpn:
            self.fpn = FPN([cs, cs * 2, cs * 4], cs)
            # 通道对齐投影
            self.channel_proj = nn.Conv2d(cs, cs * 2, kernel_size=1)

        if use_spd:
            self.spd = SPDConv(cs, scale=spd_scale, out_channels=cs)

        # === 解码器 (保持原始输出层逻辑) ===
        self.conv4 = nn.ConvTranspose2d(cs * 4, cs * 2, kernel_size=4, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(cs * 2)

        self.conv5 = nn.ConvTranspose2d(cs * 2, cs, kernel_size=4, stride=2, padding=2, output_padding=1)
        self.bn5 = nn.BatchNorm2d(cs)

        self.conv6 = nn.ConvTranspose2d(cs, cs, kernel_size=9, stride=1, padding=4)

        # === 注意力模块 ===
        if use_cbam:
            self.attention1 = CBAM(cs * 2)
            self.attention2 = CBAM(cs)
        elif use_goa:
            self.attention1 = GraspOrientedAttention(cs * 2)  # 创新
            self.attention2 = GraspOrientedAttention(cs)  # 创新
        else:
            self.attention1 = nn.Identity()
            self.attention2 = nn.Identity()

        # === 特征融合模块 ===
        if use_aff:
            self.fusion1 = AdaptiveFeatureFusion(cs * 2)  # 创新
            self.fusion2 = AdaptiveFeatureFusion(cs)  # 创新
        else:
            self.fusion1 = self._simple_fusion
            self.fusion2 = self._simple_fusion

        # === 输出头 (与原始保持一致) ===
        # 注意：使用2x2卷积保持与原始grconvnet3.py一致
        self.pos_output = nn.Conv2d(in_channels=cs, out_channels=output_channels, kernel_size=2)
        self.cos_output = nn.Conv2d(in_channels=cs, out_channels=output_channels, kernel_size=2)
        self.sin_output = nn.Conv2d(in_channels=cs, out_channels=output_channels, kernel_size=2)
        self.width_output = nn.Conv2d(in_channels=cs, out_channels=output_channels, kernel_size=2)

        # === Dropout (与原始保持一致) ===
        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)
        self.dropout_sin = nn.Dropout(p=prob)
        self.dropout_wid = nn.Dropout(p=prob)

        # === 权重初始化 (与原始保持一致) ===
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
        """简单相加融合"""
        if x2.shape[-2:] != x1.shape[-2:]:
            x2 = F.interpolate(x2, size=x1.shape[-2:], mode='nearest')
        return x1 + x2

    def forward(self, x_in):
        """
        前向传播 - 完全兼容原框架的输入输出格式
        """
        # === 编码阶段 ===
        c1, c2, c4 = self._encode(x_in)

        # === 多尺度特征提取 ===
        if self.use_fpn:
            p2, p3, p4 = self.fpn(c1, c2, c4)

        # === SPD空间增强 ===
        if self.use_spd:
            spd_features = self.spd(c1)

        # === 解码阶段 (保持原始流程) ===
        # 第一阶段：56x56 -> 112x112
        x = F.relu(self.bn4(self.conv4(c4)))
        # FPN特征融合，第一个融合点
        if self.use_fpn:
            x = self.fusion1(x, self.channel_proj(F.interpolate(p3, size=x.shape[-2:], mode='nearest')))
        # 第一个注意力点
        x = self.attention1(x)  # GOA or CBAM or identity

        # 第二阶段：112x112 -> 224x224
        x = F.relu(self.bn5(self.conv5(x)))
        # FPN特征融合，第二个融合点
        if self.use_fpn:
            x = self.fusion2(x, F.interpolate(p2, size=x.shape[-2:], mode='nearest'))
        # SPD特征融合
        if self.use_spd:
            x = self.fusion2(x, F.interpolate(spd_features, size=x.shape[-2:], mode='nearest'))
        # 第二个注意力点
        x = self.attention2(x)

        # 特征细化
        x = self.conv6(x)

        # === 输出阶段 (与原始完全一致) ===
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
        name_parts = ['GRConvNet']

        additions = []
        if config['use_fpn']: additions.append('FPN')
        if config['use_spd']: additions.append('SPD')
        if config['use_cbam']: additions.append('CBAM')
        if config['use_goa']: additions.append('GOA')
        if config['use_aff']: additions.append('AFF')

        if additions:
            name_parts.append('+'.join(additions))
        else:
            name_parts.append('Baseline')

        return '_'.join(name_parts)


# ============================================================================
# 预定义配置工厂函数 - 便于快速实验
# ============================================================================

def create_baseline_model(**kwargs):
    """基线模型：原始GR-ConvNet"""
    return GenerativeResnet(**kwargs)


def create_mas_model(**kwargs):
    """MAS模型：FPN + SPD + CBAM"""
    kwargs.update({
        'use_fpn': True,
        'use_spd': True,
        'use_cbam': True
    })
    return GenerativeResnet(**kwargs)


def create_goa_only_model(**kwargs):
    """仅GOA模型：FPN + SPD + GOA"""
    kwargs.update({
        'use_fpn': True,
        'use_spd': True,
        'use_goa': True
    })
    return GenerativeResnet(**kwargs)


def create_aff_only_model(**kwargs):
    """仅AFF模型：FPN + SPD + CBAM + AFF"""
    kwargs.update({
        'use_fpn': True,
        'use_spd': True,
        'use_cbam': True,
        'use_aff': True
    })
    return GenerativeResnet(**kwargs)


def create_improved_model(**kwargs):
    """完整改进模型：FPN + SPD + GOA + AFF"""
    kwargs.update({
        'use_fpn': True,
        'use_spd': True,
        'use_goa': True,
        'use_aff': True
    })
    return GenerativeResnet(**kwargs)


# ============================================================================
# 使用示例和测试代码
# ============================================================================

if __name__ == "__main__":
    print("🧪 测试改进版 GR-ConvNet 模型")

    # 测试不同配置
    configs = {
        'baseline': {},
        'mas': {'use_fpn': True, 'use_spd': True, 'use_cbam': True},
        'goa_only': {'use_fpn': True, 'use_spd': True, 'use_goa': True},
        'aff_only': {'use_fpn': True, 'use_spd': True, 'use_cbam': True, 'use_aff': True},
        'full': {'use_fpn': True, 'use_spd': True, 'use_goa': True, 'use_aff': True}
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
