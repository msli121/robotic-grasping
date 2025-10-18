# -*- coding: utf-8 -*-
# @Time       : 2025/10/12 15:49
# @File       : hybrid_grasp_net.py.py
# @Description:
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from inference.models.grasp_model import GraspModel


# ============================================================================
# 1. 辅助模块定义 (TransGOA)
# ============================================================================
class PositionalEncoding2D(nn.Module):
    """
    一个辅助模块，用于动态生成2D位置编码。
    """

    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding2D, self).__init__()
        # 创建一个足够大的位置编码查找表
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (B, C, H, W)
        B, C, H, W = x.shape
        # 为 H 和 W 分别生成位置编码
        pe_h = self.pe[:H].unsqueeze(1).repeat(1, W, 1)  # (H, W, C)
        pe_w = self.pe[:W].unsqueeze(0).repeat(H, 1, 1)  # (H, W, C)
        # 将 H 和 W 的位置编码相加，并调整形状以匹配输入
        pe = (pe_h + pe_w).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        return pe.repeat(B, 1, 1, 1)  # (B, C, H, W)


class TransGOA(nn.Module):
    """
    Transformer-based Grasp-Oriented Attention.
    在网络瓶颈处对高级特征进行全局上下文建模。
    """

    def __init__(self, channels, num_heads=4, num_layers=2):
        super(TransGOA, self).__init__()
        # 1. 1x1卷积用于改变通道数，适配d_model
        self.proj_in = nn.Conv2d(channels, channels, 1)

        # 2. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=channels * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3. 1x1卷积恢复
        self.proj_out = nn.Conv2d(channels, channels, 1)

        # 4. 可学习的残差缩放因子
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        x_in = x

        # a. 投影和展平
        x = self.proj_in(x).flatten(2).permute(0, 2, 1)  # (B, H*W, C)

        # b. Transformer处理
        x = self.transformer(x)

        # c. 恢复形状和投影
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = self.proj_out(x)

        # d. 残差连接
        return x_in + self.gamma * x


class GraspFormer(nn.Module):
    """
    GraspFormer V2.0 - 采用动态位置编码，真正的即插即用。
    """

    def __init__(self, channels, num_queries=10, num_heads=4, num_layers=3):
        super(GraspFormer, self).__init__()
        self.num_queries = num_queries
        self.grasp_queries = nn.Parameter(torch.randn(1, num_queries, channels))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=channels, nhead=num_heads, dim_feedforward=channels * 4,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # --- 核心改动: 不再使用固定的 nn.Parameter ---
        # self.positional_encoding = nn.Parameter(torch.randn(1, channels, 56, 56)) # 旧
        # --- 替换为动态位置编码生成器 ---
        self.pos_encoder = PositionalEncoding2D(d_model=channels)

        self.output_proj = nn.Conv2d(channels, channels, 1)
        self.layer_norm = nn.LayerNorm(channels)  # 增加LayerNorm以稳定训练

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. 动态生成并添加位置编码
        pos = self.pos_encoder(x)
        x_pos = x + pos

        # 2. 准备输入
        memory = x_pos.flatten(2).permute(0, 2, 1)
        tgt = self.grasp_queries.repeat(B, 1, 1)

        # 3. Transformer Decoder 处理
        hs = self.transformer_decoder(tgt=self.layer_norm(tgt), memory=self.layer_norm(memory))

        # 4. 融合与广播
        grasp_feature_vector = hs.mean(dim=1)
        channel_attention = torch.sigmoid(self.output_proj(grasp_feature_vector.unsqueeze(-1).unsqueeze(-1)))

        return x * channel_attention


# ============================================================================
# 2. HybridGraspNet 主模型
# ============================================================================

class HybridGraspNet(GraspModel):
    """
    一个全新的、轻量级的混合抓取网络。
    - 编码器: 轻量级CNN Stem。
    - 瓶颈: TransGOA模块，用于全局上下文建模。
    - 解码器: U-Net结构，采用现代化的“上采样+卷积”。
    """

    def __init__(self, input_channels=4, output_channels=1, channel_size=32, dropout=False, prob=0.1,
                 bottleneck_type='graspformer'):
        super(HybridGraspNet, self).__init__()
        cs = channel_size

        # === 1. 编码器 (轻量化CNN Stem) ===
        # 3x3卷积堆叠
        self.conv1_1 = nn.Conv2d(input_channels, cs, 3, padding=1)  # 不改变空间尺寸，改变通道数为cs
        self.bn1_1 = nn.BatchNorm2d(cs)
        self.conv1_2 = nn.Conv2d(cs, cs, 3, padding=1)  # 不改变空间尺寸，改变通道数为cs
        self.bn1_2 = nn.BatchNorm2d(cs)

        self.conv2 = nn.Conv2d(cs, cs * 2, 4, stride=2, padding=1)  # 空间尺寸减半，通道数加倍
        self.bn2 = nn.BatchNorm2d(cs * 2)

        self.conv3 = nn.Conv2d(cs * 2, cs * 4, 4, stride=2, padding=1)  # 空间尺寸减半，通道数加倍
        self.bn3 = nn.BatchNorm2d(cs * 4)

        # === 2. 瓶颈 (Bottleneck) ===
        if bottleneck_type.lower() == 'transgoa':
            self.bottleneck = TransGOA(channels=cs * 4, num_heads=8, num_layers=4)
        elif bottleneck_type.lower() == 'graspformer':
            self.bottleneck = GraspFormer(channels=cs * 4, num_queries=16, num_heads=8, num_layers=3)
        else:
            # 默认或错误输入时，可以是一个简单的恒等映射或ResBlock
            # self.bottleneck = nn.Identity()
            raise ValueError(f"Unknown bottleneck_type '{bottleneck_type}'. Expected 'transgoa' or 'graspformer'.")

        # === 3. 解码器 (现代化的上采样) ===
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv4 = nn.Conv2d(cs * 4 + cs * 2, cs * 2, 3, padding=1)  # Concat: (cs*4 from upsample) + (cs*2 from skip)
        self.bn4 = nn.BatchNorm2d(cs * 2)

        self.up5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv5 = nn.Conv2d(cs * 2 + cs, cs, 3, padding=1)  # Concat: (cs*2 from upsample) + (cs from skip)
        self.bn5 = nn.BatchNorm2d(cs)

        self.conv6 = nn.Conv2d(cs, cs, 3, padding=1)

        # === 4. 输出头 (1x1卷积) ===
        self.pos_output = nn.Conv2d(cs, output_channels, 1)
        self.cos_output = nn.Conv2d(cs, output_channels, 1)
        self.sin_output = nn.Conv2d(cs, output_channels, 1)
        self.width_output = nn.Conv2d(cs, output_channels, 1)

        # === 5. Dropout & 权重初始化 ===
        self.dropout = dropout
        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)
        self.dropout_sin = nn.Dropout(p=prob)
        self.dropout_wid = nn.Dropout(p=prob)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x_in):
        # --- 编码 ---
        c1_pre = F.relu(self.bn1_1(self.conv1_1(x_in)))
        c1 = F.relu(self.bn1_2(self.conv1_2(c1_pre)))  # (cs, 224, 224)

        c2 = F.relu(self.bn2(self.conv2(c1)))  # (cs*2, 112, 112)
        c3 = F.relu(self.bn3(self.conv3(c2)))  # (cs*4, 56, 56)

        # --- 瓶颈 ---
        bottleneck_out = self.bottleneck(c3)

        # --- 解码 ---
        # 第一次上采样和跳跃连接
        x = self.up4(bottleneck_out)
        x = torch.cat([x, c2], dim=1)  # Concat (U-Net 风格)
        x = F.relu(self.bn4(self.conv4(x)))

        # 第二次上采样和跳跃连接
        x = self.up5(x)
        x = torch.cat([x, c1], dim=1)  # Concat (U-Net 风格)
        x = F.relu(self.bn5(self.conv5(x)))

        # 最终细化
        x = self.conv6(x)

        # 输出头
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
        return "hybrid_grasp_net"


if __name__ == '__main__':
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HybridGraspNet(input_channels=4, channel_size=32, dropout=True, prob=0.5).to(device)
    print(model)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    # 测试前向传播
    with torch.no_grad():
        dummy_input = torch.randn(1, 4, 300, 300).to(device)
        pos_output, cos_output, sin_output, width_output = model(dummy_input)
        print(
            f"输出形状: 位置={pos_output.shape}, 余弦={cos_output.shape}, 正弦={sin_output.shape}, 宽度={width_output.shape}")
