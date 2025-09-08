# -*- coding: utf-8 -*-
# @Time       : 2025/8/9 14:57
# @File       : spdconv.py.py
# @Description:
import torch
import torch.nn as nn


class SPDConv(nn.Module):
    """ space-to-depth → 1x1 conv，用于浅层高频/小目标增强 """

    def __init__(self, in_ch, scale=2, out_ch=None):
        super().__init__()
        self.scale = scale
        out_ch = out_ch or in_ch * (scale * scale)
        self.mix = nn.Conv2d(in_ch * (scale * scale), out_ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        s = self.scale
        # (B, C, H, W) → (B, C*s*s, H/s, W/s)
        x = x.view(B, C, H // s, s, W // s, s).permute(0, 1, 3, 5, 2, 4).reshape(B, C * (s * s), H // s, W // s)
        return self.mix(x)
