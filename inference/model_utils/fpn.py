# -*- coding: utf-8 -*-
# @Time       : 2025/8/9 14:56
# @File       : fpn.py.py
# @Description:
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels_list])
        self.out = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list])

    def forward(self, *feats):
        # 兼容 self.fpn(c2,c3,c4) 和 self.fpn([c2,c3,c4])
        if len(feats) == 1 and isinstance(feats[0], (list, tuple)):
            c2, c3, c4 = feats[0]
        else:
            c2, c3, c4 = feats

        p4 = self.lateral[2](c4)
        p3 = self.lateral[1](c3) + F.interpolate(p4, size=c3.shape[-2:], mode='nearest')
        p2 = self.lateral[0](c2) + F.interpolate(p3, size=c2.shape[-2:], mode='nearest')
        p4 = self.out[2](p4)
        p3 = self.out[1](p3)
        p2 = self.out[0](p2)
        return p2, p3, p4
