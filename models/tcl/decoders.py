# ------------------------------------------------------------------------------
# CoDe
# Copyright (C) 2024 by Ji-Jia Wu. All Rights Reserved.
# ------------------------------------------------------------------------------
# Modified from TCL (https://github.com/kakaobrain/tcl)
# Copyright (c) 2023 Kakao Brain. All Rights Reserved.
# ------------------------------------------------------------------------------
import torch.nn as nn

from models.builder import MODELS
from models.tcl.modules import ResConv

from clip.model import Transformer


@MODELS.register_module()
class GDecoder(nn.Module):
    def __init__(self, C, kernel_size, norm, act, double, n_layers=2, **kwargs):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.append(
                ResConv(
                    C, C,
                    kernel_size=kernel_size,
                    padding=kernel_size//2,
                    upsample=True,
                    norm=norm,
                    activ=act,
                    double=double,
                    gate=True
                )
            )

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TextDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = Transformer(
            width=512,
            layers=2,
            heads=8,
        )

    def forward(self, middle_features):
        return self.transformer(middle_features[-1])
