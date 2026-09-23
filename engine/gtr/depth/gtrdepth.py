"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
DPT-style monocular metric depth head, ported from Depth-Anything-V2
(https://github.com/DepthAnything/Depth-Anything-V2, metric_depth/depth_anything_v2/dpt.py)
and adapted to the 3-level GTR encoder pyramid (strides 8/16/32).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core import register

__all__ = ['GTRDepth', 'DPTDepthHead']


class ResidualConvUnit(nn.Module):
    def __init__(self, features, use_bn=False):
        super().__init__()
        self.use_bn = use_bn
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        if use_bn:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)
        self.activation = nn.ReLU(False)

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        if self.use_bn:
            out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.use_bn:
            out = self.bn2(out)
        return out + x


class FeatureFusionBlock(nn.Module):
    """DPT RefineNet-style fusion: add skip branch, refine, 2x (or given-size) upsample."""

    def __init__(self, features, use_bn=False, align_corners=True):
        super().__init__()
        self.align_corners = align_corners
        self.out_conv = nn.Conv2d(features, features, kernel_size=1, stride=1, padding=0, bias=True)
        self.res_conv_unit1 = ResidualConvUnit(features, use_bn)
        self.res_conv_unit2 = ResidualConvUnit(features, use_bn)

    def forward(self, *xs, size=None):
        output = xs[0]
        if len(xs) == 2:
            output = output + self.res_conv_unit1(xs[1])
        output = self.res_conv_unit2(output)

        modifier = {'scale_factor': 2} if size is None else {'size': size}
        output = F.interpolate(output, **modifier, mode='bilinear', align_corners=self.align_corners)
        return self.out_conv(output)


@register()
class DPTDepthHead(nn.Module):
    """Fuses the encoder pyramid [stride 8, 16, 32] into a dense metric depth map.

    Mirrors the DA-V2 DPTHead with 3 pyramid levels instead of 4: per-level 3x3
    projection, top-down RefineNet fusion (each step upsamples 2x, ending at
    stride 4), then the two output convs with a Sigmoid bounded to max_depth.

    With log_depth=True the bounded Sigmoid decode is replaced by the YOLO26
    log-depth head (ultralytics nn/modules/head.py Depth): the final conv emits
    a raw logit and depth = exp(clamp(logit, -4, 5)), an unbounded ~0.02-148 m
    range — max_depth is no longer baked into the architecture (it stays a
    loss/eval valid mask only).
    """

    __share__ = ['max_depth']

    def __init__(self, in_channels=[256, 256, 256], features=128, use_bn=False, max_depth=10.0,
                 log_depth=False):
        super().__init__()
        assert len(in_channels) == 3, f'DPTDepthHead expects 3 pyramid levels, got {len(in_channels)}'
        self.max_depth = max_depth
        self.log_depth = log_depth

        self.layer_rn = nn.ModuleList([
            nn.Conv2d(ch, features, kernel_size=3, stride=1, padding=1, bias=False)
            for ch in in_channels
        ])

        self.refinenet3 = FeatureFusionBlock(features, use_bn)   # stride 32 -> 16
        self.refinenet2 = FeatureFusionBlock(features, use_bn)   # stride 16 -> 8
        self.refinenet1 = FeatureFusionBlock(features, use_bn)   # stride 8  -> 4

        self.output_conv1 = nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1)
        out_layers = [
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        ]
        if log_depth:
            # exp(0.182) ~ 1.2 m so early exp() outputs stay well-conditioned (YOLO26 init).
            out_layers[-1].bias.data.fill_(0.182)
        else:
            out_layers.append(nn.Sigmoid())
        self.output_conv2 = nn.Sequential(*out_layers)

    def forward(self, feats, out_hw):
        # feats ordered highest resolution first: [stride8, stride16, stride32]
        layer_1, layer_2, layer_3 = [rn(f) for rn, f in zip(self.layer_rn, feats)]

        path_3 = self.refinenet3(layer_3, size=layer_2.shape[2:])
        path_2 = self.refinenet2(path_3, layer_2, size=layer_1.shape[2:])
        path_1 = self.refinenet1(path_2, layer_1)

        out = self.output_conv1(path_1)
        out = F.interpolate(out, size=out_hw, mode='bilinear', align_corners=True)
        out = self.output_conv2(out)                              # [B, 1, H, W]

        if self.log_depth:
            return torch.exp(out.clamp(-4.0, 5.0)).squeeze(1)     # ~0.02-148 m, unbounded decode
        return out.squeeze(1) * self.max_depth                    # Sigmoid output in (0, 1)


@register()
class GTRDepth(nn.Module):
    """Metric depth estimation = GTR backbone + encoder + DPT-style depth head.

    The depth family runs pure FP32 (use_amp: False), so the GTR AMP-safe-mode
    branches are not replicated here.
    """

    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, \
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, targets=None):
        out_hw = x.shape[-2:]
        feats = self.encoder(self.backbone(x))
        return {'pred_depth': self.decoder(feats, out_hw)}

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
