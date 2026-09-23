"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
"""

import os

import torch
import torch.nn as nn

from ..core import register

__all__ = ['GTR', 'GTRSeg']


@register()
class GTR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]
     
    def __init__(self, \
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder

    def forward(self, x, targets=None):
        amp_safe_mode = os.getenv('GTR_AMP_SAFE_MODE', '').lower()
        x = self.backbone(x)
        if self.training and amp_safe_mode in {'encoder_decoder_fp32', 'post_backbone_fp32'}:
            # Keep assignment-sensitive encoder/query-selection/decoder math in FP32;
            # only the backbone runs under the caller's AMP autocast.
            x = _float_tensor_tree(x)
            device_type = _first_tensor_device_type(x)
            with torch.autocast(device_type=device_type, enabled=False):
                x = self.encoder(x)
                x = self.decoder(x, targets)
        else:
            x = self.encoder(x)
            if self.training and amp_safe_mode in {'decoder_fp32', 'fp32_decoder'}:
                # Keep the query selection / decoder heads numerically close to FP32 while
                # still allowing the backbone and encoder to benefit from AMP.
                x = _float_tensor_tree(x)
                device_type = _first_tensor_device_type(x)
                with torch.autocast(device_type=device_type, enabled=False):
                    x = self.decoder(x, targets)
            else:
                x = self.decoder(x, targets)

        return x

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self


@register()
class GTRSeg(nn.Module):
    """Instance segmentation = GTR + per-query mask head.

    Spatial features are taken from encoder level 0 (highest resolution) and
    fed to the SegmentationHead inside the decoder. The seg finetune family
    runs pure FP32 (USE_AMP=0), so the GTR AMP-safe-mode branches are not
    replicated here.
    """
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, \
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder

    def forward(self, x, targets=None):
        x = self.backbone(x)
        x = self.encoder(x)
        spatial_feat = x[0]
        return self.decoder(x, targets, spatial_feat)

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self


def _float_tensor_tree(value):
    if isinstance(value, torch.Tensor):
        return value.float() if value.is_floating_point() else value
    if isinstance(value, list):
        return [_float_tensor_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_float_tensor_tree(v) for v in value)
    if isinstance(value, dict):
        return {k: _float_tensor_tree(v) for k, v in value.items()}
    return value


def _first_tensor_device_type(value):
    if isinstance(value, torch.Tensor):
        return value.device.type
    if isinstance(value, dict):
        iterable = value.values()
    elif isinstance(value, (list, tuple)):
        iterable = value
    else:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    for item in iterable:
        device_type = _first_tensor_device_type(item)
        if device_type:
            return device_type
    return 'cuda' if torch.cuda.is_available() else 'cpu'
