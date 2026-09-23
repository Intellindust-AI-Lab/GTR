"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DETRPose: Real-time end-to-end transformer model for multi-person pose estimation
(https://github.com/SebastianJanampa/DETRPose)
"""

from torch import nn

from ...core import register

__all__ = ['GTRPose', ]

@register()
class GTRPose(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder',]

    def __init__(
        self,
        backbone,
        encoder,
        decoder
        ):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self

    def forward(self, samples, targets=None):
        feats = self.backbone(samples)
        feats = self.encoder(feats)
        out = self.decoder(feats, targets, samples if self.training else None)
        return out
