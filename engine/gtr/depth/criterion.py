"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Scale-invariant log loss for metric depth, from Depth-Anything-V2
(metric_depth/util/loss.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core import register

__all__ = ['GTRDepthCriterion']


@register()
class GTRDepthCriterion(nn.Module):
    __share__ = ['min_depth', 'max_depth']

    def __init__(self, weight_dict={'loss_silog': 1.0}, lambd=0.5, min_depth=0.001, max_depth=10.0):
        super().__init__()
        self.weight_dict = weight_dict
        self.lambd = lambd
        self.min_depth = min_depth
        self.max_depth = max_depth

    def forward(self, outputs, targets, **kwargs):
        pred = outputs['pred_depth']                                       # [B, H, W]
        gt = torch.stack([t['depth'] for t in targets]).as_subclass(torch.Tensor).float()
        if gt.dim() == 4:                                                  # Mask [B, 1, H, W]
            gt = gt.squeeze(1)

        if pred.shape[-2:] != gt.shape[-2:]:
            pred = F.interpolate(pred[:, None], size=gt.shape[-2:],
                                 mode='bilinear', align_corners=True).squeeze(1)

        # Padded / hole pixels fall outside [min_depth, max_depth] and drop out here.
        valid = (gt >= self.min_depth) & (gt <= self.max_depth)
        if not valid.any():
            return {'loss_silog': pred.sum() * 0.0}

        diff_log = torch.log(gt[valid]) - torch.log(pred[valid].clamp_min(1e-6))
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                          self.lambd * torch.pow(diff_log.mean(), 2))

        return {'loss_silog': self.weight_dict.get('loss_silog', 1.0) * loss}
