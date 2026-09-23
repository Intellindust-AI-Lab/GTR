"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Post-processor for oriented boxes: NMS-free top-k selection, outputs pixel-domain
(cx, cy, w, h, theta) with theta in radians.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core import register
from .rbox_ops import PI

__all__ = ['OBBPostProcessor']


def mod(a, b):
    out = a - a // b * b
    return out


@register()
class OBBPostProcessor(nn.Module):
    __share__ = [
        'num_classes',
        'use_focal_loss',
        'num_top_queries',
    ]

    def __init__(
        self,
        num_classes=15,
        use_focal_loss=True,
        num_top_queries=300,
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.deploy_mode = False

    def extra_repr(self) -> str:
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}'

    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']

        # (cx, cy, w, h) scale to pixels (square images assumed); angle to radians
        scale = orig_target_sizes.repeat(1, 2).unsqueeze(1)  # [bs, 1, 4] = (W, H, W, H)
        rbox_pred = torch.cat([boxes[..., :4] * scale, boxes[..., 4:5] * PI], dim=-1)

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            labels = mod(index, self.num_classes)
            query_index = index // self.num_classes
            boxes = rbox_pred.gather(dim=1, index=query_index.unsqueeze(-1).repeat(1, 1, rbox_pred.shape[-1]))
        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            boxes = rbox_pred
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1]))

        if self.deploy_mode:
            return labels, boxes, scores

        results = []
        for lab, box, sco in zip(labels, boxes, scores):
            results.append(dict(labels=lab, boxes=box, scores=sco))

        return results

    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self
