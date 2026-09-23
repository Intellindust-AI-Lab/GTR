"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from ...core import register


__all__ = ['PostProcessor']


def mod(a, b):
    out = a - a // b * b
    return out


@register()
class PostProcessor(nn.Module):
    __share__ = [
        'num_classes',
        'use_focal_loss',
        'num_top_queries',
        'remap_mscoco_category'
    ]

    def __init__(
        self,
        num_classes=80,
        use_focal_loss=True,
        num_top_queries=300,
        remap_mscoco_category=False
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category
        self.deploy_mode = False

    def extra_repr(self) -> str:
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}'

    # def forward(self, outputs, orig_target_sizes):
    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        pred_masks = outputs.get('pred_masks', None)

        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            labels = mod(index, self.num_classes)
            query_index = index // self.num_classes
            boxes = bbox_pred.gather(dim=1, index=query_index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))
            if pred_masks is not None:
                B, _, H, W = pred_masks.shape
                m_idx = query_index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
                masks = pred_masks.gather(dim=1, index=m_idx)
            else:
                masks = None
        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1]))
                if pred_masks is not None:
                    B, _, H, W = pred_masks.shape
                    m_idx = index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
                    masks = pred_masks.gather(dim=1, index=m_idx)
                else:
                    masks = None
            else:
                masks = pred_masks

        if self.deploy_mode:
            if masks is not None:
                return labels, boxes, scores, masks
            return labels, boxes, scores

        if self.remap_mscoco_category:
            from ...data.dataset import mscoco_label2category
            labels = torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])\
                .to(boxes.device).reshape(labels.shape)

        results = []
        if masks is not None:
            # Upsample masks to original image size for COCO segm eval.
            for i, (lab, box, sco) in enumerate(zip(labels, boxes, scores)):
                m = masks[i]  # [Q, H, W]
                tH, tW = int(orig_target_sizes[i, 1].item()), int(orig_target_sizes[i, 0].item())
                # Keep channel dim so downstream COCO segm eval can do mask[0, :, :, np.newaxis].
                m_up = F.interpolate(m.unsqueeze(1), size=(tH, tW), mode='bilinear', align_corners=False)  # [Q, 1, tH, tW]
                results.append(dict(labels=lab, boxes=box, scores=sco, masks=m_up.sigmoid()))
        else:
            for lab, box, sco in zip(labels, boxes, scores):
                results.append(dict(labels=lab, boxes=box, scores=sco))

        return results


    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self
