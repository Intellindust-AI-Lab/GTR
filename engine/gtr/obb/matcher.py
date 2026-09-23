"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Hungarian matcher for oriented boxes.

Matching cost follows the paper: focal classification cost + Chamfer distance
cost (corner sets) + KLD cost (gaussian, log1p/tau=1), replacing L1/GIoU.
"""

from typing import Dict

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from ...core import register
from .rbox_ops import chamfer_cost_pairwise, kld_cost_pairwise, rbox_norm_to_rad


@register()
class OBBHungarianMatcher(nn.Module):
    """Computes 1-to-1 assignment between oriented-box predictions and targets."""

    __share__ = [
        "use_focal_loss",
    ]

    def __init__(self, weight_dict, use_focal_loss=True, alpha=0.25, gamma=2.0):
        super().__init__()
        self.cost_class = weight_dict["cost_class"]
        self.cost_chamfer = weight_dict["cost_chamfer"]
        self.cost_kld = weight_dict["cost_kld"]

        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

        assert (
            self.cost_class != 0 or self.cost_chamfer != 0 or self.cost_kld != 0
        ), "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets, group_detr=1):
        """
        Args:
            outputs: dict with "pred_logits" [bs, q, num_classes], "pred_boxes" [bs, q, 5]
                     (sigmoid-domain oriented boxes).
            targets: list of dicts with "labels" [n] and "boxes" [n, 5] (sigmoid-domain).
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        pred_logits = outputs["pred_logits"].flatten(0, 1)
        if self.use_focal_loss:
            out_prob = pred_logits[:, tgt_ids].sigmoid()
            neg_cost_class = (
                (1 - self.alpha) * (out_prob**self.gamma) * (-(1 - out_prob + 1e-8).log())
            )
            pos_cost_class = (
                self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
            )
            cost_class = pos_cost_class - neg_cost_class
        else:
            out_prob = pred_logits.softmax(-1)
            cost_class = -out_prob[:, tgt_ids]

        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [bs*q, 5]

        out_rad = rbox_norm_to_rad(out_bbox)
        tgt_rad = rbox_norm_to_rad(tgt_bbox)
        cost_chamfer = chamfer_cost_pairwise(out_rad, tgt_rad)
        cost_kld = kld_cost_pairwise(out_rad, tgt_rad)

        C = self.cost_chamfer * cost_chamfer + self.cost_class * cost_class + self.cost_kld * cost_kld

        sizes = [len(v["boxes"]) for v in targets]
        g_num_queries = num_queries // group_detr

        C = C.view(bs, num_queries, -1).cpu()
        C = torch.nan_to_num(C, nan=1.0)
        C_groups = C.split(g_num_queries, dim=1)
        indices = None
        for g_i, C_g in enumerate(C_groups):
            indices_g_pre = [linear_sum_assignment(c[i]) for i, c in enumerate(C_g.split(sizes, -1))]
            indices_g = [
                (torch.as_tensor(i + g_i * g_num_queries, dtype=torch.int64),
                 torch.as_tensor(j, dtype=torch.int64))
                for i, j in indices_g_pre
            ]
            if indices is None:
                indices = indices_g
            else:
                indices = [
                    (torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
                    for idx1, idx2 in zip(indices, indices_g)
                ]

        return {"indices": indices}
