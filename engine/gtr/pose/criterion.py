"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DETRPose: Real-time end-to-end transformer model for multi-person pose estimation
(https://github.com/SebastianJanampa/DETRPose)
"""

import torch
import torch.nn.functional as F
from torch import nn

from ...core import register
from ...misc.dist_utils import get_world_size, is_dist_available_and_initialized
from .keypoint_loss import OKSLoss
from .utils import sigmoid_focal_loss


@register()
class GTRPoseCriterion(nn.Module):
    __inject__ = ['matcher', ]

    def __init__(self,
        num_classes,
        matcher,
        weight_dict,
        losses,
        num_body_points,
        focal_alpha=0.25,
        mal_alpha=None,
        gamma=2.0,
        ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.mal_alpha = mal_alpha
        self.gamma = gamma
        self.vis = 0.1
        self.abs = 1
        self.num_body_points = num_body_points
        self.oks = OKSLoss(linear=True,
                 num_keypoints=num_body_points,
                 eps=1e-6,
                 reduction='mean',
                 loss_weight=1.0)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        return losses

    def loss_vfl(self, outputs, targets, indices, num_boxes, log=True):
        idx = self._get_src_permutation_idx(indices)
        assert 'pred_keypoints' in outputs

        # Get keypoints
        src_keypoints = outputs['pred_keypoints'][idx] # xyxyvv
        Z_pred = src_keypoints[:, 0:(self.num_body_points * 2)]
        targets_keypoints = torch.cat([t['keypoints'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        targets_area = torch.cat([t['area'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        Z_gt = targets_keypoints[:, 0:(self.num_body_points * 2)]
        V_gt: torch.Tensor = targets_keypoints[:, (self.num_body_points * 2):]
        oks = self.oks(Z_pred,Z_gt,V_gt,targets_area,weight=None,avg_factor=None,reduction_override=None)
        oks = oks.detach()

        # vfl starts here
        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = oks.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.focal_alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    def loss_mal(self, outputs, targets, indices, num_boxes, log=True):
        idx = self._get_src_permutation_idx(indices)
        assert 'pred_keypoints' in outputs

        # Get keypoints
        src_keypoints = outputs['pred_keypoints'][idx] # xyxyvv

        Z_pred = src_keypoints[:, 0:(self.num_body_points * 2)]
        targets_keypoints = torch.cat([t['keypoints'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        targets_area = torch.cat([t['area'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        Z_gt = targets_keypoints[:, 0:(self.num_body_points * 2)]
        V_gt: torch.Tensor = targets_keypoints[:, (self.num_body_points * 2):]
        oks = self.oks(Z_pred,Z_gt,V_gt,targets_area,weight=None,avg_factor=None,reduction_override=None)
        oks = oks.detach()

        # vfl starts here
        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = oks.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        target_score = target_score.pow(self.gamma)
        if self.mal_alpha != None:
            weight = self.mal_alpha * pred_score.pow(self.gamma) * (1 - target) + target
        else:
            weight = pred_score.pow(self.gamma) * (1 - target) + target

        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_mal': loss}

    def loss_local(self, outputs, targets, indices, num_boxes, T=5):
        """Compute Fine-Grained Localization (FGL) Loss
        and Decoupled Distillation Focal (DDF) Loss."""

        losses = {}
        if "pred_corners" in outputs and "teacher_corners" in outputs:
            idx = self._get_src_permutation_idx(indices)
            reg_max = outputs['reg_max']

            # compute oks
            src_keypoints = outputs['pred_keypoints'][idx] # xyxyvv

            Z_pred = src_keypoints[:, 0:(self.num_body_points * 2)]
            targets_keypoints = torch.cat([t['keypoints'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            targets_area = torch.cat([t['area'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            Z_gt = targets_keypoints[:, 0:(self.num_body_points * 2)]
            V_gt = targets_keypoints[:, (self.num_body_points * 2):]

            oks = self.oks(Z_pred,Z_gt,V_gt,targets_area,weight=None,avg_factor=None,reduction_override=None)

            # The ddf loss starts here
            pred_corners = outputs["pred_corners"].reshape(-1, (reg_max + 1))
            target_corners = outputs["teacher_corners"].reshape(-1, (reg_max + 1))
            if torch.equal(pred_corners, target_corners):
                losses["loss_dfl"] = pred_corners.sum() * 0
            else:
                weight_targets_local = outputs["teacher_logits"].sigmoid().max(dim=-1)[0]

                mask = torch.zeros_like(weight_targets_local, dtype=torch.bool)
                mask[idx] = True
                mask = mask.unsqueeze(-1).repeat(1, 1, (self.num_body_points*2)).reshape(-1)
                weight_targets_local[idx] = oks.reshape_as(weight_targets_local[idx]).to(
                    weight_targets_local.dtype
                )
                weight_targets_local = (
                    weight_targets_local.unsqueeze(-1).repeat(1, 1, (self.num_body_points*2)).reshape(-1).detach()
                )
                loss_match_local = (
                    weight_targets_local
                    * (T**2)
                    * (
                        nn.KLDivLoss(reduction="none")(
                            F.log_softmax(pred_corners / T, dim=1),
                            F.softmax(target_corners.detach() / T, dim=1),
                        )
                    ).sum(-1)
                )
                if "is_dn" not in outputs:
                    batch_scale = (
                        8 / outputs["pred_keypoints"].shape[0]
                    )  # Avoid the influence of batch size per GPU
                    self.num_pos, self.num_neg = (
                        (mask.sum() * batch_scale) ** 0.5,
                        ((~mask).sum() * batch_scale) ** 0.5,
                    )
                loss_match_local1 = loss_match_local[mask].mean() if mask.any() else 0
                loss_match_local2 = loss_match_local[~mask].mean() if (~mask).any() else 0
                losses["loss_dfl"] = (
                    loss_match_local1 * self.num_pos + loss_match_local2 * self.num_neg
                ) / (self.num_pos + self.num_neg)

        return losses

    def loss_keypoints(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_keypoints = outputs['pred_keypoints'][idx] # xyxyvv

        if len(src_keypoints) == 0:
            device = outputs["pred_logits"].device
            losses = {
                'loss_keypoints': torch.as_tensor(0., device=device)+src_keypoints.sum()*0,
                'loss_oks': torch.as_tensor(0., device=device)+src_keypoints.sum()*0,
            }
            return losses
        Z_pred = src_keypoints[:, 0:(self.num_body_points * 2)]
        targets_keypoints = torch.cat([t['keypoints'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        targets_area = torch.cat([t['area'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        Z_gt = targets_keypoints[:, 0:(self.num_body_points * 2)]
        V_gt: torch.Tensor = targets_keypoints[:, (self.num_body_points * 2):]
        oks_loss = 1- self.oks(Z_pred,Z_gt,V_gt,targets_area,weight=None,avg_factor=None,reduction_override=None)
        pose_loss = F.l1_loss(Z_pred, Z_gt, reduction='none')
        pose_loss = pose_loss * V_gt.repeat_interleave(2, dim=1)
        losses = {}
        losses['loss_keypoints'] = pose_loss.sum() / num_boxes
        losses['loss_oks'] = oks_loss.sum() / num_boxes
        return losses

    @torch.no_grad()
    def loss_matching_cost(self, outputs, targets, indices, num_boxes):
        cost_mean_dict = indices[1]
        losses = {"set_{}".format(k):v for k,v in cost_mean_dict.items()}
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _get_go_indices(self, indices, indices_aux_list):
        """Get a matching union set across all decoder layers."""
        results = []
        for indices_aux in indices_aux_list:
            indices = [
                (torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
                for idx1, idx2 in zip(indices.copy(), indices_aux.copy())
            ]

        for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:
            unique, counts = torch.unique(ind, return_counts=True, dim=0)
            count_sort_indices = torch.argsort(counts, descending=True)
            unique_sorted = unique[count_sort_indices]
            column_to_row = {}
            for idx in unique_sorted:
                row_idx, col_idx = idx[0].item(), idx[1].item()
                if row_idx not in column_to_row:
                    column_to_row[row_idx] = col_idx
            final_rows = torch.tensor(list(column_to_row.keys()), device=ind.device)
            final_cols = torch.tensor(list(column_to_row.values()), device=ind.device)
            results.append((final_rows.long(), final_cols.long()))
        return results

    def _clear_cache(self):
        self.num_pos, self.num_neg = None, None

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            "keypoints":self.loss_keypoints,
            "matching": self.loss_matching_cost,
            "vfl": self.loss_vfl,
            "mal": self.loss_mal,
            "local": self.loss_local
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}
        device=next(iter(outputs.values())).device

        # loss for final layer
        indices = self.matcher(outputs_without_aux, targets)

        self._clear_cache()

        # Get the matching union set across all decoder layers.
        if "aux_outputs" in outputs:
            indices_aux_list, cached_indices, cached_indices_enc = [], [], []
            for i, aux_outputs in enumerate(outputs["aux_outputs"] + [outputs["aux_pre_outputs"]]):
                indices_aux = self.matcher(aux_outputs, targets)
                cached_indices.append(indices_aux)
                indices_aux_list.append(indices_aux)
            for i, aux_outputs in enumerate(outputs["aux_interm_outputs"]):
                indices_enc = self.matcher(aux_outputs, targets)
                cached_indices_enc.append(indices_enc)
                indices_aux_list.append(indices_enc)
            indices_go = self._get_go_indices(indices, indices_aux_list)

            num_boxes_go = sum(len(x[0]) for x in indices_go)
            num_boxes_go = torch.as_tensor(
                [num_boxes_go], dtype=torch.float, device=next(iter(outputs.values())).device
            )
            if is_dist_available_and_initialized():
                torch.distributed.all_reduce(num_boxes_go)
            num_boxes_go = torch.clamp(num_boxes_go / get_world_size(), min=1).item()
        else:
            assert "aux_outputs" in outputs, ""

       # Compute the average number of target keypoints accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            indices_in = indices_go if loss in ["keypoints", "local"] else indices
            num_boxes_in = num_boxes_go if loss in ["keypoints", "local"] else num_boxes
            l_dict = self.get_loss(loss, outputs, targets, indices_in, num_boxes_in)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                aux_outputs["up"], aux_outputs["reg_scale"], aux_outputs["reg_max"] = outputs["up"], outputs["reg_scale"], outputs["reg_max"]
                for loss in self.losses:
                    indices_in = indices_go if loss in ["keypoints", "local"] else cached_indices[i]
                    num_boxes_in = num_boxes_go if loss in ["keypoints", "local"] else num_boxes
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices_in, num_boxes_in
                    )
                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict
                    }
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of auxiliary traditional head output at first decoder layer.
        if "aux_pre_outputs" in outputs:
            aux_outputs = outputs["aux_pre_outputs"]
            for loss in self.losses:
                indices_in = indices_go if loss in ["keypoints", "local"] else cached_indices[-1]
                num_boxes_in = num_boxes_go if loss in ["keypoints", "local"] else num_boxes
                l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in)

                l_dict = {
                    k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict
                }
                l_dict = {k + "_pre": v for k, v in l_dict.items()}
                losses.update(l_dict)

        # In case of encoder auxiliary losses.
        if "aux_interm_outputs" in outputs:
            enc_targets = targets
            for i, aux_outputs in enumerate(outputs["aux_interm_outputs"]):
                for loss in self.losses:
                    indices_in = indices_go if loss == "keypoints" else cached_indices_enc[i]
                    num_boxes_in = num_boxes_go if loss == "keypoints" else num_boxes
                    l_dict = self.get_loss(
                        loss, aux_outputs, enc_targets, indices_in, num_boxes_in
                    )
                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict
                    }
                    l_dict = {k + f"_enc_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of cdn auxiliary losses. For dfine
        if "dn_aux_outputs" in outputs:
            assert "dn_meta" in outputs, ""
            dn_meta = outputs['dn_meta']
            single_pad, scalar = self.prep_for_dn(dn_meta)
            dn_pos_idx = []
            dn_neg_idx = []
            for i in range(len(targets)):
                if len(targets[i]['labels']) > 0:
                    t = torch.arange(len(targets[i]['labels']), device=device).long()
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(scalar), device=device) * single_pad).long().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([], device=device).long()

                dn_pos_idx.append((output_idx, tgt_idx))
                dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

            dn_outputs = outputs['dn_aux_outputs']

            for i, aux_outputs in enumerate(outputs["dn_aux_outputs"]):
                aux_outputs["is_dn"] = True
                aux_outputs["up"], aux_outputs["reg_scale"], aux_outputs["reg_max"] = outputs["up"], outputs["reg_scale"], outputs["reg_max"]
                for loss in self.losses:
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, dn_pos_idx, num_boxes*scalar
                    )
                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict
                    }
                    l_dict = {k + f"_dn_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

            # In case of auxiliary traditional head output at first decoder layer.
            if "dn_aux_pre_outputs" in outputs:
                aux_outputs = outputs["dn_aux_pre_outputs"]
                for loss in self.losses:
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, dn_pos_idx, num_boxes*scalar
                    )
                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict
                    }
                    l_dict = {k + "_dn_pre": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        losses = {k: v for k, v in sorted(losses.items(), key=lambda item: item[0])}
        return losses

    def prep_for_dn(self,dn_meta):
        num_dn_groups,pad_size=dn_meta['num_dn_group'],dn_meta['pad_size']
        assert pad_size % num_dn_groups==0
        single_pad=pad_size//num_dn_groups

        return single_pad,num_dn_groups
