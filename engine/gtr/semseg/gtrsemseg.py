"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Semantic segmentation on the GTR backbone/encoder: GTRSemSeg replaces the whole
DETR decoder stack with a dense per-pixel head on the encoder's highest-resolution
level. No queries, no matcher — plain cross-entropy with ignore_index.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core import register

__all__ = ['GTRSemSeg', 'SemSegHead', 'SemSegCriterion', 'SemSegPostProcessor']


@register()
class SemSegHead(nn.Module):
    """FCN-style head on one feature map: conv3x3-BN-ReLU -> dropout -> conv1x1.

    BatchNorm is converted to SyncBN by dist_utils.warp_model (sync_bn: True in
    the base config), matching the mmseg FCNHead convention.
    """

    def __init__(self, in_channels=256, hidden_dim=256, num_classes=19, dropout=0.1):
        super().__init__()
        # Named (not Sequential-indexed) so the BN params are `head.bn.*`: the
        # optimizer's no-decay regex matches on 'norm|bn|bias' substrings, and an
        # anonymous `conv.1.weight` would silently take weight_decay.
        self.conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, feat):
        return self.classifier(self.dropout(self.act(self.bn(self.conv(feat)))))


@register()
class GTRSemSeg(nn.Module):
    """backbone -> encoder -> SemSegHead(encoder level 0, the stride-8 level) ->
    bilinear upsample to input resolution. Mirrors GTRSeg's `spatial_feat = x[0]`
    choice; the loss runs at full input resolution, not on downsampled labels.
    """
    __inject__ = ['backbone', 'encoder', 'head', ]

    def __init__(self, \
        backbone: nn.Module,
        encoder: nn.Module,
        head: nn.Module,
    ):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.head = head

    def forward(self, x, targets=None):
        size = x.shape[-2:]
        feats = self.encoder(self.backbone(x))
        logits = self.head(feats[0])
        logits = F.interpolate(logits, size=size, mode='bilinear', align_corners=False)
        return {'pred_sem_seg': logits}

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self


@register()
class SemSegCriterion(nn.Module):
    """Per-pixel cross-entropy. `losses`/`weight_dict` keep the GTRCriterion
    config shape so det_solver's loss logging needs no special case."""

    def __init__(self, losses=['ce'], weight_dict={'loss_ce': 1.0}, ignore_index=255):
        super().__init__()
        assert list(losses) == ['ce'], f'Only the ce loss is implemented, got {losses}'
        self.losses = list(losses)
        self.weight_dict = dict(weight_dict)
        self.ignore_index = ignore_index

    def forward(self, outputs, targets, **kwargs):
        logits = outputs['pred_sem_seg']                       # (B, K, H, W)
        seg = torch.stack([t['seg_map'] for t in targets])     # (B, 1, H, W)
        seg = seg.squeeze(1).long().as_subclass(torch.Tensor)  # (B, H, W)
        # sum/count instead of mean-reduction: an all-ignore batch yields 0/1 = 0
        # instead of NaN, without a CPU-GPU sync point in the hot loop.
        loss_sum = F.cross_entropy(
            logits, seg, ignore_index=self.ignore_index, reduction='sum')
        n_valid = (seg != self.ignore_index).sum().clamp_(min=1)
        return {'loss_ce': self.weight_dict['loss_ce'] * loss_sum / n_valid}


@register()
class SemSegPostProcessor(nn.Module):
    """Resize logits back to each image's native resolution, then argmax.

    `orig_target_sizes` follows the det convention [w, h] per row. Per-image loop
    keeps peak memory at one (K, H_orig, W_orig) float map at a time.
    """

    def __init__(self, ):
        super().__init__()

    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        logits = outputs['pred_sem_seg']   # (B, K, h, w)
        results = []
        for i in range(logits.shape[0]):
            w, h = orig_target_sizes[i].tolist()
            per_img = logits[i:i + 1].float()
            # Already native-resolution under sliding-window eval: skip the no-op resize.
            if per_img.shape[-2:] != (int(h), int(w)):
                per_img = F.interpolate(
                    per_img, size=(int(h), int(w)),
                    mode='bilinear', align_corners=False)
            results.append({'sem_seg': per_img.argmax(dim=1).squeeze(0).to(torch.uint8)})
        return results

    def deploy(self, ):
        self.eval()
        return self
