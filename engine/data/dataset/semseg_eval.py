"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Semantic-segmentation evaluator that works in distributed mode. API mirrors
CocoEvaluator (reset/update/synchronize_between_processes/accumulate/summarize)
so evaluate_semseg reads like det_engine.evaluate.
"""

from collections import OrderedDict

import numpy as np
import torch

from ...core import register
from ...misc import dist_utils

__all__ = ['SemSegEvaluator']


@register()
class SemSegEvaluator(object):
    """mIoU / mAcc / aAcc from a confusion matrix accumulated over the val set.

    Confusion matrices are kept PER IMAGE keyed by image_id, and
    synchronize_between_processes deduplicates by id before summing. This makes
    distributed eval exact under DistributedSampler's wrap-around padding (the
    same trap that inflated AP in the COCO distributed-eval bug): duplicated
    images contribute once, and I/U are summed globally BEFORE the division —
    never a mean of per-image IoUs.
    """

    def __init__(self, num_classes, ignore_index=255, class_names=None):
        self.num_classes = int(num_classes)
        self.ignore_index = ignore_index
        self.class_names = class_names
        self.reset()

    def reset(self):
        self._per_image = {}   # image_id -> (K, K) np.int64, rows = GT, cols = pred
        self.mat = None
        self._metrics = None

    # det_solver-family parity (evaluate() calls cleanup on the coco evaluator)
    cleanup = reset

    @torch.no_grad()
    def update(self, res):
        """res: {image_id: (pred (H, W) int tensor, gt (H, W) int tensor)}"""
        K = self.num_classes
        for img_id, (pred, gt) in res.items():
            assert pred.shape == gt.shape, \
                f'pred {tuple(pred.shape)} vs gt {tuple(gt.shape)} for image {img_id}'
            pred = pred.reshape(-1).long()
            gt = gt.reshape(-1).long()
            valid = (gt != self.ignore_index) & (gt < K)
            idx = gt[valid] * K + pred[valid]
            mat = torch.bincount(idx, minlength=K * K).reshape(K, K)
            self._per_image[int(img_id)] = mat.cpu().numpy().astype(np.int64)

    def synchronize_between_processes(self):
        merged = {}
        for part in dist_utils.all_gather(self._per_image):
            for img_id, mat in part.items():
                if img_id not in merged:   # dedup sampler-padding duplicates
                    merged[img_id] = mat
        self._per_image = merged

    def accumulate(self):
        K = self.num_classes
        mat = np.zeros((K, K), dtype=np.int64)
        for m in self._per_image.values():
            mat += m
        self.mat = mat

        matf = mat.astype(np.float64)
        diag = np.diag(matf)
        gt_area = matf.sum(axis=1)                  # per-class GT pixels
        pred_area = matf.sum(axis=0)                # per-class predicted pixels
        union = gt_area + pred_area - diag
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = diag / union                      # NaN for absent classes
            acc = diag / gt_area
        total = matf.sum()
        self._metrics = {
            'iou': iou,
            'acc': acc,
            'miou': float(np.nanmean(iou) * 100),
            'macc': float(np.nanmean(acc) * 100),
            'aacc': float(diag.sum() / total * 100) if total > 0 else float('nan'),
            'num_images': len(self._per_image),
        }

    def summarize(self):
        assert self._metrics is not None, 'call accumulate() before summarize()'
        from tabulate import tabulate
        m = self._metrics
        names = self.class_names if self.class_names else [str(i) for i in range(self.num_classes)]
        table = [
            (name, f'{m["iou"][k] * 100:.2f}', f'{m["acc"][k] * 100:.2f}')
            for k, name in enumerate(names)
        ]
        print(tabulate(table, headers=['class', 'IoU', 'Acc'], tablefmt='pretty'))
        print(f'Semantic seg: images={m["num_images"]}  '
              f'mIoU={m["miou"]:.2f}  mAcc={m["macc"]:.2f}  aAcc={m["aacc"]:.2f}')

    def stats(self):
        """Lists so det_solver's tensorboard loop can enumerate; miou first so
        DetSolver._primary_metric_key selects it for best.pth."""
        m = self._metrics
        return OrderedDict([
            ('miou', [m['miou']]),
            ('macc', [m['macc']]),
            ('aacc', [m['aacc']]),
        ])
