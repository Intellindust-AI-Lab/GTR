"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Depth evaluation loop, protocol from Depth-Anything-V2 metric_depth/train.py:
the input image is keep-aspect resized (GT depth stays native resolution),
predictions are bilinearly resized back to the ground-truth resolution and
metrics are averaged per image over valid pixels. Optionally applies the
ZoeDepth-standard NYU Eigen crop.
"""

import torch
import torch.nn.functional as F

from ..data.dataset.depth_eval import DepthEvaluator, eval_depth
from ..misc import MetricLogger

# ZoeDepth NYU eval crop (zoedepth/utils/misc.py): eval_mask[45:471, 41:601] on 480x640.
_EIGEN_CROP = (45, 471, 41, 601)
_EIGEN_CROP_HW = (480, 640)


def _predict_full(model, samples):
    """Depth prediction covering the full (possibly non-square) frame.

    The backbone's quad-dir bid_scan requires a square patch grid, so a
    keep-aspect-resized landscape frame (NYU: 640x853) is covered with two
    overlapping square crops whose predictions are averaged in the overlap.
    """
    h, w = samples.shape[-2:]
    if h == w:
        return model(samples)['pred_depth']

    assert h < w <= 2 * h, f'evaluation frame {h}x{w} needs more than two square tiles'
    pred_sum = samples.new_zeros(samples.shape[0], h, w)
    count = samples.new_zeros(1, h, w)
    for x0 in (0, w - h):
        pred_sum[..., x0:x0 + h] += model(samples[..., x0:x0 + h])['pred_depth']
        count[..., x0:x0 + h] += 1
    return pred_sum / count


@torch.no_grad()
def evaluate_depth(model: torch.nn.Module, data_loader, evaluator: DepthEvaluator, device):
    model.eval()
    evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        pred = _predict_full(model, samples)                             # [B, h, w]

        for i, target in enumerate(targets):
            gt = target['depth'].as_subclass(torch.Tensor).float().to(device)
            if gt.dim() == 3:                                            # Mask [1, H, W]
                gt = gt.squeeze(0)

            pred_i = F.interpolate(pred[i][None, None], size=gt.shape[-2:],
                                   mode='bilinear', align_corners=True)[0, 0]

            valid = (gt >= evaluator.min_depth) & (gt <= evaluator.max_depth)
            if evaluator.eval_crop == 'eigen':
                assert gt.shape == _EIGEN_CROP_HW, \
                    f'eigen crop is defined for {_EIGEN_CROP_HW} NYU frames, got {tuple(gt.shape)}'
                y0, y1, x0, x1 = _EIGEN_CROP
                crop = torch.zeros_like(valid)
                crop[y0:y1, x0:x1] = True
                valid &= crop
            if valid.sum() < 10:
                continue

            metrics = eval_depth(pred_i[valid], gt[valid])
            evaluator.update(int(target['idx'].item()), metrics)

    metric_logger.synchronize_between_processes()
    evaluator.synchronize_between_processes()
    stats = evaluator.summarize()
    return stats, None
