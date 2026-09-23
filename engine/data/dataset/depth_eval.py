"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Depth metrics follow Depth-Anything-V2 metric_depth/util/metric.py.
"""

from collections import OrderedDict

import torch

from ...core import register
from ...misc import dist_utils

__all__ = ['DepthEvaluator', 'eval_depth']

# d1/d2/d3 are maximized, the rest minimized. d1 comes first: DetSolver picks the
# first key of the summary as the best-checkpoint metric (higher is better).
DEPTH_METRIC_KEYS = ('d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'log10', 'silog')


def eval_depth(pred, target):
    """Per-image metrics over valid pixels only (1D tensors of equal length)."""
    assert pred.shape == target.shape

    thresh = torch.max((target / pred), (pred / target))

    d1 = torch.sum(thresh < 1.25).float() / len(thresh)
    d2 = torch.sum(thresh < 1.25 ** 2).float() / len(thresh)
    d3 = torch.sum(thresh < 1.25 ** 3).float() / len(thresh)

    diff = pred - target
    diff_log = torch.log(pred) - torch.log(target)

    abs_rel = torch.mean(torch.abs(diff) / target)
    sq_rel = torch.mean(torch.pow(diff, 2) / target)

    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log, 2)))

    log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))
    silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))

    return {'d1': d1.item(), 'd2': d2.item(), 'd3': d3.item(), 'abs_rel': abs_rel.item(),
            'sq_rel': sq_rel.item(), 'rmse': rmse.item(), 'rmse_log': rmse_log.item(),
            'log10': log10.item(), 'silog': silog.item()}


@register()
class DepthEvaluator(object):
    """Accumulates per-image depth metrics keyed by sample index.

    Keying by index makes the distributed merge exact: DistributedSampler pads
    ranks with duplicated samples, which would otherwise skew the averages.
    """

    __share__ = ['min_depth', 'max_depth']

    def __init__(self, min_depth=0.001, max_depth=10.0, eval_crop=None):
        assert eval_crop in (None, 'eigen'), f'unsupported eval_crop: {eval_crop}'
        self.min_depth = min_depth
        self.max_depth = max_depth
        # 'eigen': ZoeDepth-standard NYU eval crop [45:471, 41:601] on 480x640 GT.
        self.eval_crop = eval_crop
        self.records = {}

    def cleanup(self):
        self.records = {}

    def update(self, index: int, metrics: dict):
        self.records[int(index)] = metrics

    def synchronize_between_processes(self):
        merged = {}
        for rank_records in dist_utils.all_gather(self.records):
            merged.update(rank_records)
        self.records = merged

    def summarize(self):
        """Mean of per-image metrics; values wrapped in 1-element lists to match
        the test_stats format DetSolver.fit expects."""
        if not self.records:
            print('DepthEvaluator: no valid samples evaluated.')
            return OrderedDict((k, [float('nan')]) for k in DEPTH_METRIC_KEYS)

        stats = OrderedDict()
        n = len(self.records)
        for key in DEPTH_METRIC_KEYS:
            stats[key] = [sum(rec[key] for rec in self.records.values()) / n]

        if dist_utils.is_main_process():
            header = ', '.join(f'{k:>8}' for k in DEPTH_METRIC_KEYS)
            values = ', '.join(f'{stats[k][0]:8.3f}' for k in DEPTH_METRIC_KEYS)
            print('=' * 90)
            print(f'Depth eval on {n} images (min_depth={self.min_depth}, '
                  f'max_depth={self.max_depth}, eval_crop={self.eval_crop})')
            print(header)
            print(values)
            print('=' * 90)
        return stats
