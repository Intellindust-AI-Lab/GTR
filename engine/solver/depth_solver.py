"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
"""

from .det_solver import DetSolver
from .depth_engine import evaluate_depth


class DepthSolver(DetSolver):
    """Reuses the DetSolver training loop; only evaluation differs.

    evaluate_depth returns stats ordered with d1 first, so the inherited
    best-checkpoint selection (first non-epoch key, higher is better) picks d1.
    """

    def _evaluate_current_model(self):
        module = self.ema.module if self.ema else self.model
        test_stats, _ = evaluate_depth(module, self.val_dataloader, self.evaluator, self.device)
        current_stat = self._summarize_eval_stats(self.last_epoch, test_stats)
        return current_stat, test_stats, None

    def val(self, ):
        self.eval()
        module = self.ema.module if self.ema else self.model
        evaluate_depth(module, self.val_dataloader, self.evaluator, self.device)
        return
