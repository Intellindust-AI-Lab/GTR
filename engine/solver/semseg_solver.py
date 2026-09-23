"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Semantic-segmentation solver: reuses DetSolver.fit()/train_one_epoch unchanged
(every coco-specific branch there is guarded by `coco_evaluator is not None`)
and only swaps the evaluation path for a SemSegEvaluator-based one.
"""

import torch

from ..misc import MetricLogger, dist_utils
from .det_solver import DetSolver


@torch.no_grad()
def slide_inference(model, images, crop_size, stride):
    """mmseg `test_cfg=dict(mode='slide')`: run the model over overlapping windows of
    the native-resolution image and average the logits where windows overlap.

    The window is `eval_spatial_size` — the exact square the model trains on — so the
    backbone never sees a resolution it was not trained at, and the image is never
    squashed. For Cityscapes 2048x1024 with crop 1024 / stride 768 that is 3 windows
    per image (1 row x 3 columns, overlapping by 256 px).
    """
    b, _, h_img, w_img = images.shape
    h_crop, w_crop = crop_size
    h_stride, w_stride = stride
    assert h_img >= h_crop and w_img >= w_crop, (
        f'slide window {crop_size} does not fit in the image {(h_img, w_img)}; '
        f'slide_inference does not pad.')

    h_grids = (h_img - h_crop + h_stride - 1) // h_stride + 1
    w_grids = (w_img - w_crop + w_stride - 1) // w_stride + 1

    logits = None
    count = images.new_zeros((b, 1, h_img, w_img))
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1, x1 = h_idx * h_stride, w_idx * w_stride
            y2, x2 = min(y1 + h_crop, h_img), min(x1 + w_crop, w_img)
            # Shift the last window back inside the image instead of shrinking it:
            # a smaller (or non-square) window would break the backbone's token grid.
            y1, x1 = y2 - h_crop, x2 - w_crop
            crop_logits = model(images[:, :, y1:y2, x1:x2])['pred_sem_seg']
            if logits is None:
                logits = images.new_zeros((b, crop_logits.shape[1], h_img, w_img))
            logits[:, :, y1:y2, x1:x2] += crop_logits
            count[:, :, y1:y2, x1:x2] += 1

    return {'pred_sem_seg': logits / count}


@torch.no_grad()
def evaluate_semseg(model, postprocessor, data_loader, evaluator, device,
                    crop_size=None, slide_stride=None):
    model.eval()
    evaluator.reset()
    if getattr(evaluator, 'class_names', None) is None:
        names = getattr(data_loader.dataset, 'CLASSES', None)
        evaluator.class_names = list(names) if names else None

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        if slide_stride is None:
            outputs = model(samples)
        else:
            outputs = slide_inference(model, samples, crop_size, slide_stride)

        orig_target_sizes = torch.stack([t['orig_size'] for t in targets], dim=0).to(device)
        results = postprocessor(outputs, orig_target_sizes)

        res = {}
        for target, result in zip(targets, results):
            gt = target['seg_map'].to(device)
            gt = gt.squeeze(0) if gt.dim() == 3 else gt   # (1, H, W) Mask -> (H, W)
            res[int(target['image_id'].item())] = (result['sem_seg'], gt)
        evaluator.update(res)

    metric_logger.synchronize_between_processes()
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    if dist_utils.is_main_process():
        evaluator.summarize()

    return evaluator.stats()


class SemSegSolver(DetSolver):

    def _eval_geometry(self):
        """(crop_size, slide_stride) for evaluate_semseg. The slide window is always
        eval_spatial_size: it is the resolution the model trains at and the only shape
        the backbone's square token grid accepts."""
        return self.cfg.yaml_cfg['eval_spatial_size'], self.cfg.slide_stride

    def _evaluate_current_model(self):
        module = self.ema.module if self.ema else self.model
        crop_size, slide_stride = self._eval_geometry()
        test_stats = evaluate_semseg(
            module,
            self.postprocessor,
            self.val_dataloader,
            self.evaluator,
            self.device,
            crop_size=crop_size,
            slide_stride=slide_stride,
        )
        current_stat = self._summarize_eval_stats(self.last_epoch, test_stats)
        # No coco evaluator: DetSolver.fit's eval-dump block is None-guarded.
        return current_stat, test_stats, None

    def val(self, ):
        self.eval()
        module = self.ema.module if self.ema else self.model
        crop_size, slide_stride = self._eval_geometry()
        evaluate_semseg(module, self.postprocessor, self.val_dataloader,
                        self.evaluator, self.device,
                        crop_size=crop_size, slide_stride=slide_stride)
        return
