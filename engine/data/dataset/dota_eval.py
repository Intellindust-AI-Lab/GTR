"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
DOTA evaluator: VOC07 11-point AP at rotated IoU 0.5, exact convex polygon IoU.

Duck-types the CocoEvaluator surface used by det_engine/det_solver:
cleanup / update / synchronize_between_processes / accumulate / summarize,
iou_types, labels, and coco_eval['bbox'].stats / .eval.
"""

import math
from types import SimpleNamespace

import numpy as np
import torch

from ...core import register
from ...misc import dist_utils
from .dota_dataset import DOTA_CLASSES

__all__ = ['DOTAEvaluator', 'rbox_iou_matrix', 'voc_eval_rbox', 'rbox_to_poly_np']


def rbox_to_poly_np(rboxes: np.ndarray) -> np.ndarray:
    """(N, 5) pixel rboxes (theta radians) -> (N, 4, 2) corner polygons."""
    cx, cy, w, h, t = [rboxes[:, i] for i in range(5)]
    cos_t, sin_t = np.cos(t), np.sin(t)
    dw = np.stack([cos_t, sin_t], -1) * w[:, None] * 0.5
    dh = np.stack([-sin_t, cos_t], -1) * h[:, None] * 0.5
    c = np.stack([cx, cy], -1)
    return np.stack([c + dw + dh, c + dw - dh, c - dw - dh, c - dw + dh], axis=1)


def _poly_area(polys: np.ndarray) -> np.ndarray:
    """Shoelace area, (..., K, 2) -> (...); vertices in consistent winding."""
    x, y = polys[..., 0], polys[..., 1]
    x2, y2 = np.roll(x, -1, axis=-1), np.roll(y, -1, axis=-1)
    return 0.5 * np.abs((x * y2 - x2 * y).sum(-1))


def _convex_inter_area(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Intersection area of paired convex quads p1, p2: (B, 4, 2) x (B, 4, 2) -> (B,).

    Candidate-point method: vertices of one quad inside the other plus all edge
    intersections, sorted by polar angle around their centroid (convex result).
    """
    B = p1.shape[0]
    if B == 0:
        return np.zeros((0,), dtype=np.float64)

    def edges(p):
        return p, np.roll(p, -1, axis=1)   # (B,4,2), (B,4,2)

    def inside(pts, poly):
        # pts (B,P,2) inside convex poly (B,4,2) (any winding): all cross same sign
        a, b = edges(poly)
        d = b - a                                             # (B,4,2)
        v = pts[:, :, None, :] - a[:, None, :, :]             # (B,P,4,2)
        cross = d[:, None, :, 0] * v[..., 1] - d[:, None, :, 1] * v[..., 0]
        return (cross >= -1e-9).all(-1) | (cross <= 1e-9).all(-1)   # (B,P)

    # 1) vertices inside the other quad
    in1 = inside(p1, p2)   # (B,4)
    in2 = inside(p2, p1)

    # 2) all 4x4 edge-pair intersections
    a1, b1 = edges(p1)
    a2, b2 = edges(p2)
    A = a1[:, :, None, :]          # (B,4,1,2)
    r = (b1 - a1)[:, :, None, :]
    C = a2[:, None, :, :]
    s = (b2 - a2)[:, None, :, :]
    denom = r[..., 0] * s[..., 1] - r[..., 1] * s[..., 0]     # (B,4,4)
    diff = C - A
    t_num = diff[..., 0] * s[..., 1] - diff[..., 1] * s[..., 0]
    u_num = diff[..., 0] * r[..., 1] - diff[..., 1] * r[..., 0]
    with np.errstate(divide='ignore', invalid='ignore'):
        t = t_num / denom
        u = u_num / denom
        hit = (np.abs(denom) > 1e-12) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
        inter_pts = A + t[..., None] * r                      # (B,4,4,2)

    # gather candidates: 4 + 4 + 16 = 24
    cand = np.concatenate([p1, p2, inter_pts.reshape(B, 16, 2)], axis=1)      # (B,24,2)
    valid = np.concatenate([in1, in2, hit.reshape(B, 16)], axis=1)            # (B,24)
    # degenerate (parallel-edge) intersections produce nan/inf coords; they are masked
    # invalid but nan*0 would still poison the centroid sum below — zero them out.
    cand = np.where(valid[..., None], cand, 0.0)

    cnt = valid.sum(-1)
    has_poly = cnt >= 3
    area = np.zeros(B, dtype=np.float64)
    if not has_poly.any():
        return area

    cand = cand[has_poly]
    valid = valid[has_poly]
    # centroid of valid candidates
    vf = valid[..., None].astype(np.float64)
    centroid = (cand * vf).sum(1) / np.maximum(vf.sum(1), 1)
    ang = np.arctan2(cand[..., 1] - centroid[:, None, 1], cand[..., 0] - centroid[:, None, 0])
    ang = np.where(valid, ang, np.inf)                        # invalid points sort last
    order = np.argsort(ang, axis=1)
    cand = np.take_along_axis(cand, order[..., None], axis=1)
    valid_sorted = np.take_along_axis(valid, order, axis=1)
    # replace invalid tail with the first (valid) vertex: duplicates add no area
    first = cand[:, :1, :]
    cand = np.where(valid_sorted[..., None], cand, first)
    area[has_poly] = _poly_area(cand)
    return area


def rbox_iou_matrix(rboxes1: np.ndarray, rboxes2: np.ndarray, chunk=200000) -> np.ndarray:
    """Exact rotated IoU matrix: (N, 5) x (M, 5) pixel rboxes -> (N, M)."""
    N, M = len(rboxes1), len(rboxes2)
    if N == 0 or M == 0:
        return np.zeros((N, M), dtype=np.float64)
    poly1 = rbox_to_poly_np(rboxes1.astype(np.float64))
    poly2 = rbox_to_poly_np(rboxes2.astype(np.float64))
    area1 = (rboxes1[:, 2] * rboxes1[:, 3]).astype(np.float64)
    area2 = (rboxes2[:, 2] * rboxes2[:, 3]).astype(np.float64)

    idx1, idx2 = np.meshgrid(np.arange(N), np.arange(M), indexing='ij')
    idx1, idx2 = idx1.ravel(), idx2.ravel()
    inter = np.zeros(N * M, dtype=np.float64)
    for s in range(0, N * M, chunk):
        e = min(s + chunk, N * M)
        inter[s:e] = _convex_inter_area(poly1[idx1[s:e]], poly2[idx2[s:e]])
    union = area1[idx1] + area2[idx2] - inter
    iou = np.where(union > 0, inter / union, 0.0)
    return iou.reshape(N, M)


def _voc_ap_11point(rec: np.ndarray, prec: np.ndarray) -> float:
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        p = prec[rec >= t].max() if (rec >= t).any() else 0.0
        ap += p / 11.0
    return float(ap)


def voc_eval_rbox(predictions: dict, gt: dict, num_classes: int, iou_thr: float = 0.5,
                  ignore: dict = None) -> np.ndarray:
    """VOC07 11-point AP per class over rotated boxes.

    Args:
        predictions: {img_id: {'labels': (N,), 'boxes': (N, 5) px/rad, 'scores': (N,)}}
        gt: {img_id: {class_id: (M, 5) px/rad rboxes}}
        ignore: optional {img_id: {class_id: (M,) bool}} aligned with `gt`, marking
            VOC-style ignore instances: left out of the recall denominator, and a
            detection whose best match is one counts as neither TP nor FP.

    Returns:
        (num_classes,) AP array.
    """
    def ign_of(img_id, c, n):
        if ignore is None:
            return np.zeros(n, dtype=bool)
        return ignore.get(img_id, {}).get(c, np.zeros(n, dtype=bool))

    per_class_ap = np.zeros(num_classes)
    for c in range(num_classes):
        img_ids, scores, boxes = [], [], []
        for img_id, pred in predictions.items():
            m = pred['labels'] == c
            if m.any():
                img_ids.append(np.full(int(m.sum()), img_id, dtype=object))
                scores.append(pred['scores'][m])
                boxes.append(pred['boxes'][m])
        npos = sum(int((~ign_of(i, c, len(g.get(c, ())))).sum()) for i, g in gt.items())
        if not img_ids:
            per_class_ap[c] = 0.0
            continue
        img_ids = np.concatenate(img_ids)
        scores = np.concatenate(scores)
        boxes = np.concatenate(boxes)

        order = np.argsort(-scores)
        img_ids, boxes = img_ids[order], boxes[order]

        matched = {img_id: np.zeros(len(g.get(c, ())), dtype=bool) for img_id, g in gt.items()}
        tp = np.zeros(len(boxes))
        fp = np.zeros(len(boxes))
        iou_cache = {}
        for i, (img_id, box) in enumerate(zip(img_ids, boxes)):
            gts = gt.get(img_id, {}).get(c, np.zeros((0, 5)))
            if len(gts) == 0:
                fp[i] = 1
                continue
            if img_id not in iou_cache:
                det_mask = img_ids == img_id
                ious_all = rbox_iou_matrix(boxes[det_mask], gts)
                iou_cache[img_id] = dict(zip(np.nonzero(det_mask)[0].tolist(), ious_all))
            ious = iou_cache[img_id][i]
            j = int(ious.argmax())
            if ious[j] < iou_thr:
                fp[i] = 1
            elif ign_of(img_id, c, len(gts))[j]:
                pass          # ignore instance: neither TP nor FP, as in the VOC devkit
            elif not matched[img_id][j]:
                matched[img_id][j] = True
                tp[i] = 1
            else:
                fp[i] = 1

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        rec = tp_cum / max(npos, 1)
        prec = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float64).eps)
        per_class_ap[c] = _voc_ap_11point(rec, prec)
    return per_class_ap


@register()
class DOTAEvaluator(object):
    """Accumulates rotated-box predictions and computes VOC07 mAP@0.5 (rotated IoU)."""

    def __init__(self, dataset, iou_thr=0.5, difficulty=100, verbose=True):
        """
        difficulty: DOTA annotations carry a per-instance `difficult` flag (9.7% of the
            1024px val patches used here). Instances with ``difficult > difficulty`` are
            treated as VOC-style ignore regions: excluded from the recall denominator, and
            a detection matching one counts as neither TP nor FP.

            The default 100 keeps every instance as a normal positive, which is the
            mmrotate convention (``DOTADataset(difficulty=100)``) and the one all GTR
            numbers were produced with -- lowering it is NOT comparable to them. Set
            ``difficulty: 0`` to reproduce the official DOTA devkit convention instead,
            which ignores difficult instances. Training always uses every instance
            regardless of this setting.
        """
        self.dataset = dataset
        self.iou_thr = iou_thr
        self.difficulty = difficulty
        self.verbose = verbose
        self.iou_types = ['bbox']
        self.labels = None   # det_engine's coco precision table is skipped
        self.class_names = list(DOTA_CLASSES)
        self._gt_cache = None
        self.cleanup()

    def cleanup(self):
        self.predictions = {}
        self.coco_eval = {'bbox': SimpleNamespace(stats=np.zeros(2), eval={})}

    def _build_gt(self):
        """image_id -> per-class pixel rboxes + ignore flags; parsed once from the val annfiles."""
        if self._gt_cache is not None:
            return self._gt_cache
        import cv2
        gt, ignore = {}, {}
        for idx in range(len(self.dataset)):
            polys, labels, difficult = self.dataset.load_annotation(idx)
            polys, labels, difficult = polys.numpy(), labels.numpy(), difficult.numpy()
            rboxes = np.zeros((len(polys), 5), dtype=np.float64)
            for i, poly in enumerate(polys):
                (cx, cy), (bw, bh), angle = cv2.minAreaRect(poly.reshape(4, 2).astype(np.float32))
                rboxes[i] = (cx, cy, bw, bh, math.radians(angle))
            ign = difficult > self.difficulty
            gt[idx] = {c: rboxes[labels == c] for c in range(len(self.class_names))}
            ignore[idx] = {c: ign[labels == c] for c in range(len(self.class_names))}
        self._gt_cache = (gt, ignore)
        return self._gt_cache

    def update(self, predictions):
        """predictions: {image_id: {'labels', 'boxes' (N,5 px, rad), 'scores'}}"""
        for img_id, pred in predictions.items():
            self.predictions[int(img_id)] = {
                'labels': pred['labels'].detach().cpu().numpy(),
                'boxes': pred['boxes'].detach().cpu().numpy(),
                'scores': pred['scores'].detach().cpu().numpy(),
            }

    def synchronize_between_processes(self):
        all_preds = dist_utils.all_gather(self.predictions)
        merged = {}
        for p in all_preds:
            merged.update(p)
        self.predictions = merged

    def accumulate(self):
        gt, ignore = self._build_gt()
        self.per_class_ap = voc_eval_rbox(self.predictions, gt, len(self.class_names),
                                          self.iou_thr, ignore)
        self.mAP = float(self.per_class_ap.mean())
        self.coco_eval['bbox'].stats = np.array([self.mAP, self.mAP])

    def summarize(self):
        if not dist_utils.is_main_process():
            return
        try:
            from tabulate import tabulate
            rows = [(name, f'{self.per_class_ap[i] * 100:.2f}') for i, name in enumerate(self.class_names)]
            print(tabulate(rows, headers=['class', f'AP{int(self.iou_thr * 100)}'], tablefmt='pretty'))
        except ImportError:
            for i, name in enumerate(self.class_names):
                print(f'{name}: {self.per_class_ap[i] * 100:.2f}')
        print(f'DOTA mAP@{self.iou_thr:.2f} (VOC07 11-point) = {self.mAP * 100:.2f}')
