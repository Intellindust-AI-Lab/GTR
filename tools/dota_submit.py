"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
DOTA Task1 submission pipeline: run OBB inference on 1024x1024 patches, merge
patch predictions back to the original images (offset + per-class rotated NMS,
following mmrotate DOTAMetric.merge_results), and export Task1_{cls}.txt + zip
for the official evaluation server (https://captain-whu.github.io/DOTA/evaluation.html).

Typical usage
-------------
# 1) local sanity check on val (has original-image GT):
python tools/dota_submit.py -c configs/obb/dota_finetune/gtrobb_x.yml -r <ckpt.pth> \
    --img-dir ./dataset/DOTA/split_ss_1024/val/images \
    --out-dir  outputs/dota_submit_val \
    --gt-ann-dir ./dataset/DOTA/val/labelTxt

# 2) official test submission (test images are NOT in the train/val archives;
#    download DOTA-v1.0 test from the official site first, then cut 1024px
#    patches with gap 500 -- e.g. mmrotate's img_split.py with its ss_test
#    config -- into ./dataset/DOTA/split_ss_1024/test/images):
python tools/dota_submit.py -c configs/obb/dota_finetune/gtrobb_x.yml -r <ckpt.pth> \
    --weights ema \
    --img-dir ./dataset/DOTA/split_ss_1024/test/images \
    --out-dir outputs/dota_submit_test
# upload outputs/dota_submit_test/Task1/Task1.zip to the DOTA server.
"""

import argparse
import math
import os
import re
import shutil
import sys
import zipfile
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

from engine.core import YAMLConfig
from engine.data.dataset.dota_dataset import DOTA_CLASSES, DOTA_NAME2LABEL
from engine.data.dataset.dota_eval import (rbox_iou_matrix, rbox_to_poly_np,
                                           voc_eval_rbox)
from engine.misc.dist_utils import configure_tf32_from_env

PATCH_RE = re.compile(r'^(.+)__\d+__(\d+)___(\d+)$')
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
# --tta views: exactly the flip set RandomOBBFlip uses during training, at the native
# patch resolution. Following mmyolo's TTA (configs/_base_/det_p5_tta.py), which runs
# N parallel augmented pipelines and merges them with mmdet DetTTAModel semantics
# (concat -> class-aware NMS -> max_per_img). mmyolo also varies the input scale; that
# is deliberately NOT done here because these models train at a fixed 1024 input.
TTA_FLIPS = ('none', 'horizontal', 'vertical', 'diagonal')


def load_model(cfg: YAMLConfig, ckpt_path: str, device, weights_source='auto'):
    model = cfg.model
    if ckpt_path:
        state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if weights_source == 'ema':
            if not isinstance(state, dict) or state.get('ema') is None:
                raise KeyError(f'EMA weights not found in {ckpt_path}')
            weights = state['ema']['module']
            print(f'Loaded EMA weights from {ckpt_path}')
        elif weights_source == 'raw':
            if isinstance(state, dict) and 'model' in state:
                weights = state['model']
            elif isinstance(state, dict) and 'ema' in state:
                raise KeyError(f'Raw model weights not found in {ckpt_path}')
            else:
                weights = state
            print(f'Loaded raw model weights from {ckpt_path}')
        elif isinstance(state, dict) and state.get('ema') is not None:
            weights = state['ema']['module']
            print(f'Loaded EMA weights from {ckpt_path} (auto)')
        elif isinstance(state, dict) and 'model' in state:
            weights = state['model']
            print(f'Loaded raw model weights from {ckpt_path} (auto)')
        else:
            weights = state
        model.load_state_dict(weights)
    else:
        print('WARNING: no checkpoint given, using randomly initialized weights')
    return model.to(device).eval()


def init_distributed():
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if world_size == 1:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return 0, 1, device

    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        backend = 'nccl'
    else:
        device = torch.device('cpu')
        backend = 'gloo'
    dist.init_process_group(backend=backend)
    return dist.get_rank(), dist.get_world_size(), device


def collect_patch_results(local_results, out_dir, rank, world_size):
    if world_size == 1:
        return local_results

    part_dir = os.path.join(out_dir, '.dist_parts')
    if rank == 0:
        shutil.rmtree(part_dir, ignore_errors=True)
        os.makedirs(part_dir, exist_ok=True)
    dist.barrier()

    part_path = os.path.join(part_dir, f'rank{rank}.pth')
    tmp_path = part_path + '.tmp'
    torch.save(local_results, tmp_path)
    os.replace(tmp_path, part_path)
    dist.barrier()

    results = None
    if rank == 0:
        results = {}
        for part_rank in range(world_size):
            shard = torch.load(os.path.join(part_dir, f'rank{part_rank}.pth'),
                               map_location='cpu', weights_only=False)
            duplicates = results.keys() & shard.keys()
            if duplicates:
                raise RuntimeError(f'duplicate patch results: {sorted(duplicates)[:3]}')
            results.update(shard)

    dist.barrier()
    if rank == 0:
        shutil.rmtree(part_dir)
    dist.barrier()
    return results


def flip_image(x: torch.Tensor, direction: str) -> torch.Tensor:
    """(B, 3, H, W) -> flipped view; 'none' returns the input tensor itself."""
    if direction == 'horizontal':
        return x.flip(-1)
    if direction == 'vertical':
        return x.flip(-2)
    if direction == 'diagonal':
        return x.flip(-2, -1)
    return x


def unflip_rboxes(rboxes: np.ndarray, direction: str, w: int, h: int) -> np.ndarray:
    """Map dets predicted on a flipped patch back into the original patch frame.

    rboxes are (cx, cy, w, h, theta) px/rad in the long-edge convention (theta in
    [0, pi)). Coordinates follow RandomOBBFlip's x -> W - x, i.e. exactly what the
    model was trained against. A h/v flip mirrors the box, which negates theta;
    'diagonal' is a 180 degree rotation, which leaves theta unchanged.
    """
    if direction == 'none' or len(rboxes) == 0:
        return rboxes
    out = rboxes.copy()
    if direction in ('horizontal', 'diagonal'):
        out[:, 0] = w - out[:, 0]
    if direction in ('vertical', 'diagonal'):
        out[:, 1] = h - out[:, 1]
    if direction != 'diagonal':
        out[:, 4] = (-out[:, 4]) % math.pi
    return out


def merge_tta_views(views, nms_iou, max_per_img):
    """Per-patch merge of the TTA views (mmdet DetTTAModel.merge_preds for rboxes).

    views: list of (labels, rboxes, scores), each already mapped back to the patch
    frame. Concat -> per-class rotated NMS -> keep the max_per_img best, so a TTA
    patch feeds merge_patches the same detection budget a plain patch would.
    """
    labels = np.concatenate([v[0] for v in views])
    boxes = np.concatenate([v[1] for v in views])
    scores = np.concatenate([v[2] for v in views])
    if len(labels) == 0:
        return labels, boxes, scores
    keep_labels, keep_boxes, keep_scores = [], [], []
    for c in np.unique(labels):
        m = labels == c
        keep = rotated_nms(boxes[m], scores[m], nms_iou)
        keep_labels.append(np.full(len(keep), c))
        keep_boxes.append(boxes[m][keep])
        keep_scores.append(scores[m][keep])
    labels = np.concatenate(keep_labels)
    boxes = np.concatenate(keep_boxes)
    scores = np.concatenate(keep_scores)
    if len(scores) > max_per_img:
        top = np.argsort(-scores)[:max_per_img]
        labels, boxes, scores = labels[top], boxes[top], scores[top]
    return labels, boxes, scores


@torch.no_grad()
def run_inference(model, postprocessor, img_dir, device, batch_size=8, score_thr=0.05,
                  rank=0, world_size=1, tta_flips=('none',), tta_nms_iou=0.1,
                  tta_max_per_img=300):
    """-> {patch_name: (labels (N,), rboxes (N,5) px/rad, scores (N,))}"""
    all_names = sorted(os.path.splitext(f)[0] for f in os.listdir(img_dir)
                       if f.lower().endswith(('.png', '.jpg', '.bmp', '.tif')))
    names = all_names[rank::world_size]
    exts = {os.path.splitext(f)[0]: os.path.splitext(f)[1] for f in os.listdir(img_dir)}
    results = {}
    batch_imgs, batch_names = [], []

    def flush():
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs).to(device)
        sizes = torch.tensor([[x.shape[-1], x.shape[-2]]] * len(batch_imgs), device=device)
        per_view = []
        for direction in tta_flips:
            outs = postprocessor(model(flip_image(x, direction)), sizes)
            view = []
            for out in outs:
                keep = out['scores'] > score_thr
                view.append((out['labels'][keep].cpu().numpy(),
                             unflip_rboxes(out['boxes'][keep].cpu().numpy(), direction,
                                           x.shape[-1], x.shape[-2]),
                             out['scores'][keep].cpu().numpy()))
            per_view.append(view)
        for i, name in enumerate(batch_names):
            views = [v[i] for v in per_view]
            # single view (no --tta) stays bit-identical to the non-TTA path
            results[name] = views[0] if len(views) == 1 else \
                merge_tta_views(views, tta_nms_iou, tta_max_per_img)
        batch_imgs.clear()
        batch_names.clear()

    for i, name in enumerate(names):
        img = Image.open(os.path.join(img_dir, name + exts[name])).convert('RGB')
        t = torch.from_numpy(np.array(img, copy=True)).permute(2, 0, 1).float() / 255.
        t = (t - IMAGENET_MEAN) / IMAGENET_STD
        batch_imgs.append(t)
        batch_names.append(name)
        if len(batch_imgs) == batch_size:
            flush()
        if (i + 1) % 500 == 0:
            prefix = f'rank {rank}: ' if world_size > 1 else ''
            print(f'  {prefix}inference {i + 1}/{len(names)}', flush=True)
    flush()
    if world_size > 1:
        print(f'  rank {rank}: finished {len(names)}/{len(all_names)} patches', flush=True)
    return results


def rotated_nms(rboxes: np.ndarray, scores: np.ndarray, iou_thr: float):
    """Greedy rotated NMS with exact polygon IoU; returns kept indices."""
    order = np.argsort(-scores)
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        ious = rbox_iou_matrix(rboxes[i:i + 1], rboxes[order[1:]])[0]
        order = order[1:][ious <= iou_thr]
    return np.array(keep, dtype=np.int64)


def merge_patches(patch_results: dict, nms_iou=0.1):
    """Merge patch dets into original-image dets (mmrotate DOTAMetric.merge_results).

    -> {ori_name: {'labels', 'boxes' (N,5), 'scores'}}
    """
    collector = defaultdict(list)
    for patch_name, (labels, rboxes, scores) in patch_results.items():
        m = PATCH_RE.match(patch_name)
        if m is None:
            ori, x, y = patch_name, 0, 0   # non-split image: keep as is
        else:
            ori, x, y = m.group(1), int(m.group(2)), int(m.group(3))
        if len(labels) == 0:
            collector[ori]   # register the image even if empty
            continue
        shifted = rboxes.copy()
        shifted[:, 0] += x
        shifted[:, 1] += y
        collector[ori].append((labels, shifted, scores))

    merged = {}
    for ori, chunks in collector.items():
        if not chunks:
            merged[ori] = {'labels': np.zeros(0, dtype=np.int64),
                           'boxes': np.zeros((0, 5)), 'scores': np.zeros(0)}
            continue
        labels = np.concatenate([c[0] for c in chunks])
        boxes = np.concatenate([c[1] for c in chunks])
        scores = np.concatenate([c[2] for c in chunks])
        keep_labels, keep_boxes, keep_scores = [], [], []
        for c in np.unique(labels):
            m = labels == c
            keep = rotated_nms(boxes[m], scores[m], nms_iou)
            keep_labels.append(np.full(len(keep), c))
            keep_boxes.append(boxes[m][keep])
            keep_scores.append(scores[m][keep])
        merged[ori] = {'labels': np.concatenate(keep_labels),
                       'boxes': np.concatenate(keep_boxes),
                       'scores': np.concatenate(keep_scores)}
    return merged


def export_task1(merged: dict, out_dir: str):
    """Write Task1_{cls}.txt files + Task1.zip (DOTA Task1 format)."""
    os.makedirs(out_dir, exist_ok=True)
    files = [os.path.join(out_dir, f'Task1_{cls}.txt') for cls in DOTA_CLASSES]
    handles = [open(f, 'w') for f in files]
    for ori in sorted(merged):
        det = merged[ori]
        if len(det['labels']) == 0:
            continue
        polys = rbox_to_poly_np(det['boxes']).reshape(-1, 8)
        for c, poly, s in zip(det['labels'], polys, det['scores']):
            # Keep enough score precision for the evaluator's global ranking.
            row = [ori, f'{float(s):.8f}'] + [f'{p:.2f}' for p in poly]
            handles[int(c)].write(' '.join(row) + '\n')
    for h in handles:
        h.close()
    zip_path = os.path.join(out_dir, 'Task1.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, os.path.basename(f))
    print(f'Task1 submission written to {zip_path}')
    return zip_path


def load_fullimage_gt(ann_dir: str):
    """Original-image DOTA annotations -> {ori_name: {class_id: (M,5) px/rad}}."""
    import cv2
    gt = {}
    for fn in sorted(os.listdir(ann_dir)):
        if not fn.endswith('.txt'):
            continue
        name = os.path.splitext(fn)[0]
        per_cls = defaultdict(list)
        with open(os.path.join(ann_dir, fn)) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 9 or parts[8] not in DOTA_NAME2LABEL:
                    continue
                try:
                    poly = np.array([float(v) for v in parts[:8]], dtype=np.float32)
                except ValueError:
                    continue
                (cx, cy), (bw, bh), ang = cv2.minAreaRect(poly.reshape(4, 2))
                per_cls[DOTA_NAME2LABEL[parts[8]]].append((cx, cy, bw, bh, math.radians(ang)))
        gt[name] = {c: np.array(v, dtype=np.float64) for c, v in per_cls.items()}
    if not gt:
        # Fail loudly: an empty GT would silently evaluate to all-zero AP.
        subdirs = [d for d in sorted(os.listdir(ann_dir))
                   if os.path.isdir(os.path.join(ann_dir, d))]
        hint = (f' No *.txt files here, but subdirectories exist: {subdirs}.'
                ' DOTA layouts often nest annfiles, e.g. annotations/version1.0/.'
                if subdirs else '')
        raise FileNotFoundError(f'No DOTA annfiles parsed from {ann_dir}.{hint}')
    return gt


def evaluate_fullimage(merged: dict, ann_dir: str, iou_thr=0.5):
    gt = load_fullimage_gt(ann_dir)
    preds = {k: v for k, v in merged.items() if k in gt}
    missing = set(merged) - set(gt)
    if missing:
        print(f'WARNING: {len(missing)} predicted images have no GT annfile, ignored')
    if not preds:
        raise ValueError(
            f'None of the {len(merged)} merged image ids match any GT annfile in {ann_dir}; '
            '--gt-ann-dir must point to ORIGINAL-image annotations matching the inference split.')
    ap = voc_eval_rbox(preds, gt, len(DOTA_CLASSES), iou_thr)
    for i, cls in enumerate(DOTA_CLASSES):
        print(f'  {cls}: {ap[i] * 100:.2f}')
    print(f'Full-image DOTA mAP@{iou_thr:.2f} (VOC07 11-point) = {ap.mean() * 100:.2f}')
    return ap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('-r', '--resume', default='', help='checkpoint (.pth)')
    parser.add_argument('--weights', choices=('auto', 'raw', 'ema'), default='auto',
                        help='checkpoint weight source; auto prefers EMA (default: auto)')
    parser.add_argument('--img-dir', required=True, help='directory of 1024x1024 patches')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--gt-ann-dir', default='',
                        help='original-image annfiles for local eval, e.g. '
                             '/path/to/DOTA/val/labelTxt '
                             '(the dir that directly contains P*.txt; must be the '
                             'Task1 OBB labelTxt, NOT the Task2 HBB annotations)')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--score-thr', type=float, default=0.05)
    parser.add_argument('--nms-iou', type=float, default=0.1)
    parser.add_argument('--tta', action='store_true',
                        help='test-time augmentation: infer each patch under the '
                             f'{len(TTA_FLIPS)} flips {TTA_FLIPS} RandomOBBFlip uses at '
                             'train time, map the boxes back and merge them per patch '
                             '(concat + per-class rotated NMS at --nms-iou, capped at '
                             'the head\'s num_top_queries). Costs one forward pass per '
                             'flip; off by default, and off is bit-identical to no TTA')
    args = parser.parse_args()

    rank, world_size, device = init_distributed()
    configure_tf32_from_env()
    cfg = YAMLConfig(args.config)
    model = load_model(cfg, args.resume, device, weights_source=args.weights)
    postprocessor = cfg.postprocessor

    tta_flips = TTA_FLIPS if args.tta else ('none',)
    if rank == 0:
        print(f'Running inference on {args.img_dir} with {world_size} process(es) ...')
        if args.tta:
            print(f'TTA on: {len(tta_flips)} flip views {tta_flips}, merged per patch '
                  f'with rotated NMS iou {args.nms_iou}, max {postprocessor.num_top_queries} '
                  'boxes per patch')
    patch_results = run_inference(model, postprocessor, args.img_dir, device,
                                  batch_size=args.batch_size, score_thr=args.score_thr,
                                  rank=rank, world_size=world_size, tta_flips=tta_flips,
                                  tta_nms_iou=args.nms_iou,
                                  tta_max_per_img=postprocessor.num_top_queries)
    patch_results = collect_patch_results(patch_results, args.out_dir, rank, world_size)

    if rank == 0:
        print(f'Merging {len(patch_results)} patches ...')
        merged = merge_patches(patch_results, nms_iou=args.nms_iou)
        print(f'{len(merged)} original images after merge')
        export_task1(merged, os.path.join(args.out_dir, 'Task1'))

        if args.gt_ann_dir:
            evaluate_fullimage(merged, args.gt_ann_dir)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
