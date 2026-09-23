"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
NYU Depth V2 evaluation of the GTR depth models (paper protocol).

654-image Eigen test split of nyu_depth_v2_labeled.mat (filled depth), ZoeDepth Eigen
crop, GT in [1e-3, 10] m, per-image metrics averaged over the split.
  - input: shortest edge resized to the base size (480x640 -> 640x853); the backbone
    needs a square patch grid, so the frame is covered by two square tiles averaged
    in their overlap.
  - six test-time views: scales {0.75, 1.0, 1.25} x {original, horizontal flip},
    averaged in metric space.
  - per-image log-affine fit to GT (log gt ~ a * log pred + b), clamped to [1e-3, 10] m.

Usage:
  python tools/eval_nyu_depth.py -c configs/depth/pretrain/gtrdepth_s.yml -r weights/gtrdepth_s.pth
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F
import torchvision.transforms as T

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from engine.core import YAMLConfig                                    # noqa: E402
from engine.data.dataset.depth_eval import eval_depth                 # noqa: E402
from engine.data.dataset.nyu_depth_dataset import NYUDepthV2          # noqa: E402

# ZoeDepth NYU eval crop (zoedepth/utils/misc.py): eval_mask[45:471, 41:601] on 480x640.
_EIGEN_CROP = (45, 471, 41, 601)
_GT_HW = (480, 640)
_MIN_DEPTH, _MAX_DEPTH = 1e-3, 10.0
_TTA_SIZES = (480, 640, 800)
_METRIC_KEYS = ('d1', 'abs_rel', 'rmse')


def load_model(config, ckpt, device):
    cfg = YAMLConfig(config)
    state = torch.load(ckpt, map_location='cpu', weights_only=False)
    cfg.model.load_state_dict(state['ema']['module'])
    return cfg.model.eval().to(device)


def _predict_full(model, x):
    """A keep-aspect landscape frame is covered by two overlapping square tiles."""
    h, w = x.shape[-2:]
    assert h < w <= 2 * h, f'frame {h}x{w} needs more than two square tiles'
    pred_sum = x.new_zeros(x.shape[0], h, w)
    count = x.new_zeros(1, h, w)
    for x0 in (0, w - h):
        pred_sum[..., x0:x0 + h] += model(x[..., x0:x0 + h])['pred_depth']
        count[..., x0:x0 + h] += 1
    return pred_sum / count


@torch.no_grad()
def predict(model, pil_img, size, device, flip):
    """Resize -> forward -> bilinear back to the native 480x640 GT grid."""
    tf = T.Compose([T.Resize(size), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    x = tf(pil_img)[None].to(device)
    if flip:
        x = x.flip(-1)
    pred = _predict_full(model, x)                                     # [1, h, w], meters
    if flip:
        pred = pred.flip(-1)
    return F.interpolate(pred[:, None], size=_GT_HW, mode='bilinear',
                         align_corners=True)[0, 0].float()


def align_log_affine(pred, gt):
    """Least-squares a, b for log gt ~ a * log pred + b on the evaluated pixels."""
    x, y = torch.log(pred), torch.log(gt)
    A = torch.stack([x, torch.ones_like(x)], dim=1)
    a, b = torch.linalg.lstsq(A, y[:, None]).solution[:, 0]
    return torch.exp(a * x + b)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-c', '--config', required=True)
    p.add_argument('-r', '--ckpt', required=True)
    p.add_argument('--mat', default='./dataset/nyu/nyu_depth_v2_labeled.mat')
    p.add_argument('--splits', default='./dataset/nyu/splits.mat')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    torch.set_num_threads(4)
    device = torch.device(args.device)
    model = load_model(args.config, args.ckpt, device)
    ds = NYUDepthV2(args.mat, transforms=None, split='test', splits_file=args.splits)

    crop_mask = torch.zeros(_GT_HW, dtype=torch.bool, device=device)
    y0, y1, x0, x1 = _EIGEN_CROP
    crop_mask[y0:y1, x0:x1] = True

    acc = {k: 0.0 for k in _METRIC_KEYS}
    for i in range(len(ds)):
        img, target = ds.load_item(i)
        gt = target['depth'].to(device).float()
        views = [predict(model, img, size, device, flip)
                 for size in _TTA_SIZES for flip in (False, True)]
        pred = torch.stack(views).mean(0)

        valid = (gt >= _MIN_DEPTH) & (gt <= _MAX_DEPTH) & crop_mask
        aligned = align_log_affine(pred[valid], gt[valid]).clamp(_MIN_DEPTH, _MAX_DEPTH)
        m = eval_depth(aligned, gt[valid])
        for k in _METRIC_KEYS:
            acc[k] += m[k]
        if (i + 1) % 50 == 0:
            print(f'  {i + 1}/{len(ds)}', flush=True)

    print(f'{os.path.basename(args.ckpt)}  {len(ds)} images  '
          + '  '.join(f'{k}={acc[k] / len(ds):.4f}' for k in _METRIC_KEYS))


if __name__ == '__main__':
    main()
