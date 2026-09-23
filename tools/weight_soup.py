#!/usr/bin/env python
"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Weight averaging (model soup / tail-SWA) for gtr checkpoints.

Averages the EMA weights (ckpt['ema']['module']) across several checkpoints and
writes the result back into BOTH ema['module'] and model of a base checkpoint, so
that `train.py --test-only -r <soup>` (which evals self.ema.module) picks up the
averaged weights.

Only floating-point tensors with matching shapes across all inputs are averaged;
non-float buffers (e.g. *.num_batches_tracked) and resolution buffers are copied
from the base checkpoint unchanged.
"""

import argparse
import torch


def load_weights(path, src):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if src == 'ema':
        assert 'ema' in ckpt, f'{path} has no ema'
        return ckpt['ema']['module']
    return ckpt['model']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts', nargs='+', required=True, help='checkpoints to average')
    ap.add_argument('--out', required=True, help='output soup checkpoint path')
    ap.add_argument('--src', choices=['ema', 'model'], default='ema',
                    help='which weights to average from each input ckpt')
    ap.add_argument('--base', default=None,
                    help='checkpoint to take metadata + non-averaged keys from (default: first ckpt)')
    args = ap.parse_args()

    base_path = args.base or args.ckpts[0]
    base = torch.load(base_path, map_location='cpu', weights_only=False)

    sds = [load_weights(p, args.src) for p in args.ckpts]
    n = len(sds)
    ref = sds[0]

    avg, averaged, copied = {}, 0, 0
    for k, v in ref.items():
        vals = [sd[k] for sd in sds if k in sd]
        same = (len(vals) == n
                and v.is_floating_point()
                and all(t.shape == v.shape for t in vals))
        if same:
            acc = torch.zeros_like(v, dtype=torch.float64)
            for t in vals:
                acc += t.to(torch.float64)
            avg[k] = (acc / n).to(v.dtype)
            averaged += 1
        else:
            avg[k] = base['ema']['module'].get(k, v) if 'ema' in base else v
            copied += 1

    # write soup into the slots eval reads (ema.module) + model, keep other metadata
    if 'ema' in base:
        base['ema']['module'] = avg
    base['model'] = {k: avg[k] for k in base['model']} if 'model' in base else avg
    base['best_stat'] = {}
    base['__soup__'] = {'src': args.src, 'ckpts': args.ckpts, 'n': n}

    torch.save(base, args.out)
    print(f'soup({args.src}) of {n} ckpts -> {args.out}')
    print(f'  averaged {averaged} tensors, copied {copied} (non-float / shape-mismatch)')


if __name__ == '__main__':
    main()
