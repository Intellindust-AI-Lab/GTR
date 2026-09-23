"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Re-pack a trained GTR checkpoint into the bare `{'model': state_dict}` form that
`train.py -t` consumes as an initialisation.

Why a tool: `_Solver.load_tuning_state` prefers `state['ema']['module']` whenever the
checkpoint carries an `ema` entry, so handing a raw checkpoint to `-t` silently picks the
EMA weights. Every released stage was initialised from the RAW (`model`) weights of the
previous stage; this script pins that choice and optionally slices the backbone.

Which slice each config expects (see the config headers):
    det/coco_finetune        whole detector from the obj365 pretrain (model, ep 35)
    seg/coco_seg_finetune    whole detector from det/coco_finetune (model, last epoch)
    semseg/cityscapes_*      backbone.* of the obj365 pretrain (--backbone-only)
    obb/dota_finetune        backbone.* of the obj365 pretrain (--backbone-only)
    pose/coco_pose_finetune  backbone.* of det/coco_finetune (--backbone-only)
    depth/pretrain           the obj365 pretrain checkpoint can be passed to -t as is
                             (backbone + encoder land, the DPT head is new)

Usage:
    python tools/export_init_weights.py --ckpt outputs/obj365_pretrain/gtr_s/checkpoint0035.pth \
        --out weights/init/gtr_s_obj365_full_raw.pth
    python tools/export_init_weights.py --ckpt outputs/coco_finetune/gtr_x/last.pth \
        --out weights/init/gtr_x_coco_backbone_raw.pth --backbone-only
"""

import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ckpt', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source', choices=['model', 'ema'], default='model',
                        help="'model' = raw weights (the released recipes), 'ema' = EMA weights")
    parser.add_argument('--backbone-only', action='store_true', help='keep only backbone.* tensors')
    parser.add_argument('--expect-epoch', type=int, default=None,
                        help="assert the checkpoint's last_epoch (guards against a moved last.pth)")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    if args.expect_epoch is not None:
        assert ckpt.get('last_epoch') == args.expect_epoch, \
            f"{args.ckpt}: last_epoch={ckpt.get('last_epoch')}, expected {args.expect_epoch}"
    if args.source == 'ema':
        if not ckpt.get('ema'):
            raise KeyError(f"No 'ema' entry in {args.ckpt}; use --source model")
        state = ckpt['ema']['module']
    else:
        state = ckpt['model']
    total = len(state)
    if args.backbone_only:
        state = {k: v for k, v in state.items() if k.startswith('backbone.')}
        if not state:
            raise RuntimeError(f'no backbone.* tensors in {args.ckpt}')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': state}, args.out)
    print(f"{args.ckpt} [{args.source}{'/backbone-only' if args.backbone_only else ''}] "
          f"epoch={ckpt.get('last_epoch')} -> {args.out} ({len(state)}/{total} tensors)")


if __name__ == '__main__':
    main()
