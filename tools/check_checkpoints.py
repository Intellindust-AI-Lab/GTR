"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Check that every config under configs/ still builds, and that each released checkpoint
still loads into its config with strict=True (both the raw `model` and the `ema` weights).

The solver loads checkpoints with strict=False and drops shape-mismatched tensors, so a
renamed or resized parameter would leave part of a released model randomly initialised
without any error. Runs on CPU (no forward pass), which is what CI uses.

Usage:
  python tools/check_checkpoints.py          # all configs + the GTR-S checkpoints (~0.6 GB)
  python tools/check_checkpoints.py --all    # all configs + all released checkpoints (~5.8 GB)
"""
import argparse
import os
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from engine.core import YAMLConfig                                    # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
HF_REPO = 'Phoenix8125/GTR'

# config -> released checkpoint in the Hugging Face repo (README "Model zoo")
RELEASED = {}
for s in 'smlx':
    RELEASED[f'configs/det/obj365_pretrain/gtr_{s}.yml'] = f'obj365/gtr_{s}_obj365.pth'
    RELEASED[f'configs/det/coco_finetune/gtr_{s}.yml'] = f'det/gtr_{s}_coco.pth'
    RELEASED[f'configs/seg/coco_seg_finetune/gtrseg_{s}.yml'] = f'seg/gtrseg_{s}_coco.pth'
    RELEASED[f'configs/pose/coco_pose_finetune/gtrpose_{s}.yml'] = f'pose/gtrpose_{s}_coco.pth'
    RELEASED[f'configs/semseg/cityscapes_finetune/gtrsemseg_{s}.yml'] = f'semseg/gtrsemseg_{s}_cityscapes.pth'
    RELEASED[f'configs/depth/pretrain/gtrdepth_{s}.yml'] = f'depth/gtrdepth_{s}.pth'
for s in 'sx':
    RELEASED[f'configs/obb/dota_finetune/gtrobb_{s}.yml'] = f'obb/gtrobb_{s}_dota.pth'


def build(config):
    cfg = YAMLConfig(str(ROOT / config))
    # Both backbone flavours; the checkpoint overwrites the pretrained ViT weights anyway.
    for _bb in ('ViTAdapter', 'ViTAdapterSpatialSwiGLU'):
        if _bb in cfg.yaml_cfg:
            cfg.yaml_cfg[_bb]['skip_weights_warning'] = True
    return cfg.model


def strict_load_problems(model, state_dict):
    """What load_state_dict(strict=True) would reject; empty if the load is clean."""
    try:
        res = model.load_state_dict(state_dict, strict=False)
    except RuntimeError as e:  # size mismatches raise even with strict=False
        lines = [line.strip() for line in str(e).splitlines()[1:] if line.strip()]
        return [f'{len(lines)} size mismatch(es), e.g. {lines[0]}']
    problems = []
    if res.missing_keys:
        problems.append(f'{len(res.missing_keys)} missing key(s), e.g. {res.missing_keys[:3]}')
    if res.unexpected_keys:
        problems.append(f'{len(res.unexpected_keys)} unexpected key(s), e.g. {res.unexpected_keys[:3]}')
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--all', action='store_true', help='also check the M/L/X checkpoints (default: GTR-S only)')
    parser.add_argument('--weights-dir', default=str(ROOT / 'weights'),
                        help='download location, same layout as `hf download Phoenix8125/GTR --local-dir weights`')
    args = parser.parse_args()

    configs = sorted(p.relative_to(ROOT).as_posix() for p in ROOT.glob('configs/*/*/*.yml'))
    failures, n_loaded = [], 0
    for config in configs:
        try:
            model = build(config)
        except Exception as e:
            failures.append(f'{config}: build failed: {e!r}')
            print(f'FAIL  {config}: build failed')
            continue

        ckpt = RELEASED.get(config)
        if ckpt is None or not (args.all or config.endswith('_s.yml')):
            print(f'ok    {config}')
            continue

        state = torch.load(hf_hub_download(HF_REPO, ckpt, local_dir=args.weights_dir),
                           map_location='cpu', weights_only=False)
        weights = {'model': state.get('model'), 'ema': (state.get('ema') or {}).get('module')}
        problems = [f'{k}: {p}' for k, sd in weights.items() if sd is not None
                    for p in strict_load_problems(model, sd)]
        if all(sd is None for sd in weights.values()):
            problems = ['no model/ema weights in the checkpoint']
        if problems:
            failures += [f'{config} <- {ckpt}: {p}' for p in problems]
            print(f'FAIL  {config} <- {ckpt}')
        else:
            n_loaded += 1
            print(f'ok    {config} <- {ckpt} ({", ".join(k for k, sd in weights.items() if sd is not None)})')

    # On GitHub Actions, surface the summary and each problem as annotations on the check run.
    in_actions = os.environ.get('GITHUB_ACTIONS')
    summary = f'{len(configs)} configs, {n_loaded} checkpoints loaded with strict=True, {len(failures)} problem(s)'
    print(f'\n::notice::{summary}' if in_actions else f'\n{summary}')
    for f in failures:
        print(f'::error::{f}' if in_actions else f'  {f}')
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
