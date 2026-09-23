"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
DEIMv2: Real-Time Object Detection Meets DINOv3
Copyright (c) 2025 The DEIMv2 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse

import torch

from engine.core import YAMLConfig, yaml_utils
from engine.misc import dist_utils
from engine.solver import TASKS

debug=False

warnings.filterwarnings("ignore")

if debug:
    import torch
    def custom_repr(self):
        return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'
    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr


def _is_valid_checkpoint(path: Path) -> bool:
    try:
        torch.load(path, map_location='cpu', weights_only=False)
    except Exception as exc:
        print(f'Skip invalid checkpoint {path}: {exc}')
        return False
    return True


def _find_latest_checkpoint(output_dir: str):
    if not output_dir:
        return None

    output_path = Path(output_dir)
    if not output_path.exists():
        return None

    candidates = []
    last_path = output_path / 'last.pth'
    best_path = output_path / 'best.pth'

    if last_path.is_file():
        candidates.append(last_path)

    candidates.extend(sorted(output_path.glob('checkpoint*.pth'), reverse=True))

    if best_path.is_file():
        candidates.append(best_path)

    visited = set()
    for path in candidates:
        resolved = str(path.resolve())
        if resolved in visited:
            continue
        visited.add(resolved)

        if _is_valid_checkpoint(path):
            return str(path)

    return None


def _maybe_enable_auto_resume(args, cfg):
    if args.resume or args.tuning or args.test_only:
        return

    if not args.auto_resume:
        print('Auto resume disabled, training will start from scratch unless --resume is set.')
        return

    resume_path = _find_latest_checkpoint(cfg.output_dir)
    if resume_path:
        cfg.resume = resume_path
        cfg.yaml_cfg['resume'] = resume_path
        print(f'Auto resume checkpoint found: {resume_path}')
    else:
        print(f'No checkpoint found under {cfg.output_dir}, training will start from scratch.')

def main(args, ) -> None:
    """main
    """
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'


    update_dict = yaml_utils.parse_cli(args.update) # update cfg from command line
    update_dict.update({k: v for k, v in args.__dict__.items() \
        if k not in ['update', 'auto_resume'] and v is not None})

    cfg = YAMLConfig(args.config, **update_dict)
    _maybe_enable_auto_resume(args, cfg)

    if cfg.resume or cfg.tuning:
        # Both backbone flavours: plain ViTAdapter and the Spatial SwiGLU variant.
        for _bb in ('ViTAdapter', 'ViTAdapterSpatialSwiGLU'):
            if _bb in cfg.yaml_cfg:
                cfg.yaml_cfg[_bb]['skip_weights_warning'] = True

    print('cfg: ', cfg.__dict__)

    solver = TASKS[cfg.yaml_cfg['task']](cfg)

    if args.test_only:
        solver.val()
    else:
        solver.fit()

    dist_utils.cleanup()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # priority 0
    parser.add_argument('-c', '--config', type=str, default='')
    parser.add_argument('-r', '--resume', type=str, help='resume from checkpoint')
    parser.add_argument('-t', '--tuning', type=str, help='tuning from checkpoint')
    parser.add_argument('-d', '--device', type=str, help='device',)
    parser.add_argument('--seed', type=int, default=0, help='exp reproducibility')
    parser.add_argument('--use-amp', action='store_true', help='auto mixed precision training')
    parser.add_argument('--output-dir', type=str, help='output directoy')
    parser.add_argument('--summary-dir', type=str, help='tensorboard summry')
    parser.add_argument('--test-only', action='store_true', default=False,)
    parser.add_argument('--auto-resume', dest='auto_resume', action='store_true', default=True,
                        help='auto resume from the latest checkpoint in output_dir')
    parser.add_argument('--no-auto-resume', dest='auto_resume', action='store_false',
                        help='disable auto resume and start from scratch unless --resume is set')

    # priority 1
    parser.add_argument('-u', '--update', nargs='+', help='update yaml config')

    # env
    parser.add_argument('--print-method', type=str, default='builtin', help='print method')
    parser.add_argument('--print-rank', type=int, default=0, help='print rank id')

    parser.add_argument('--local-rank', type=int, help='local rank id')
    args = parser.parse_args()

    main(args)
