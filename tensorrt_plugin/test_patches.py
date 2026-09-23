"""Equivalence test for the export patches, in pure PyTorch (fp16 deploy path, same CUDA kernels).

    python test_patches.py -c ../configs/det/coco_finetune/gtr_s.yml [--no-static-perm] [--no-fuse-gate]
"""

import argparse
import copy
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, '..')))
sys.path.insert(0, HERE)

import torch

from export_onnx import build_model
from gtr_onnx_patches import TASK_OUTPUTS, prepare_for_export
from verify_trt import compare


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-c', '--config', required=True)
    p.add_argument('--backbone', default='ViTAdapterSpatialSwiGLU')
    p.add_argument('--dtype', default='fp16', choices=['fp16', 'fp32'])
    p.add_argument('--no-static-perm', action='store_true')
    p.add_argument('--no-fuse-gate', action='store_true')
    p.add_argument('--gk-in-plugin', action='store_true')
    p.add_argument('--split-qkv', action='store_true')
    p.add_argument('--no-fold-gk', action='store_true')
    p.add_argument('--atlas', default='none', choices=['none', 'concat', 'pad'])
    args = p.parse_args()
    dtype = torch.float16 if args.dtype == 'fp16' else torch.float32

    torch.manual_seed(0)
    model, _, task, (H, W) = build_model(args.config, args.backbone)
    model = model.cuda().to(dtype).eval()
    for m in model.modules():
        if hasattr(m, '_deploy_parallel_stages'):
            m._deploy_parallel_stages = False
    patched = copy.deepcopy(model)
    prepare_for_export(patched, (H, W), fuse_gate=not args.no_fuse_gate, gk_in_plugin=args.gk_in_plugin,
                       static_perm=not args.no_static_perm, split_qkv=args.split_qkv, fold_gk=not args.no_fold_gk,
                       atlas=args.atlas)

    x = torch.randn(1, 3, H, W, device='cuda', dtype=dtype)
    with torch.no_grad():
        ref = model(x)
        out = patched(x)
        feats_ref = model.backbone(x)
        feats_out = patched.backbone(x)
    print(f'[patches] {task} {H}x{W} {args.dtype}: static_perm={not args.no_static_perm} fuse_gate={not args.no_fuse_gate} gk_in_plugin={args.gk_in_plugin} atlas={args.atlas}')
    for i, (a, b) in enumerate(zip(feats_ref, feats_out)):
        r = compare(a, b)
        print(f"  backbone feat{i} {str(r['shape']):<22} cos={r['cos']:.6f} max_abs={r['max_abs']:.4g} (ref |mean|={r['ref_abs_mean']:.4g})")
    query_tasks = {'detection', 'segmentation', 'obb', 'pose'}
    for k in TASK_OUTPUTS[task]:
        r = compare(ref[k], out[k], sort_axis=1 if task in query_tasks else None)
        line = f"  {k:<16} {str(r['shape']):<22} cos={r['cos']:.6f} max_abs={r['max_abs']:.4g}"
        if 'sorted_cos' in r:
            line += f"  sorted_cos={r['sorted_cos']:.6f} sorted_max_abs={r['sorted_max_abs']:.4g}"
        print(line)


if __name__ == '__main__':
    main()
