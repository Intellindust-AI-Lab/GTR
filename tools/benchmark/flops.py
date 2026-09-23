"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
FLOPs for gtr s/m/l/x at eval resolution, via torch FlopCounterMode.

NOTE: the GLA linear-attention core (custom CUDA `gla_torch_ext` /
fla Triton `chunk_gla`) and deformable-attn grid_sample are kernel-level
ops invisible to ANY op-level FLOP counter (fvcore/thop/calflops too).
Reported number = countable GEMM/conv/SDPA FLOPs (the paper-comparable
figure); GLA recurrence + MSDA sampling excluded. q/k/v/gk projections,
all Linear/Conv, backbone, encoder, decoder GEMMs ARE counted.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))

import torch
from torch.utils.flop_counter import FlopCounterMode

from engine.core import YAMLConfig


def build(config_path):
    cfg = YAMLConfig(config_path)
    # Both backbone flavours: plain ViTAdapter and the Spatial SwiGLU variant.
    for _bb in ('ViTAdapter', 'ViTAdapterSpatialSwiGLU'):
        if _bb in cfg.yaml_cfg:
            cfg.yaml_cfg[_bb]['skip_weights_warning'] = True
        cfg.yaml_cfg['ViTAdapter']['weights_path'] = None
    model = cfg.model
    model.eval()
    eval_size = cfg.yaml_cfg.get('eval_spatial_size', [640, 640])
    return model, eval_size


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--dtype', default='fp16', choices=['fp32', 'fp16'])
    args = p.parse_args()

    name = os.path.splitext(os.path.basename(args.config))[0]
    model, eval_size = build(args.config)
    device = torch.device('cuda')
    dtype = {'fp32': torch.float32, 'fp16': torch.float16}[args.dtype]
    model = model.to(device=device, dtype=dtype)
    H, W = eval_size
    x = torch.randn(1, 3, H, W, device=device, dtype=dtype)

    params = sum(p.numel() for p in model.parameters()) / 1e6

    fcm = FlopCounterMode(display=False, depth=2)
    with fcm:
        model(x)
    total = fcm.get_total_flops()  # torch convention: 1 MAC = 2 FLOPs
    by_mod = fcm.get_flop_counts()

    print(f"{name}  eval={eval_size}  params={params:.2f}M  dtype={args.dtype}")
    print(f"  TOTAL countable: {total/1e9:.2f} GFLOPs  ({total/1e9/2:.2f} GMACs)")
    g = by_mod.get('Global', {})
    for opname, fl in sorted(g.items(), key=lambda kv: -kv[1]):
        if fl > 0:
            print(f"    {str(opname):<34} {fl/1e9:7.2f} GFLOPs")
    for mod in ['backbone', 'encoder', 'decoder']:
        key = f'GTR.{mod}'
        if key in by_mod:
            s = sum(by_mod[key].values())
            print(f"  ~{mod:<10} {s/1e9:7.2f} GFLOPs")


if __name__ == '__main__':
    main()
