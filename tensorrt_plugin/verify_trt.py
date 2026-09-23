"""Compare a TensorRT engine against the PyTorch deploy model on the same input.

The PyTorch reference runs the unpatched eval model in fp32 (fla Triton chunk_gla, no
CUDA extension), so it is independent of the plugin path. Query-based heads (det/seg/
obb/pose) are compared both raw and after sorting along the query axis, because a
near-tied top-k can permute rows without changing the set of predictions.

    python verify_trt.py -c ../configs/det/coco_finetune/gtr_s.yml --engine out/engines/gtr_s_fp16.engine
"""

import argparse
import ctypes
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, '..')))
sys.path.insert(0, HERE)

import numpy as np
import torch
import torch.nn.functional as F

from export_onnx import build_model
from gtr_onnx_patches import TASK_OUTPUTS

TRT_ROOT = os.environ.get('TRT_ROOT', os.path.join(HERE, 'tensorrt'))
PLUGIN = os.environ.get('GLA_PLUGIN', os.path.join(HERE, 'trt_plugin', 'libgla_plugin.so'))


class TRTRunner:
    def __init__(self, engine_path, plugin=PLUGIN):
        import tensorrt as trt
        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        if plugin:
            ctypes.CDLL(plugin, mode=ctypes.RTLD_GLOBAL)
        trt.init_libnvinfer_plugins(self.logger, '')
        with open(engine_path, 'rb') as f:
            blob = f.read()
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(blob)
        assert self.engine is not None, f'failed to deserialize {engine_path}'
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs = [], []
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            (self.inputs if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT else self.outputs).append(n)

    def dtype(self, name):
        m = {self.trt.float32: torch.float32, self.trt.float16: torch.float16, self.trt.int32: torch.int32,
             self.trt.int64: torch.int64, self.trt.bool: torch.bool, self.trt.uint8: torch.uint8}
        return m[self.engine.get_tensor_dtype(name)]

    def run(self, feeds):
        bufs = {}
        for n in self.inputs:
            t = feeds[n].to(self.dtype(n)).contiguous()
            bufs[n] = t
            self.context.set_input_shape(n, tuple(t.shape))
            self.context.set_tensor_address(n, t.data_ptr())
        for n in self.outputs:
            shape = tuple(self.context.get_tensor_shape(n))
            t = torch.empty(shape, dtype=self.dtype(n), device='cuda')
            bufs[n] = t
            self.context.set_tensor_address(n, t.data_ptr())
        stream = torch.cuda.current_stream()
        ok = self.context.execute_async_v3(stream.cuda_stream)
        assert ok, 'execute_async_v3 failed'
        stream.synchronize()
        return {n: bufs[n] for n in self.outputs}


def compare(ref, out, sort_axis=None):
    r, o = ref.float().flatten(), out.float().flatten()
    res = dict(shape=list(ref.shape), cos=F.cosine_similarity(r, o, dim=0).item(),
               max_abs=(r - o).abs().max().item(), ref_abs_mean=r.abs().mean().item())
    if sort_axis is not None and ref.dim() > sort_axis:
        rs, os_ = ref.float().sort(dim=sort_axis).values.flatten(), out.float().sort(dim=sort_axis).values.flatten()
        res['sorted_cos'] = F.cosine_similarity(rs, os_, dim=0).item()
        res['sorted_max_abs'] = (rs - os_).abs().max().item()
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-c', '--config', required=True)
    p.add_argument('--engine', required=True)
    p.add_argument('--backbone', default='ViTAdapterSpatialSwiGLU')
    p.add_argument('--resume', default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--ref-fp16', action='store_true', help='also compare against the torch fp16 deploy path (same CUDA kernels)')
    p.add_argument('--part', default='full', help="engine exported with --part ('full' | 'backbone' | 'encoder' | 'blocks:N')")
    p.add_argument('--json', default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    model, _, task, (H, W) = build_model(args.config, args.backbone, args.resume)
    model = model.cuda().eval().float()
    for m in model.modules():
        if hasattr(m, '_deploy_parallel_stages'):
            m._deploy_parallel_stages = False
    x = torch.randn(1, 3, H, W, device='cuda')

    def run_ref(mdl, inp):
        from export_onnx import run_part
        with torch.no_grad():
            if args.part == 'full':
                return mdl(inp)
            outs = run_part(mdl, args.part, inp)
            return {f'{args.part.replace(":", "")}_{i}': o for i, o in enumerate(outs)}

    ref = run_ref(model, x)                             # fp32 eager: fla Triton chunk_gla
    keys = TASK_OUTPUTS[task] if args.part == 'full' else list(ref.keys())
    if args.part != 'full':
        task = args.part
    ref16 = None
    if args.ref_fp16:
        # fp16 eager deploy path = the same hand-written CUDA kernels the plugin runs
        # (chunk_gla_run_gated), so this isolates plugin/graph mistakes from the
        # kernel's own fp16/bf16 numerics.
        model16 = model.half()
        ref16 = run_ref(model16, x.half())

    runner = TRTRunner(args.engine)
    feeds = {'images': x}
    if 'orig_target_sizes' in runner.inputs:
        feeds['orig_target_sizes'] = torch.tensor([[W, H]], device='cuda')
    out = runner.run(feeds)

    query_tasks = {'detection', 'segmentation', 'obb', 'pose'}
    sort_axis = 1 if task in query_tasks else None

    def show(title, a, b):
        print(title)
        rep = {}
        for k in keys:
            if k not in a or k not in b:
                continue
            rep[k] = r = compare(a[k], b[k], sort_axis=sort_axis)
            line = f"  {k:<16} {str(r['shape']):<22} cos={r['cos']:.6f} max_abs={r['max_abs']:.4g} (ref |mean|={r['ref_abs_mean']:.4g})"
            if 'sorted_cos' in r:
                line += f"  sorted_cos={r['sorted_cos']:.6f} sorted_max_abs={r['sorted_max_abs']:.4g}"
            print(line)
        return rep

    report = {'trt_vs_torch_fp32': show(f'[verify] {os.path.basename(args.engine)} vs torch fp32 (fla) ({task}, {H}x{W})', ref, out)}
    if ref16 is not None:
        report['trt_vs_torch_fp16'] = show(f'[verify] {os.path.basename(args.engine)} vs torch fp16 (CUDA ext)', ref16, out)
        report['torch_fp16_vs_fp32'] = show('[verify] torch fp16 (CUDA ext) vs torch fp32 (fla)  [kernel numerics baseline]', ref, ref16)
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(dict(config=args.config, engine=args.engine, task=task, report=report), f, indent=2)


if __name__ == '__main__':
    main()
