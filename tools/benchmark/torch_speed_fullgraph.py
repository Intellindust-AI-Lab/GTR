"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Forward-only latency benchmark: FP16 deploy model, torch.compile(fullgraph=True),
one CUDA graph per forward, verified with the PyTorch profiler.

The whole model (backbone + encoder + task head; no pre/post-processing) is traced
into a single FX graph -- fullgraph=True turns any data-dependent graph break or
device sync into a compile-time error -- compiled by inductor, then captured into
ONE CUDA graph that is replayed per forward. The steady state is checked with the
profiler: exactly 1 cudaGraphLaunch, 0 GPU->CPU scalar syncs, 0 device-to-host
copies per forward.

Usage:
    python tools/benchmark/torch_speed_fullgraph.py --all-tasks --json results.json
    python tools/benchmark/torch_speed_fullgraph.py \
        --configs configs/det/coco_finetune/gtr_s.yml configs/obb/dota_finetune/gtrobb_s.yml
"""

import argparse
import ast
import gc
import json
import os
import platform
import statistics
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function

from engine.core import YAMLConfig

TASKS = {
    'det': 'configs/det/coco_finetune/gtr_{}.yml',
    'seg': 'configs/seg/coco_seg_finetune/gtrseg_{}.yml',
    'obb': 'configs/obb/dota_finetune/gtrobb_{}.yml',
    'pose': 'configs/pose/coco_pose_finetune/gtrpose_{}.yml',
    'semseg': 'configs/semseg/cityscapes_finetune/gtrsemseg_{}.yml',
    'depth': 'configs/depth/pretrain/gtrdepth_{}.yml',
}
SIZES = ('s', 'm', 'l', 'x')
BACKBONES = ('ViTAdapter', 'ViTAdapterSpatialSwiGLU')


# ----------------------------------------------------------------------------- model
def build_model(config_path, backbone=None):
    cfg = YAMLConfig(config_path)
    y = cfg.yaml_cfg
    model_key = y['model']
    if backbone is not None and y[model_key].get('backbone') != backbone:
        # Both flavours take the same config keys: carry the size-specific section over.
        src = y[model_key]['backbone']
        sec = dict(y.get(src, {}))
        sec.update(y.get(backbone, {}))
        y[backbone] = sec
        y[model_key]['backbone'] = backbone
    for bb in BACKBONES:
        if bb in y:
            y[bb]['skip_weights_warning'] = True
            y[bb]['weights_path'] = None
    model = cfg.model
    model = model.deploy()
    return model, y[model_key]['backbone'], tuple(y['eval_spatial_size'])


def prepare_model(model, device, dtype, parallel_stages):
    model = model.to(device=device, dtype=dtype)
    # Conv weights in the layout inductor's channels_last plan wants: no per-forward
    # weight re-layout kernels and fewer activation layout conversions (-25 kernels/fwd on gtr_s).
    model = model.to(memory_format=torch.channels_last)
    for m in model.modules():
        # The encoder's multi-stream stage fork/join is eager-only (inductor has no
        # multi-stream codegen); it is kept for --compile split and dropped for
        # --compile whole, where the stages run sequentially inside the single graph.
        if hasattr(m, '_deploy_parallel_stages'):
            m._deploy_parallel_stages = parallel_stages
    return model


def compile_model(model, args):
    kw = dict(mode=args.mode, fullgraph=True, dynamic=False)
    if args.compile == 'whole':
        return torch.compile(model, **kw)
    # split: one fullgraph=True compile per region (backbone, each encoder stage, task
    # head). The eager glue is GTR.forward plus the encoder's stream fork/join, so the
    # three encoder stages overlap on idle SMs inside the same captured CUDA graph.
    model.backbone = torch.compile(model.backbone, **kw)
    enc = model.encoder
    if getattr(enc, '_deploy_parallel_stages', False) and len(enc.stages) > 1:
        enc._run_stage = torch.compile(enc._run_stage, **kw)   # specialises per stage index
    else:
        model.encoder = torch.compile(model.encoder, **kw)
    head = 'decoder' if hasattr(model, 'decoder') else 'head'
    setattr(model, head, torch.compile(getattr(model, head), **kw))
    return model


# ----------------------------------------------------------------------------- outputs
def flatten_outputs(o):
    if isinstance(o, torch.Tensor):
        return [o]
    if isinstance(o, dict):
        return [t for v in o.values() for t in flatten_outputs(v)]
    if isinstance(o, (list, tuple)):
        return [t for v in o for t in flatten_outputs(v)]
    return []


def compare_outputs(ref, out):
    """Compare two output trees. Query-based heads emit rows whose order comes from a
    top-k over (with random weights) near-tied scores, so a last-bit difference can
    permute rows; sorting along dim 1 makes the comparison permutation-invariant."""
    res = []
    for r, o in zip(flatten_outputs(ref), flatten_outputs(out)):
        r, o = r.float(), o.float()
        raw_cos = F.cosine_similarity(r.flatten(), o.flatten(), dim=0).item()
        if r.dim() >= 2 and r.shape[1] > 1:
            r, o = r.sort(dim=1).values, o.sort(dim=1).values
        res.append(dict(shape=list(r.shape), raw_cos=raw_cos,
                        sorted_cos=F.cosine_similarity(r.flatten(), o.flatten(), dim=0).item(),
                        sorted_max_abs=(r - o).abs().max().item()))
    return res


# ----------------------------------------------------------------------------- timing
def time_fn(fn, iters, warmup, warmup_seconds):
    # The SM clock ramps on a ~1 s timescale under sustained load (no clock locking in
    # a container), so warm up by wall time, not just by iteration count.
    t0 = time.time()
    n = 0
    while n < warmup or time.time() - t0 < warmup_seconds:
        fn()
        n += 1
        if n % 50 == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    raw = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    t = sorted(raw)
    chunk = max(1, iters // 10)
    return dict(min=t[0], mean=statistics.fmean(t), median=statistics.median(t),
                p95=t[max(0, int(len(t) * 0.95) - 1)], p99=t[max(0, int(len(t) * 0.99) - 1)],
                max=t[-1], iters=iters,
                chunk_means=[statistics.fmean(raw[i:i + chunk]) for i in range(0, iters, chunk)])


SYNC_NAMES = ('cudaStreamSynchronize', 'cudaDeviceSynchronize', 'cudaEventSynchronize')
SCALAR_SYNC_OPS = ('aten::item', 'aten::_local_scalar_dense', 'aten::nonzero',
                   'aten::masked_select', 'aten::unique')


def verify_steady_state(fn, n=10):
    """Profile n steady-state forwards and count the CUDA runtime calls that matter.
    Totals over the whole profiled span are divided by n (a record_function window
    per forward is not reliable: Kineto emits two events per annotation)."""
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            fn()
        # no explicit sync here: the profiler's own teardown sync lands after the last launch
    evs = list(prof.events())
    cpu = [e for e in evs if e.device_type == torch.autograd.DeviceType.CPU]
    gpu = [e for e in evs if e.device_type == torch.autograd.DeviceType.CUDA]
    launches = [e for e in cpu if e.name == 'cudaGraphLaunch']
    last_launch = max((e.time_range.start for e in launches), default=0)
    syncs = [e for e in cpu if e.name in SYNC_NAMES]
    syncs_in_run = [e for e in syncs if e.time_range.start <= last_launch]
    kernels = [e for e in gpu if 'Memcpy' not in e.name and 'Memset' not in e.name]
    summary = dict(
        forwards=n,
        graph_launches=len(launches),
        graph_launch_per_fwd=len(launches) / n,
        kernel_launches=sum(1 for e in cpu if e.name == 'cudaLaunchKernel'),
        syncs_in_run=len(syncs_in_run),
        syncs_total=len(syncs),   # includes the profiler teardown sync after the last launch
        memcpy_calls=sum(1 for e in cpu if e.name.startswith('cudaMemcpy')),
        scalar_ops=sum(1 for e in cpu if e.name in SCALAR_SYNC_OPS),
        d2h_copies=sum(1 for e in gpu if 'DtoH' in e.name),
        h2d_copies=sum(1 for e in gpu if 'HtoD' in e.name),
        gpu_kernels_per_fwd=len(kernels) / n,
        gpu_kernel_time_ms_per_fwd=sum(e.time_range.elapsed_us() for e in kernels) / n / 1000.0,
    )
    # The gate: exactly one graph launch per forward, no host sync, no host-scalar op, no D2H copy.
    summary['ok'] = (summary['graph_launches'] == n and summary['syncs_in_run'] == 0
                     and summary['scalar_ops'] == 0 and summary['d2h_copies'] == 0)
    return summary


# ----------------------------------------------------------------------------- one model
def benchmark_one(config_path, args, device, dtype):
    name = os.path.splitext(os.path.basename(config_path))[0]
    task = os.path.relpath(config_path, 'configs').split(os.sep)[0]
    print(f"\n{'=' * 78}\n>>> {name} [{task}]  ({config_path})")
    model, backbone, (H, W) = build_model(config_path, args.backbone)
    model = prepare_model(model, device, dtype, parallel_stages=(args.compile == 'split'))
    params = sum(p.numel() for p in model.parameters()) / 1e6
    tokens = (H // 16) * (W // 16)
    print(f"    backbone={backbone} eval={H}x{W} tokens={tokens} dtype={args.dtype} "
          f"deploy-params={params:.2f}M")

    torch.manual_seed(0)
    x = torch.randn(args.batch, 3, H, W, device=device, dtype=dtype)
    if args.graph == 'inductor':
        # cudagraph trees: a static-address input is read in place instead of being
        # copied into a placeholder before every replay.
        torch._dynamo.mark_static_address(x)
    with torch.inference_mode():
        for _ in range(3):
            ref = model(x)
        torch.cuda.synchronize()

    t0 = time.time()
    compiled = compile_model(model, args)
    with torch.inference_mode():
        out = compiled(x)
        torch.cuda.synchronize()
        compile_s = time.time() - t0
        for _ in range(3):
            out = compiled(x)
        torch.cuda.synchronize()
    print(f"    compile ({args.compile}, fullgraph=True, mode={args.mode}): {compile_s:.0f}s")

    graph = None
    if args.graph == 'manual':
        # Whole compiled forward -> one CUDA graph. Warm up on a side stream first so
        # cuDNN/cuBLAS/Triton lazy init never lands inside the capture.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.inference_mode():
            for _ in range(3):
                compiled(x)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), torch.inference_mode():
            static_out = compiled(x)
        torch.cuda.synchronize()

        def fn():
            graph.replay()
        fn()
        torch.cuda.synchronize()
        out = static_out
    else:
        def fn():
            with torch.inference_mode():
                return compiled(x)
        with torch.inference_mode():
            for _ in range(5):   # cudagraph trees: warm-up run, record, then replay
                out = compiled(x)
        torch.cuda.synchronize()

    correctness = compare_outputs(ref, out)
    for c in correctness:
        print(f"    output {c['shape']}: raw_cos={c['raw_cos']:.6f} "
              f"sorted_cos={c['sorted_cos']:.6f} sorted_max_abs={c['sorted_max_abs']:.4g}")

    timing = time_fn(fn, args.iters, args.warmup, args.warmup_seconds)
    print(f"    latency: min={timing['min']:.3f} mean={timing['mean']:.3f} "
          f"median={timing['median']:.3f} p95={timing['p95']:.3f} p99={timing['p99']:.3f} ms  "
          f"({args.iters} iters)")
    print('    chunk means:', ' '.join(f'{c:.3f}' for c in timing['chunk_means']))

    verify = verify_steady_state(fn, n=args.profile_iters)
    print(f"    profiler ({verify['forwards']} fwd): graph_launch/fwd={verify['graph_launch_per_fwd']:.1f} "
          f"cudaLaunchKernel={verify['kernel_launches']} sync(in run)={verify['syncs_in_run']} "
          f"memcpy_calls={verify['memcpy_calls']} scalar_ops={verify['scalar_ops']} "
          f"D2H={verify['d2h_copies']} H2D={verify['h2d_copies']} "
          f"kernels/fwd={verify['gpu_kernels_per_fwd']:.0f} gpu_time={verify['gpu_kernel_time_ms_per_fwd']:.3f}ms "
          f"-> {'OK' if verify['ok'] else 'FAIL'}")

    result = dict(name=name, task=task, config=config_path, backbone=backbone, eval_size=[H, W],
                  tokens=tokens, deploy_params_M=params, compile=args.compile, compile_s=compile_s,
                  timing=timing, verify=verify, correctness=correctness)

    del graph, compiled, model, fn, out, ref, x
    gc.collect()
    torch.cuda.empty_cache()
    torch._dynamo.reset()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--configs', nargs='*', default=[])
    p.add_argument('--all-tasks', action='store_true', help='all 6 tasks x s/m/l/x')
    p.add_argument('--tasks', nargs='*', default=None, help=f'subset of {list(TASKS)} for --all-tasks')
    p.add_argument('--sizes', nargs='*', default=list(SIZES))
    p.add_argument('--backbone', default='ViTAdapterSpatialSwiGLU', choices=list(BACKBONES) + ['config'],
                   help="backbone flavour for every model ('config' keeps each config's own)")
    p.add_argument('--dtype', default='fp16', choices=['fp16', 'bf16', 'fp32'])
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--compile', default='whole', choices=['whole', 'split'],
                   help="'whole': torch.compile(model, fullgraph=True) as one graph; 'split': fullgraph=True per "
                        "region (backbone / encoder stages / head) with the encoder's multi-stream fork/join "
                        "kept in eager -- both are captured into ONE CUDA graph")
    p.add_argument('--mode', default='max-autotune-no-cudagraphs',
                   choices=['default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs'])
    p.add_argument('--graph', default='manual', choices=['manual', 'inductor'],
                   help="'manual': capture the compiled forward into one torch.cuda.CUDAGraph; "
                        "'inductor': rely on inductor cudagraph trees (reduce-overhead / max-autotune)")
    p.add_argument('--ic', action='append', default=[], help='inductor config override key=value')
    p.add_argument('--warmup', type=int, default=100, help='minimum warm-up iterations')
    p.add_argument('--warmup-seconds', type=float, default=3.0, help='minimum warm-up wall time (clock ramp)')
    p.add_argument('--iters', type=int, default=1000)
    p.add_argument('--profile-iters', type=int, default=10)
    p.add_argument('--json', default=None)
    args = p.parse_args()
    if args.backbone == 'config':
        args.backbone = None
    if args.mode in ('reduce-overhead', 'max-autotune'):
        args.graph = 'inductor'   # the compiled callable already replays its own graph

    import torch._inductor.config as icfg
    for kv in args.ic:
        k, v = kv.split('=', 1)
        obj = icfg
        parts = k.split('.')
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], ast.literal_eval(v))
        print(f'inductor config: {k} = {ast.literal_eval(v)}')

    configs = list(args.configs)
    if args.all_tasks:
        for task in (args.tasks or list(TASKS)):
            configs += [TASKS[task].format(s) for s in args.sizes]
    if not configs:
        p.error('give --configs or --all-tasks')

    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}[args.dtype]

    from engine.gtr.backbone.vit_adapter import _gla_cuda_ext, _gla_op_gated
    assert _gla_cuda_ext is not None and _gla_op_gated is not None, \
        'CUDA GLA extension not loaded: run bash engine/gtr/backbone/csrc/build.sh'
    import triton
    env = dict(gpu=torch.cuda.get_device_name(0), python=platform.python_version(),
               torch=torch.__version__, torchvision=__import__('torchvision').__version__,
               triton=triton.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
               mode=args.mode, graph=args.graph, compile=args.compile, dtype=args.dtype, batch=args.batch,
               inductor_overrides=args.ic, backbone=args.backbone or 'per-config')
    print('env:', json.dumps(env))

    results = []
    for cfg_path in configs:
        try:
            results.append(benchmark_one(cfg_path, args, device, dtype))
        except Exception:
            traceback.print_exc()
            results.append(dict(name=os.path.splitext(os.path.basename(cfg_path))[0],
                                config=cfg_path, error=traceback.format_exc().splitlines()[-1]))
            gc.collect()
            torch.cuda.empty_cache()
            torch._dynamo.reset()
        if args.json:
            with open(args.json, 'w') as f:
                json.dump(dict(env=env, results=results), f, indent=2)

    print(f"\n{'=' * 78}\nSummary  (fp16, batch={args.batch}, fullgraph=True, compile={args.compile}, "
          f"mode={args.mode}, 1 CUDA graph / forward; ms)")
    print(f"  {'model':<14} {'task':<7} {'eval':<10} {'min':>8} {'mean':>8} {'median':>8} {'p95':>8} {'verify':>7}")
    for r in results:
        if 'error' in r:
            print(f"  {r['name']:<14} ERROR: {r['error']}")
            continue
        t = r['timing']
        print(f"  {r['name']:<14} {r['task']:<7} {r['eval_size'][0]}x{r['eval_size'][1]:<5} "
              f"{t['min']:>8.3f} {t['mean']:>8.3f} {t['median']:>8.3f} {t['p95']:>8.3f} "
              f"{'OK' if r['verify']['ok'] else 'FAIL':>7}")
    return 0 if all('error' not in r and r['verify']['ok'] for r in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
