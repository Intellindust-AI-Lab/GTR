"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
PyTorch forward-inference latency benchmark for gtr s/m/l/x.

Usage:
    python tools/benchmark/torch_speed.py \
        --configs configs/det/coco_finetune/gtr_s.yml configs/det/coco_finetune/gtr_m.yml \
                  configs/det/coco_finetune/gtr_l.yml configs/det/coco_finetune/gtr_x.yml \
        [--dtype fp16|bf16|fp32] [--cuda-graph] [--no-deploy]
"""

import argparse
import gc
import os
import sys
import time
from contextlib import nullcontext

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))

import torch
import torch.nn as nn

from engine.core import YAMLConfig


def build_model(config_path: str, deploy: bool = True):
    cfg = YAMLConfig(config_path)
    # Both backbone flavours: plain ViTAdapter and the Spatial SwiGLU variant.
    for _bb in ('ViTAdapter', 'ViTAdapterSpatialSwiGLU'):
        if _bb in cfg.yaml_cfg:
            cfg.yaml_cfg[_bb]['skip_weights_warning'] = True
        cfg.yaml_cfg['ViTAdapter']['weights_path'] = None

    model = cfg.model
    model.eval()
    if deploy:
        model = model.deploy()
    eval_size = cfg.yaml_cfg.get('eval_spatial_size', [640, 640])
    return model, eval_size


@torch.inference_mode()
def measure_latency(fn, n_warmup: int, n_iter: int):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for i in range(n_iter):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times_ms = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    times_ms.sort()
    return times_ms


def _summary(times_ms):
    n = len(times_ms)
    avg = sum(times_ms) / n
    p50 = times_ms[n // 2]
    p95 = times_ms[max(0, int(n * 0.95) - 1)]
    p99 = times_ms[max(0, int(n * 0.99) - 1)]
    mn = times_ms[0]
    mx = times_ms[-1]
    return dict(min=mn, p50=p50, avg=avg, p95=p95, p99=p99, max=mx)


def make_cuda_graph_runner(model, x_shape, device, dtype, n_warmup=20):
    """Capture the entire forward as a CUDA graph (static input shape)."""
    static_x = torch.randn(*x_shape, device=device, dtype=dtype)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.inference_mode():
        for _ in range(n_warmup):
            _ = model(static_x)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g), torch.inference_mode():
        static_out = model(static_x)

    def runner(x):
        static_x.copy_(x)
        g.replay()
        return static_out

    return runner, static_out


def benchmark_one(config_path, args):
    name = os.path.splitext(os.path.basename(config_path))[0]
    print(f"\n{'='*72}\n>>> {name}: building model...")
    model, eval_size = build_model(config_path, deploy=not args.no_deploy)

    device = torch.device('cuda')
    dtype = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}[args.dtype]
    model = model.to(device=device, dtype=dtype)

    # Tag dtype-aware forward path: AMP autocast is unnecessary because we cast model + input
    # together; FlashAttention / GLA kernels accept fp16/bf16 directly.

    H, W = eval_size
    x_shape = (args.batch, 3, H, W)
    print(f"    eval_size={eval_size}, dtype={args.dtype}, batch={args.batch}")
    print(f"    params: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.fp8_quant:
        from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig
        quantize_(model, Float8DynamicActivationFloat8WeightConfig())
        print("    [fp8 dynamic activation/weight quant applied]")

    if args.compile:
        # reduce-overhead uses CUDA graphs internally; default uses regular Inductor codegen.
        compiled = torch.compile(model, mode=args.compile_mode, fullgraph=False, dynamic=False)
        # warmup compile
        with torch.inference_mode():
            x_w = torch.randn(*x_shape, device=device, dtype=dtype)
            for _ in range(3):
                compiled(x_w)
            torch.cuda.synchronize()
        model = compiled
    elif args.compile_submodules:
        # Compile each top-level submodule (backbone/encoder/decoder) separately so Dynamo's
        # graph breaks at custom Triton ops (chunk_gla) don't leak across submodules.
        # We run our own CUDA-graph capture downstream to eliminate dispatch entirely.
        no_cg_mode = 'max-autotune-no-cudagraphs'
        try:
            if args.compile_backbone:
                # Increase recompile cap so the 12-layer ViT can specialise per-layer if needed.
                torch._dynamo.config.recompile_limit = 32
                model.backbone = torch.compile(model.backbone, mode=no_cg_mode, fullgraph=False, dynamic=False)
            if getattr(model.encoder, '_deploy_parallel_stages', False):
                # Stream fork/join must stay in eager python; compile each stage's
                # compute (sampling branches + C2f fuse) individually instead.
                enc = model.encoder
                for si in range(len(enc.stages)):
                    enc.stages[si] = torch.compile(enc.stages[si], mode=no_cg_mode, fullgraph=False, dynamic=False)
                    slist = enc.stages_sampling[si]
                    for sj in range(len(slist)):
                        if not isinstance(slist[sj], nn.Identity):
                            slist[sj] = torch.compile(slist[sj], mode=no_cg_mode, fullgraph=False, dynamic=False)
            else:
                model.encoder = torch.compile(model.encoder, mode=no_cg_mode, fullgraph=False, dynamic=False)
            model.decoder = torch.compile(model.decoder, mode=no_cg_mode, fullgraph=False, dynamic=False)
        except Exception as e:
            print(f"    [compile-submodules failed: {e}; falling back]")
        # warmup
        with torch.inference_mode():
            x_w = torch.randn(*x_shape, device=device, dtype=dtype)
            if args.channels_last:
                x_w = x_w.contiguous(memory_format=torch.channels_last)
            for _ in range(3):
                model(x_w)
            torch.cuda.synchronize()

    x = torch.randn(*x_shape, device=device, dtype=dtype)
    if args.channels_last:
        x = x.contiguous(memory_format=torch.channels_last)

    if args.cuda_graph:
        # CUDA graph capture
        runner, _ = make_cuda_graph_runner(model, x_shape, device, dtype, n_warmup=args.warmup)
        fn = lambda: runner(x)
        label = "cuda-graph"
    else:
        fn = lambda: model(x)
        label = "compile" if args.compile else "eager"

    # Lock GPU clocks (best effort, may need root). Skip if fails.
    if args.profile:
        from torch.profiler import profile, record_function, ProfilerActivity
        # warmup outside profiler
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
            for _ in range(20):
                fn()
        torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    times = measure_latency(fn, n_warmup=args.warmup, n_iter=args.iters)
    s = _summary(times)
    print(f"    [{label}] min={s['min']:.3f}ms p50={s['p50']:.3f}ms avg={s['avg']:.3f}ms "
          f"p95={s['p95']:.3f}ms p99={s['p99']:.3f}ms max={s['max']:.3f}ms")

    # cleanup
    del model, x
    if args.cuda_graph:
        del runner
    gc.collect()
    torch.cuda.empty_cache()
    return name, s


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--configs', nargs='+', required=True)
    p.add_argument('--dtype', default='fp16', choices=['fp32', 'fp16', 'bf16'])
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--warmup', type=int, default=30)
    p.add_argument('--iters', type=int, default=200)
    p.add_argument('--cuda-graph', action='store_true')
    p.add_argument('--no-deploy', action='store_true', help='do not call model.deploy()')
    p.add_argument('--tf32', action='store_true', help='enable tf32 (fp32 only)')
    p.add_argument('--compile', action='store_true', help='wrap model in torch.compile')
    p.add_argument('--compile-submodules', action='store_true', help='compile encoder+decoder (and backbone if --compile-backbone)')
    p.add_argument('--compile-backbone', action='store_true', help='also compile the ViT backbone')
    p.add_argument('--fp8-quant', action='store_true', help='apply torchao FP8 dynamic quant to all nn.Linear')
    p.add_argument('--compile-mode', default='reduce-overhead',
                   choices=['default', 'reduce-overhead', 'max-autotune'])
    p.add_argument('--channels-last', action='store_true')
    p.add_argument('--profile', action='store_true', help='dump pytorch profiler trace')
    args = p.parse_args()

    torch.backends.cudnn.benchmark = True
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    # Let SDPA pick the fastest backend
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    except Exception:
        pass

    from engine.gtr.backbone.vit_adapter import _gla_cuda_ext
    print('cuda_gla extension:',
          'LOADED (hand-written CUDA chunk_gla active)' if _gla_cuda_ext is not None
          else 'NOT FOUND -> fla Triton fallback (~0.45ms slower on S); '
               'run: bash engine/gtr/backbone/csrc/build.sh')

    results = []
    for cfg_path in args.configs:
        name, s = benchmark_one(cfg_path, args)
        results.append((name, s))

    print("\n" + "=" * 72)
    print(f"Summary  (dtype={args.dtype} batch={args.batch} cuda-graph={args.cuda_graph})")
    print(f"  {'model':<14} {'min':>9} {'p50':>9} {'avg':>9} {'p95':>9} {'p99':>9}")
    for name, s in results:
        print(f"  {name:<14} {s['min']:>8.3f}ms {s['p50']:>8.3f}ms {s['avg']:>8.3f}ms "
              f"{s['p95']:>8.3f}ms {s['p99']:>8.3f}ms")
    # Verdict
    bad = [(n, s) for n, s in results if s['min'] > 4.0]
    if bad:
        print("\n[FAIL] models exceeding 4ms minimum latency:")
        for n, s in bad:
            print(f"    {n}: min={s['min']:.3f}ms")
        return 1
    print("\n[OK] all models <= 4.000ms minimum latency")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
