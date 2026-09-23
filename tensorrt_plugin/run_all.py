"""Export -> build -> benchmark every GTR model (6 tasks x s/m/l/x) in fp32 and fp16.

    python run_all.py --all                       # 24 models, fp32 + fp16
    python run_all.py --tasks det --sizes s m     # subset
    python run_all.py --configs ../configs/det/coco_finetune/gtr_s.yml --dtypes fp16 \
        --variant fused --export-args "--no-static-perm"      # named A/B variant
    python run_all.py --all --skip-export         # reuse existing ONNX files

Every (model, dtype, variant) produces out/onnx/<name>_<dtype>[_<variant>].onnx, an engine
under out/engines/, a trtexec log next to the ONNX and a JSON under out/results/. The
summary table is regenerated from all JSON files by make_report.py.
"""

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
TASKS = {
    'det': '../configs/det/coco_finetune/gtr_{}.yml',
    'seg': '../configs/seg/coco_seg_finetune/gtrseg_{}.yml',
    'obb': '../configs/obb/dota_finetune/gtrobb_{}.yml',
    'pose': '../configs/pose/coco_pose_finetune/gtrpose_{}.yml',
    'semseg': '../configs/semseg/cityscapes_finetune/gtrsemseg_{}.yml',
    'depth': '../configs/depth/pretrain/gtrdepth_{}.yml',
}
SIZES = ('s', 'm', 'l', 'x')


def sh(cmd, log=None, timeout=None):
    print('$', ' '.join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout,
                          env=dict(os.environ, PATH='/usr/local/cuda-12.8/bin:' + os.environ.get('PATH', '')))
    if log:
        with open(log, 'w') as f:
            f.write(' '.join(cmd) + '\n\n' + proc.stdout)
    return proc


def verify_min_cos(path, dtype):
    """Smallest (sorted-)cosine over the engine's outputs vs the matching PyTorch reference
    (torch fp16 for fp16 engines, torch fp32 otherwise). NaN means the engine emits NaN."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rep = json.load(f)['report']
    key = 'trt_vs_torch_fp16' if dtype == 'fp16' and 'trt_vs_torch_fp16' in rep else 'trt_vs_torch_fp32'
    vals = [v.get('sorted_cos', v['cos']) for v in rep.get(key, {}).values()]
    return min(vals) if vals else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--configs', nargs='*', default=[])
    p.add_argument('--all', action='store_true')
    p.add_argument('--tasks', nargs='*', default=None)
    p.add_argument('--sizes', nargs='*', default=list(SIZES))
    p.add_argument('--dtypes', nargs='*', default=['fp32', 'fp16'])
    p.add_argument('--variant', default='', help='name suffix for an A/B variant')
    p.add_argument('--export-args', default='', help='extra args for export_onnx.py (quoted string)')
    p.add_argument('--trt-args', default='', help='extra trtexec flags (quoted string)')
    p.add_argument('--opt-level', type=int, default=None)
    p.add_argument('--skip-export', action='store_true')
    p.add_argument('--skip-bench', action='store_true')
    p.add_argument('--verify', action='store_true', help='compare every engine against PyTorch (verify_trt.py)')
    p.add_argument('--fallback', action='store_true',
                   help='if verification fails (NaN or min cos < 0.99: TensorRT 11.2 Myelin data-movement '
                        'fusion corrupts buffers in some model x build-flag combos), rebuild the engine at '
                        'lower opt levels until it verifies: opt4 -> opt3 -> opt3 without extra trt args')
    p.add_argument('--skip-done', action='store_true',
                   help='skip a (model, dtype) whose result JSON already has the same export/trt args and a '
                        'passing verification')
    p.add_argument('--warmup-ms', type=int, default=2000)
    p.add_argument('--duration', type=int, default=10)
    p.add_argument('--iterations', type=int, default=500)
    p.add_argument('--out', default=os.path.join(HERE, 'out'))
    args = p.parse_args()

    configs = list(args.configs)
    if args.all or args.tasks:
        for task in (args.tasks or list(TASKS)):
            configs += [TASKS[task].format(s) for s in args.sizes]
    if not configs:
        p.error('give --configs, --tasks or --all')

    for d in ('onnx', 'engines', 'results', 'logs'):
        os.makedirs(os.path.join(args.out, d), exist_ok=True)

    summary = []
    for cfg in configs:
        name = os.path.splitext(os.path.basename(cfg))[0]
        for dtype in args.dtypes:
            tag = f'{name}_{dtype}' + (f'_{args.variant}' if args.variant else '')
            onnx_path = os.path.join(args.out, 'onnx', f'{tag}.onnx')
            engine = os.path.join(args.out, 'engines', f'{tag}.engine')
            result = os.path.join(args.out, 'results', f'{tag}.json')
            print(f"\n{'=' * 78}\n>>> {tag}")
            if args.skip_done and os.path.exists(result):
                with open(result) as f:
                    prev = json.load(f)
                v = prev.get('verify_min_cos')
                if (prev.get('export_args', '') == args.export_args and prev.get('trt_args', '') == args.trt_args
                        and v is not None and v == v and v >= 0.99):
                    print(f"[run_all] SKIP {tag}: already done (verify {v:.4f})")
                    summary.append(dict(tag=tag, verify=v, **prev['gpu_compute_ms']))
                    continue
            t0 = time.time()
            if not args.skip_export or not os.path.exists(onnx_path):
                cmd = [PY, 'export_onnx.py', '-c', cfg, '-o', onnx_path, '--dtype', dtype] + args.export_args.split()
                proc = sh(cmd, log=os.path.join(args.out, 'logs', f'{tag}.export.log'), timeout=3600)
                if proc.returncode != 0:
                    print(f'[run_all] EXPORT FAILED for {tag}:\n' + '\n'.join(proc.stdout.splitlines()[-15:]))
                    summary.append(dict(tag=tag, error='export'))
                    continue
                print('\n'.join(l for l in proc.stdout.splitlines() if l.startswith('[')))
            if args.skip_bench:
                continue
            attempts = [(args.opt_level, args.trt_args)]
            if args.fallback:
                for alt in [(4, args.trt_args), (3, args.trt_args), (3, '')]:
                    if alt not in attempts:
                        attempts.append(alt)
            r = None
            for n_try, (opt_level, trt_args) in enumerate(attempts):
                if n_try:
                    print(f"[run_all] FALLBACK {tag}: rebuilding with opt_level={opt_level} trt_args='{trt_args}'")
                cmd = [PY, 'bench_trtexec.py', '--onnx', onnx_path, '--engine', engine, '--json', result,
                       '--warmup-ms', str(args.warmup_ms), '--duration', str(args.duration),
                       '--iterations', str(args.iterations)]
                if opt_level is not None:
                    cmd += ['--opt-level', str(opt_level)]
                if trt_args:
                    cmd += ['--extra'] + trt_args.split()
                proc = sh(cmd, timeout=7200)
                print(proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else '(no output)')
                if proc.returncode != 0:
                    r = None
                    continue
                with open(result) as f:
                    r = json.load(f)
                r.update(name=name, dtype=dtype, variant=args.variant, config=cfg, export_args=args.export_args,
                         trt_args=args.trt_args, opt_level=args.opt_level, opt_level_used=opt_level,
                         trt_args_used=trt_args, fallback_attempt=n_try, total_s=time.time() - t0)
                meta_path = os.path.splitext(onnx_path)[0] + '.meta.json'
                if os.path.exists(meta_path):
                    with open(meta_path) as f:
                        r['export'] = json.load(f)
                with open(result, 'w') as f:
                    json.dump(r, f, indent=2)
                if not args.verify:
                    break
                vcmd = [PY, 'verify_trt.py', '-c', cfg, '--engine', engine, '--json', result[:-5] + '.verify.json']
                if dtype == 'fp16':
                    vcmd.append('--ref-fp16')
                for a in args.export_args.split():
                    if a == '--resume':
                        vcmd += ['--resume', args.export_args.split()[args.export_args.split().index(a) + 1]]
                    if a == '--part':
                        vcmd += ['--part', args.export_args.split()[args.export_args.split().index(a) + 1]]
                vproc = sh(vcmd, log=os.path.join(args.out, 'logs', f'{tag}.verify.log'), timeout=1800)
                print('\n'.join(l for l in vproc.stdout.splitlines() if l.startswith(('[verify]', '  '))))
                r['verify_min_cos'] = verify_min_cos(result[:-5] + '.verify.json', dtype)
                print(f"[run_all] verify {tag}: min cos = {r['verify_min_cos']} "
                      f"(opt_level={opt_level}, trt_args='{trt_args}')")
                with open(result, 'w') as f:
                    json.dump(r, f, indent=2)
                v = r['verify_min_cos']
                if v is None or (v == v and v >= 0.99):
                    break
            if r is None:
                summary.append(dict(tag=tag, error='bench'))
                continue
            summary.append(dict(tag=tag, verify=r.get('verify_min_cos'), **r['gpu_compute_ms']))

    print(f"\n{'=' * 78}\nSummary (trtexec GPU Compute Time, ms; verify = min sorted-cos vs PyTorch, NaN = broken engine)")
    print(f"  {'model':<34} {'min':>8} {'mean':>8} {'median':>8} {'p99':>8} {'verify':>8}")
    for s in summary:
        if 'error' in s:
            print(f"  {s['tag']:<34} ERROR ({s['error']})")
        else:
            v = s.get('verify')
            print(f"  {s['tag']:<34} {s['min']:>8.3f} {s['mean']:>8.3f} {s['median']:>8.3f} {s['p99']:>8.3f} "
                  f"{'' if v is None else f'{v:8.4f}':>8}")


if __name__ == '__main__':
    main()
