"""Collect out/results/*.json (bench + verify) into a markdown summary table.

    python make_report.py --results out/results --out out/RESULTS.md
"""

import argparse
import glob
import json
import os
import re

TASK_OF = {'gtr': 'det', 'gtrseg': 'seg', 'gtrobb': 'obb', 'gtrpose': 'pose', 'gtrsemseg': 'semseg', 'gtrdepth': 'depth'}
TASK_ORDER = ['det', 'seg', 'obb', 'pose', 'semseg', 'depth']
SIZE_ORDER = ['s', 'm', 'l', 'x']

# torch.compile(fullgraph=True) + one CUDA graph, fp16, median ms (gtr_latency_results/README.md)
TORCH_FP16_MEDIAN = {
    'gtr_s': 1.225, 'gtr_m': 1.462, 'gtr_l': 1.908, 'gtr_x': 2.115,
    'gtrseg_s': 1.465, 'gtrseg_m': 1.807, 'gtrseg_l': 2.250, 'gtrseg_x': 2.472,
    'gtrobb_s': 1.946, 'gtrobb_m': 2.496, 'gtrobb_l': 3.493, 'gtrobb_x': 4.060,
    'gtrpose_s': 1.455, 'gtrpose_m': 1.860, 'gtrpose_l': 2.328, 'gtrpose_x': 2.590,
    'gtrsemseg_s': 1.495, 'gtrsemseg_m': 1.970, 'gtrsemseg_l': 2.958, 'gtrsemseg_x': 3.482,
    'gtrdepth_s': 1.296, 'gtrdepth_m': 1.508, 'gtrdepth_l': 1.948, 'gtrdepth_x': 2.158,
}


def parse_tag(tag):
    m = re.match(r'^(gtr(?:seg|obb|pose|semseg|depth)?)_([smlx])_(fp16|fp32)(?:_(.+))?$', tag)
    if not m:
        return None
    fam, size, dtype, variant = m.groups()
    return dict(family=fam, task=TASK_OF[fam], size=size, name=f'{fam}_{size}', dtype=dtype, variant=variant or '')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out', 'results'))
    p.add_argument('--out', default=None)
    p.add_argument('--variant', default='', help='which variant to tabulate as the main result ("" = baseline)')
    args = p.parse_args()

    rows = {}
    for path in sorted(glob.glob(os.path.join(args.results, '*.json'))):
        if path.endswith('.verify.json'):
            continue
        tag = os.path.splitext(os.path.basename(path))[0]
        info = parse_tag(tag)
        if not info or info['variant'] != args.variant:
            continue
        with open(path) as f:
            r = json.load(f)
        if 'gpu_compute_ms' not in r:
            continue
        vpath = path[:-5] + '.verify.json'
        ver = None
        if os.path.exists(vpath):
            with open(vpath) as f:
                ver = json.load(f)
        rows[(info['name'], info['dtype'])] = dict(info=info, bench=r, verify=ver)

    lines = ['| task | model | input | dtype | min | mean | median | p99 | build(s) | opt | engine(MB) | verify sorted-cos | torch fp16 median | TRT/torch |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|---:|---:|']
    names = sorted({k[0] for k in rows}, key=lambda n: (TASK_ORDER.index(TASK_OF[n.rsplit('_', 1)[0]]), SIZE_ORDER.index(n[-1])))
    for name in names:
        for dtype in ('fp32', 'fp16'):
            row = rows.get((name, dtype))
            if not row:
                continue
            b = row['bench']
            t = b['gpu_compute_ms']
            ex = b.get('export', {})
            eval_size = ex.get('eval_size', ['?', '?'])
            engine_mb = os.path.getsize(b['engine']) / 1e6 if os.path.exists(b.get('engine', '')) else float('nan')
            cos = ''
            if row['verify']:
                rep = row['verify'].get('report', {})
                key = 'trt_vs_torch_fp16' if dtype == 'fp16' and 'trt_vs_torch_fp16' in rep else 'trt_vs_torch_fp32'
                vals = [v.get('sorted_cos', v.get('cos')) for v in rep.get(key, {}).values()]
                if vals:
                    cos = '**NaN**' if any(v != v for v in vals) else f"{min(vals):.4f}"
            torch_ms = TORCH_FP16_MEDIAN.get(name)
            ratio = f"{t['median'] / torch_ms:.2f}x" if (torch_ms and dtype == 'fp16') else ''
            opt = b.get('opt_level_used', b.get('opt_level'))
            opt = '' if opt is None else (f"{opt}↓" if b.get('fallback_attempt') else f"{opt}")
            lines.append(f"| {row['info']['task']} | `{name}` | {eval_size[0]}x{eval_size[1]} | {dtype} | {t['min']:.3f} | {t['mean']:.3f} | "
                         f"{t['median']:.3f} | {t['p99']:.3f} | {b.get('build_s', float('nan')):.0f} | {opt} | {engine_mb:.0f} | {cos} | "
                         f"{torch_ms if torch_ms else ''} | {ratio} |")
    text = '\n'.join(lines)

    # A/B variants (tags with a suffix), grouped by model/dtype, sorted by median.
    variants = {}
    for path in sorted(glob.glob(os.path.join(args.results, '*.json'))):
        if path.endswith('.verify.json'):
            continue
        tag = os.path.splitext(os.path.basename(path))[0]
        info = parse_tag(tag)
        if not info or not info['variant']:
            continue
        with open(path) as f:
            r = json.load(f)
        if 'gpu_compute_ms' in r:
            variants.setdefault((info['name'], info['dtype']), []).append((info['variant'], r))
    if variants:
        text += ('\n\n### A/B variants (same model, trtexec GPU Compute median ms; NaN in the verify column = the engine '
                 'emits NaN, see the TensorRT Pad+Add+GridSample fusion bug in README.md)\n\n'
                 '| model | dtype | variant | min | median | p99 | build(s) | verify sorted-cos |\n|---|---|---|---:|---:|---:|---:|---|\n')
        for (name, dtype), items in sorted(variants.items()):
            for variant, r in sorted(items, key=lambda kv: kv[1]['gpu_compute_ms']['median']):
                t = r['gpu_compute_ms']
                cos = ''
                vpath = os.path.join(args.results, f'{name}_{dtype}_{variant}.verify.json')
                if os.path.exists(vpath):
                    with open(vpath) as f:
                        rep = json.load(f).get('report', {})
                    key = 'trt_vs_torch_fp16' if dtype == 'fp16' and 'trt_vs_torch_fp16' in rep else 'trt_vs_torch_fp32'
                    vals = [v.get('sorted_cos', v.get('cos')) for v in rep.get(key, {}).values()]
                    if vals:
                        cos = '**NaN**' if any(v != v for v in vals) else f"{min(vals):.4f}"
                text += (f"| `{name}` | {dtype} | {variant} | {t['min']:.3f} | {t['median']:.3f} | {t['p99']:.3f} | "
                         f"{r.get('build_s', float('nan')):.0f} | {cos} |\n")
    print(text)
    if args.out:
        with open(args.out, 'w') as f:
            f.write(text + '\n')


if __name__ == '__main__':
    main()
