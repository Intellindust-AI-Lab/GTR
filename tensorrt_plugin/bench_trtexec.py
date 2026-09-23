"""Build a TensorRT engine from an ONNX file with trtexec and benchmark it.

TensorRT 11 semantics: every network is strongly typed (the ONNX dtypes decide the
precision), CUDA graph + spin-wait are on and H2D/D2H transfers are excluded by default,
so the reported "GPU Compute Time" is the pure engine latency.

    python bench_trtexec.py --onnx out/onnx/gtr_s_fp16.onnx --json out/results/gtr_s_fp16.json
    python bench_trtexec.py --onnx out/onnx/gtr_s_fp32.onnx --extra --noTF32 --tag notf32
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TRT_ROOT = os.environ.get('TRT_ROOT', os.path.join(HERE, 'tensorrt'))
CUDA_HOME = os.environ.get('CUDA_HOME', '/usr/local/cuda-12.8')
TRTEXEC = os.environ.get('TRTEXEC', os.path.join(TRT_ROOT, 'bin', 'trtexec'))
PLUGIN = os.environ.get('GLA_PLUGIN', os.path.join(HERE, 'trt_plugin', 'libgla_plugin.so'))


def trt_major():
    """Major version of the TensorRT install in TRT_ROOT (decides which trtexec defaults exist)."""
    with open(os.path.join(TRT_ROOT, 'include', 'NvInferVersion.h')) as f:
        return int(re.search(r'#define NV_TENSORRT_MAJOR (\d+)', f.read()).group(1))


def trt_env():
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = ':'.join(
        p for p in [os.path.join(TRT_ROOT, 'lib'), os.path.join(CUDA_HOME, 'lib64'), env.get('LD_LIBRARY_PATH', '')] if p)
    return env


def parse_trtexec(text):
    """Pull the timing summary out of trtexec's log."""
    res = {}
    m = re.search(r'GPU Compute Time: min = ([\d.]+) ms, max = ([\d.]+) ms, mean = ([\d.]+) ms, median = ([\d.]+) ms, '
                  r'percentile\(90%\) = ([\d.]+) ms, percentile\(95%\) = ([\d.]+) ms, percentile\(99%\) = ([\d.]+) ms', text)
    if m:
        keys = ['min', 'max', 'mean', 'median', 'p90', 'p95', 'p99']
        res['gpu_compute_ms'] = {k: float(v) for k, v in zip(keys, m.groups())}
    m = re.search(r'Throughput: ([\d.]+) qps', text)
    if m:
        res['throughput_qps'] = float(m.group(1))
    m = re.search(r'Latency: min = ([\d.]+) ms, max = ([\d.]+) ms, mean = ([\d.]+) ms, median = ([\d.]+) ms', text)
    if m:
        res['host_latency_ms'] = dict(zip(['min', 'max', 'mean', 'median'], map(float, m.groups())))
    m = re.search(r'Total GPU Compute Time: ([\d.]+) s', text)
    if m:
        res['total_gpu_compute_s'] = float(m.group(1))
    m = re.search(r'Engine built in ([\d.]+) sec', text)
    if m:
        res['build_s'] = float(m.group(1))
    m = re.search(r'Engine deserialized in ([\d.]+) sec', text)
    if m:
        res['deserialize_s'] = float(m.group(1))
    m = re.search(r'Total Host Walltime: ([\d.]+) s', text)
    if m:
        res['walltime_s'] = float(m.group(1))
    m = re.search(r'Total Inferences: (\d+)', text) or re.search(r'Total (\d+) queries', text)
    if m:
        res['inferences'] = int(m.group(1))
    res['ok'] = '&&&& PASSED' in text
    return res


def run_trtexec(onnx_path=None, engine_path=None, load_engine=False, warmup_ms=2000, duration_s=10, iterations=500,
                extra=(), plugin=PLUGIN, log_path=None, opt_level=None, verbose=False):
    cmd = [TRTEXEC]
    if load_engine:
        cmd.append(f'--loadEngine={engine_path}')
    else:
        cmd.append(f'--onnx={onnx_path}')
        if engine_path:
            cmd.append(f'--saveEngine={engine_path}')
        if opt_level is not None:
            cmd.append(f'--builderOptimizationLevel={opt_level}')
    if plugin:
        cmd.append(f'--staticPlugins={plugin}')
    cmd += [f'--warmUp={warmup_ms}', f'--duration={duration_s}', f'--iterations={iterations}']
    if trt_major() < 11:
        # TensorRT 11 makes all of these the default: strongly typed networks, CUDA graph,
        # spin-wait and no H2D/D2H inside the timing loop. Spell them out on 10.x so the
        # engines and the reported GPU Compute Time mean the same thing.
        if not load_engine:
            cmd.append('--stronglyTyped')
        cmd += ['--useCudaGraph', '--useSpinWait', '--noDataTransfers']
    if verbose:
        cmd.append('--verbose')
    cmd += list(extra)
    t0 = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=trt_env())
    text = proc.stdout
    if log_path:
        with open(log_path, 'w') as f:
            f.write(' '.join(cmd) + '\n\n' + text)
    res = parse_trtexec(text)
    res.update(cmd=' '.join(cmd), returncode=proc.returncode, wall_s=time.time() - t0)
    if proc.returncode != 0 or not res.get('ok'):
        tail = '\n'.join(text.strip().splitlines()[-25:])
        res['error_tail'] = tail
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--onnx', required=True)
    p.add_argument('--engine', default=None, help='engine path (default: alongside the ONNX, .engine)')
    p.add_argument('--tag', default='', help='suffix for engine/log/json names')
    p.add_argument('--json', default=None)
    p.add_argument('--warmup-ms', type=int, default=2000)
    p.add_argument('--duration', type=int, default=10)
    p.add_argument('--iterations', type=int, default=500)
    p.add_argument('--opt-level', type=int, default=None)
    p.add_argument('--no-plugin', action='store_true')
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--extra', nargs=argparse.REMAINDER, default=[], help='extra trtexec flags (after --extra)')
    args = p.parse_args()

    stem = os.path.splitext(args.onnx)[0] + (f'_{args.tag}' if args.tag else '')
    engine = args.engine or (stem + '.engine')
    log = os.path.splitext(engine)[0] + '.trtexec.log'   # next to the engine, not the ONNX
    res = run_trtexec(args.onnx, engine, warmup_ms=args.warmup_ms, duration_s=args.duration,
                      iterations=args.iterations, extra=args.extra, plugin=None if args.no_plugin else PLUGIN,
                      log_path=log, opt_level=args.opt_level, verbose=args.verbose)
    res.update(onnx=args.onnx, engine=engine, log=log, tag=args.tag)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, 'w') as f:
            json.dump(res, f, indent=2)
    t = res.get('gpu_compute_ms')
    if t:
        print(f"[bench] {os.path.basename(args.onnx)} {args.tag}: GPU compute min={t['min']:.3f} mean={t['mean']:.3f} "
              f"median={t['median']:.3f} p99={t['p99']:.3f} ms  (build {res.get('build_s', float('nan')):.0f}s, "
              f"{res.get('inferences', '?')} inferences)")
    else:
        print(f"[bench] FAILED: {os.path.basename(args.onnx)}\n{res.get('error_tail', '')}")
        sys.exit(1)


if __name__ == '__main__':
    main()
