"""Isolated test of the GatedLinearAttention TensorRT plugin against the torch extension.

Builds a one-node ONNX graph (4-input and 6-input forms), compiles it with the TensorRT
Python API, runs it on the same random inputs as `gla_torch_ext.chunk_gla_run(_gated)` and
the fla reference, and reports cosine / max-abs.

    python test_plugin.py [--T 1600 --H 3] [--gk-normalizer 16]
"""

import argparse
import ctypes
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, '..')))
sys.path.insert(0, HERE)

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from onnx import TensorProto, helper, numpy_helper

PLUGIN = os.environ.get('GLA_PLUGIN', os.path.join(HERE, 'trt_plugin', 'libgla_plugin.so'))


def make_model(B, T, H, K, V, fused, scale, gk_normalizer, rms_eps, w=None):
    names = ['q', 'k', 'v', 'gk'] + (['g', 'rms_w'] if fused else [])
    inputs = [helper.make_tensor_value_info('q', TensorProto.FLOAT16, [B, T, H, K]),
              helper.make_tensor_value_info('k', TensorProto.FLOAT16, [B, T, H, K]),
              helper.make_tensor_value_info('v', TensorProto.FLOAT16, [B, T, H, V]),
              helper.make_tensor_value_info('gk', TensorProto.FLOAT16, [B, T, H, K])]
    inits = []
    if fused:
        inputs.append(helper.make_tensor_value_info('g', TensorProto.FLOAT16, [B, T, H, V]))
        inits.append(numpy_helper.from_array(w.cpu().numpy().astype(np.float16), 'rms_w'))
    node = helper.make_node('GatedLinearAttention', names, ['o'], scale=float(scale),
                            gk_normalizer=float(gk_normalizer), rms_eps=float(rms_eps))
    out = helper.make_tensor_value_info('o', TensorProto.FLOAT16, [B, T, H, V])
    graph = helper.make_graph([node], 'gla', inputs, [out], initializer=inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])


def build_engine(model_bytes, logger):
    import tensorrt as trt
    builder = trt.Builder(logger)
    flags = 0
    if int(trt.__version__.split('.')[0]) < 11:
        # strongly typed is the default (and only mode) from TensorRT 11 on; the plugin
        # only supports fp16, so a weakly typed network finds no valid format combination
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(model_bytes):
        for i in range(parser.num_errors):
            print('parser error:', parser.get_error(i))
        raise RuntimeError('parse failed')
    config = builder.create_builder_config()
    blob = builder.build_serialized_network(network, config)
    assert blob is not None, 'build failed'
    return bytes(blob)


def metrics(ref, got):
    r, g = ref.float().flatten(), got.float().flatten()
    return (F.cosine_similarity(r, g, dim=0).item(), ((g - r).norm() / (r.norm() + 1e-12)).item(),
            (g - r).abs().max().item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--B', type=int, default=1)
    p.add_argument('--T', type=int, default=1600)
    p.add_argument('--H', type=int, default=3)
    p.add_argument('--gk-normalizer', type=float, default=0.0, help='>0: feed raw gk and let the plugin fold logsigmoid')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    import tensorrt as trt
    from verify_trt import TRTRunner
    import gla_torch_ext as ext
    from fla.ops.gla import chunk_gla as fla_chunk_gla
    logger = trt.Logger(trt.Logger.WARNING)
    ctypes.CDLL(PLUGIN, mode=ctypes.RTLD_GLOBAL)

    torch.manual_seed(args.seed)
    B, T, H, K, V = args.B, args.T, args.H, ext.kK, ext.kV
    dev, dt = 'cuda', torch.float16
    q = torch.randn(B, T, H, K, device=dev, dtype=dt)
    k = torch.randn(B, T, H, K, device=dev, dtype=dt)
    v = torch.randn(B, T, H, V, device=dev, dtype=dt)
    gk_raw = torch.randn(B, T, H, K, device=dev, dtype=dt) * 0.5 - 1.0
    gk = F.logsigmoid(gk_raw.float()).half() / 16.0
    g = torch.randn(B, T, H, V, device=dev, dtype=dt)
    w = (torch.randn(V, device=dev, dtype=dt).abs() + 0.5)
    scale, eps = 1.0 / K ** 0.5, 1e-5
    gk_in = gk_raw if args.gk_normalizer > 0 else gk
    gk_norm = args.gk_normalizer

    o_fla, _ = fla_chunk_gla(q=q, k=k, v=v, g=gk, output_final_state=False)
    ws = ext.make_workspace(B, T, H, q.device)
    o_ext = torch.empty_like(v)
    ext.chunk_gla_run(q, k, v, gk_in, scale, ws, o_ext, gk_norm)
    y_ext = torch.empty_like(v)
    ext.chunk_gla_run_gated(q, k, v, gk_in, g, w, scale, eps, ws, y_ext, gk_norm)
    y_ref = (o_fla.float() * torch.rsqrt(o_fla.float().pow(2).mean(-1, keepdim=True) + eps) * w.float() * F.silu(g.float()))

    print(f'B={B} T={T} H={H} K={K} V={V} gk_normalizer={gk_norm}')
    for fused in (False, True):
        model = make_model(B, T, H, K, V, fused, scale, gk_norm, eps if fused else 0.0, w)
        path = os.path.join(HERE, 'out', f'plugin_test_{"6in" if fused else "4in"}.onnx')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        onnx.save(model, path)
        blob = build_engine(model.SerializeToString(), logger)
        eng_path = path.replace('.onnx', '.engine')
        with open(eng_path, 'wb') as f:
            f.write(blob)
        runner = TRTRunner(eng_path, plugin=None)
        feeds = dict(q=q, k=k, v=v, gk=gk_in)
        if fused:
            feeds['g'] = g
        out = runner.run(feeds)['o']
        if fused:
            print(f'[6-input fused] TRT vs ext.chunk_gla_run_gated: cos={metrics(y_ext, out)[0]:.6f} rel={metrics(y_ext, out)[1]:.3e} max_abs={metrics(y_ext, out)[2]:.4g}')
            print(f'[6-input fused] TRT vs fla+manual gate        : cos={metrics(y_ref, out)[0]:.6f} rel={metrics(y_ref, out)[1]:.3e}')
            print(f'[6-input fused] ext vs fla+manual gate        : cos={metrics(y_ref, y_ext)[0]:.6f} rel={metrics(y_ref, y_ext)[1]:.3e}')
        else:
            print(f'[4-input] TRT vs ext.chunk_gla_run : cos={metrics(o_ext, out)[0]:.6f} rel={metrics(o_ext, out)[1]:.3e} max_abs={metrics(o_ext, out)[2]:.4g}')
            print(f'[4-input] TRT vs fla               : cos={metrics(o_fla, out)[0]:.6f} rel={metrics(o_fla, out)[1]:.3e}')
            print(f'[4-input] ext vs fla               : cos={metrics(o_fla, o_ext)[0]:.6f} rel={metrics(o_fla, o_ext)[1]:.3e}')
        # run twice more: a workspace/flag problem shows up as run-to-run differences
        out2 = runner.run(feeds)['o']
        print(f'   run-to-run max_abs diff: {(out2.float() - out.float()).abs().max().item():.4g}')


if __name__ == '__main__':
    main()
