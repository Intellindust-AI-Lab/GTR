"""Attribute trtexec per-layer profile time to model regions using the layer-info metadata.

    trtexec --onnx=m.onnx --saveEngine=m_prof.engine --profilingVerbosity=detailed --skipInference ...
    trtexec --loadEngine=m_prof.engine --separateProfileRun --dumpProfile \
            --exportProfile=prof.json --exportLayerInfo=layers.json ...
    python profile_summary.py --profile prof.json --layers layers.json [--top 30]

Myelin ("kgen") layers carry the ONNX node names they fused in `Metadata`; each layer's
average time is split evenly across the regions those nodes belong to.
"""

import argparse
import json
import re
from collections import Counter, defaultdict

REGION_RULES = [
    (r'/backbone/backbone/patch_embed', 'backbone.patch_embed'),
    (r'/backbone/backbone/blocks\.\d+/attn/GatedLinearAttention', 'backbone.gla_plugin'),
    (r'/backbone/backbone/blocks\.\d+/attn/(qkv_g_gkl_proj|gk_high_proj)', 'backbone.attn.proj_gemm'),
    (r'/backbone/backbone/blocks\.\d+/attn/o_proj', 'backbone.attn.o_proj'),
    (r'/backbone/backbone/blocks\.\d+/attn/(Split|Reshape|Cast|Slice|Softplus|Neg|Div|Log|Sigmoid|Mul|Pow|ReduceMean|Add|Sqrt|Reciprocal|Sub)', 'backbone.attn.glue'),
    (r'/backbone/backbone/blocks\.\d+/attn', 'backbone.attn.other'),
    (r'/backbone/backbone/blocks\.\d+/mlp/dwconv', 'backbone.mlp.dwconv'),
    (r'/backbone/backbone/blocks\.\d+/mlp/(MatMul|MatMul_\d+|Gemm)', 'backbone.mlp.gemm'),
    (r'/backbone/backbone/blocks\.\d+/mlp', 'backbone.mlp.glue'),
    (r'/backbone/backbone/blocks\.\d+/norm', 'backbone.layernorm'),
    (r'/backbone/backbone/(Gather|Slice|Transpose|Reshape|Flatten)', 'backbone.bid_scan'),
    (r'/backbone/backbone/blocks\.\d+/Add', 'backbone.residual'),
    (r'/backbone', 'backbone.other'),
    (r'/encoder/stages_sampling', 'encoder.resample'),
    (r'/encoder/stages\.(\d+)', 'encoder.stage{0}'),
    (r'/encoder', 'encoder.other'),
    (r'/decoder/decoder/layers\.\d+/cross_attn', 'decoder.cross_attn'),
    (r'/decoder/decoder/layers\.\d+/self_attn|/decoder/decoder/layers\.\d+/(MatMul|Softmax|Transpose|Reshape|Add|Div|Mul|Split)', 'decoder.self_attn'),
    (r'/decoder/decoder/layers\.\d+/(linear|activation|gateway|norm)', 'decoder.ffn_gate_norm'),
    (r'/decoder/decoder/layers', 'decoder.layers.other'),
    (r'/decoder/(TopK|enc_score|enc_bbox|Gather|Where|Mul|Cast)', 'decoder.query_select'),
    (r'/decoder/decoder/(segmentation_head)', 'decoder.seg_head'),
    (r'/decoder/decoder/(dec_bbox|dec_score|lqe|bbox_head|score_head|integral|pre_bbox|query_pos|Softmax|MatMul|Sigmoid|Log|Clip|Sub|Add|Div|Mul|Concat|Reshape|Unsqueeze|Gather|Split|Slice|Squeeze|ReduceMean|Exp|TopK|Cos|Sin|Where|Cast|Transpose|Expand|GridSample|Pad)', 'decoder.heads_glue'),
    (r'/decoder', 'decoder.other'),
    (r'/head', 'head'),
]


def region_of(onnx_name):
    for pat, reg in REGION_RULES:
        m = re.search(pat, onnx_name)
        if m:
            return reg.format(*m.groups()) if m.groups() else reg
    return 'unattributed'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--profile', required=True)
    p.add_argument('--layers', required=True)
    p.add_argument('--top', type=int, default=30)
    p.add_argument('--real-ms', type=float, default=None, help='engine median from the timed run, to rescale')
    args = p.parse_args()

    prof = [x for x in json.load(open(args.profile)) if 'name' in x]
    info = json.load(open(args.layers))['Layers']
    meta = {l['Name']: l for l in info}
    total = sum(x['averageMs'] for x in prof)
    scale = (args.real_ms / total) if args.real_ms else 1.0
    unit = 'real-ms(est)' if args.real_ms else 'profile-ms'

    region_t = defaultdict(float)
    region_n = defaultdict(int)
    stream = Counter()
    rows = []
    for x in prof:
        name, t = x['name'], x['averageMs']
        l = meta.get(name, {})
        md = l.get('Metadata', '') or ''
        onnx_nodes = re.findall(r'\[ONNX Layer: ([^\]]+)\]', md)
        if not onnx_nodes:
            onnx_nodes = [name]
        regs = Counter(region_of(n) for n in onnx_nodes)
        for r, c in regs.items():
            region_t[r] += t * c / sum(regs.values())
        region_n[max(regs, key=regs.get)] += 1
        stream[l.get('StreamId', '?')] += t
        rows.append((t, name, l.get('LayerType', '?'), l.get('StreamId', '?'), onnx_nodes))

    print(f'{len(prof)} layers, sum(avg) = {total:.3f} profile-ms' + (f', rescaled to {args.real_ms:.3f} ms' if args.real_ms else ''))
    print(f'\n{"region":<28} {unit:>13} {"%":>6} {"#layers":>8}')
    for r, t in sorted(region_t.items(), key=lambda kv: -kv[1]):
        print(f'{r:<28} {t * scale:13.3f} {100 * t / total:6.1f} {region_n[r]:8d}')
    print('\nby TRT stream id:', {k: f'{v * scale:.3f}' for k, v in stream.items()})
    print(f'\nTop {args.top} layers:')
    for t, name, lt, sid, nodes in sorted(rows, key=lambda r: -r[0])[:args.top]:
        short = ', '.join(n.split('/')[-1] for n in nodes[:6]) + (' ...' if len(nodes) > 6 else '')
        print(f'  {t * scale:8.4f}  s{sid} {lt:<8} {name[:48]:<48} <- {short[:90]}')


if __name__ == '__main__':
    main()
