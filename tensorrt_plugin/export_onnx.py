"""Export a GTR model (backbone + encoder + task head, raw outputs) to ONNX.

The hand-written chunk-GLA kernel becomes one custom node per ViT block
(`GatedLinearAttention`, served by trt_plugin/libgla_plugin.so in TensorRT);
everything else is standard ONNX (opset 17: LayerNormalization, GridSample, TopK ...).

Examples:
    python export_onnx.py -c ../configs/det/coco_finetune/gtr_s.yml -o out/gtr_s_fp32.onnx
    python export_onnx.py -c ../configs/det/coco_finetune/gtr_s.yml -o out/gtr_s_fp16.onnx --dtype fp16
    python export_onnx.py -c ../configs/depth/pretrain/gtrdepth_l.yml --resume ../weights/depth/gtrdepth_l.pth ...
"""

import argparse
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
GTR_ROOT = os.path.normpath(os.path.join(HERE, '..'))
sys.path.insert(0, GTR_ROOT)
sys.path.insert(0, HERE)

import torch
import torch.nn as nn

from engine.core import YAMLConfig
from gtr_onnx_patches import CUSTOM_DOMAIN, GLA_OP, TASK_OUTPUTS, prepare_for_export

BACKBONES = ('ViTAdapter', 'ViTAdapterSpatialSwiGLU')
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def build_model(config_path, backbone='ViTAdapterSpatialSwiGLU', resume=None, with_postproc=False):
    cfg = YAMLConfig(config_path)
    y = cfg.yaml_cfg
    model_key = y['model']
    if backbone and y[model_key].get('backbone') != backbone:
        # Same config keys for both flavours: carry the size-specific section over.
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
    if resume:
        ck = torch.load(resume, map_location='cpu', weights_only=False)
        state = (ck.get('ema') or {}).get('module') or ck['model']
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f'[info] loaded {resume}: missing={len(missing)} unexpected={len(unexpected)}')
        if missing:
            print('   missing (first 5):', missing[:5])
    model = model.deploy()
    post = cfg.postprocessor.deploy() if with_postproc else None
    return model, post, y['task'], tuple(y['eval_spatial_size'])


class RawWrapper(nn.Module):
    def __init__(self, model, task, uint8_input=False):
        super().__init__()
        self.model = model
        self.keys = TASK_OUTPUTS[task]
        self.uint8_input = uint8_input
        if uint8_input:
            mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
            std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
            self.register_buffer('pre_scale', (1.0 / 255.0) / std)
            self.register_buffer('pre_bias', -mean / std)

    def forward(self, images):
        if self.uint8_input:
            images = images.to(self.pre_scale.dtype) * self.pre_scale + self.pre_bias
        out = self.model(images)
        return tuple(out[k] for k in self.keys)


def run_part(model, part, images):
    """Sub-model forward shared by export (patched model) and verification (reference model):
    'backbone' -> ViT features, 'encoder' -> encoder pyramid, 'blocks:N' -> token tensor after
    patch_embed + the first N ViT blocks (+ bid_scan), for numerical bisection."""
    if part.startswith('blocks:'):
        n = int(part.split(':')[1])
        vit = model.backbone.backbone
        x = vit._model.patch_embed(images)
        for i, blk in enumerate(vit._model.blocks[:n]):
            x = blk(x)
            if vit.bid_scan:
                x = vit._apply_bid_scan(x, i)
        return (x,)
    feats = model.backbone(images)
    if part == 'backbone':
        return tuple(feats)
    return tuple(model.encoder(feats))


class PartWrapper(nn.Module):
    def __init__(self, model, part):
        super().__init__()
        self.model = model
        self.part = part

    def forward(self, images):
        return run_part(self.model, self.part, images)


class PostprocWrapper(RawWrapper):
    def __init__(self, model, post, task, uint8_input=False):
        super().__init__(model, task, uint8_input)
        self.post = post

    def forward(self, images, orig_target_sizes):
        if self.uint8_input:
            images = images.to(self.pre_scale.dtype) * self.pre_scale + self.pre_bias
        return self.post(self.model(images), orig_target_sizes)


POSTPROC_OUTPUTS = {
    'detection': ['labels', 'boxes', 'scores'],
    'segmentation': ['labels', 'boxes', 'scores', 'masks'],
    'obb': ['labels', 'boxes', 'scores'],
    'pose': ['scores', 'labels', 'keypoints'],
}


def _infer_shapes(m):
    import onnx
    # The exporter records the custom node's output shape in value_info, so ONNX shape
    # inference (non-strict) walks straight through it. Full shape coverage lets onnxsim
    # fold every residual Shape/Gather chain (the tracer's dynamic reshapes).
    return onnx.shape_inference.infer_shapes(m, strict_mode=False, data_prop=True)


def _drop_noop_casts(m):
    """Remove Cast nodes whose input already has the target dtype (fp16 export: the fp16
    casts around the plugin become identities). Returns the number removed."""
    import onnx
    m2 = _infer_shapes(m)
    types = {v.name: v.type.tensor_type.elem_type for v in list(m2.graph.value_info) + list(m2.graph.input)}
    for t in m2.graph.initializer:
        types[t.name] = t.data_type
    rename, keep, removed = {}, [], 0
    for node in m.graph.node:
        if node.op_type == 'Cast':
            to = next(a.i for a in node.attribute if a.name == 'to')
            if types.get(node.input[0]) == to:
                rename[node.output[0]] = node.input[0]
                removed += 1
                continue
        keep.append(node)
    if not removed:
        return 0

    def resolve(name):
        while name in rename:
            name = rename[name]
        return name

    for node in keep:
        for i, x in enumerate(node.input):
            node.input[i] = resolve(x)
    extra = [onnx.helper.make_node('Identity', [resolve(o.name)], [o.name])
             for o in m.graph.output if o.name in rename]
    del m.graph.node[:]
    m.graph.node.extend(keep + extra)
    return removed


def finalize_onnx(path, simplify=True, check=True):
    import onnx
    m = onnx.load(path)
    print(f'[info] raw graph: {len(m.graph.node)} nodes, {len(m.graph.initializer)} initializers')
    if simplify:
        import onnxsim
        try:
            m2, ok = onnxsim.simplify(_infer_shapes(m))
        except Exception as e:  # shape inference through the custom op can trip older onnxsim
            print(f'[warn] onnxsim failed ({e!r}); retrying with skip_shape_inference=True')
            m2, ok = onnxsim.simplify(m, skip_shape_inference=True)
        if ok:
            m = m2
            print(f'[ok] onnxsim: {len(m.graph.node)} nodes')
        else:
            print('[warn] onnxsim validation failed; keeping the raw graph')
        removed = _drop_noop_casts(m)
        if removed:
            m2, ok = onnxsim.simplify(_infer_shapes(m))
            if ok:
                m = m2
            print(f'[ok] dropped {removed} no-op Cast nodes: {len(m.graph.node)} nodes')
    # TensorRT resolves plugins from the default ("") domain: rename the custom nodes and
    # drop the custom opset import.
    n = 0
    for node in m.graph.node:
        if node.op_type == GLA_OP and node.domain == CUSTOM_DOMAIN:
            node.domain = ''
            n += 1
    kept = [o for o in m.opset_import if o.domain != CUSTOM_DOMAIN]
    del m.opset_import[:]
    m.opset_import.extend(kept)
    onnx.save(m, path)
    hist = Counter(node.op_type for node in m.graph.node)
    print(f'[info] {GLA_OP} nodes: {n}; final graph {len(m.graph.node)} nodes')
    print('[info] op histogram:', ', '.join(f'{k}:{v}' for k, v in hist.most_common(25)))
    if check:
        # The custom op lives in the default domain now, so the checker cannot validate the
        # node itself; validate the rest by temporarily moving it to a custom opset.
        tmp = onnx.load(path)
        for node in tmp.graph.node:
            if node.op_type == GLA_OP:
                node.domain = CUSTOM_DOMAIN
        tmp.opset_import.append(onnx.helper.make_opsetid(CUSTOM_DOMAIN, 1))
        onnx.checker.check_model(tmp)
        print('[ok] onnx.checker passed')
    return dict(nodes=len(m.graph.node), gla_nodes=n, ops=dict(hist))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-c', '--config', required=True)
    p.add_argument('-o', '--output', required=True)
    p.add_argument('--backbone', default='ViTAdapterSpatialSwiGLU', choices=list(BACKBONES) + ['config'],
                   help="backbone flavour ('config' keeps the config's own)")
    p.add_argument('--resume', default=None, help='checkpoint (.pth); random init otherwise')
    p.add_argument('--dtype', default='fp32', choices=['fp32', 'fp16'],
                   help='export the model in this dtype (fp16 = weights + activations fp16)')
    p.add_argument('--opset', type=int, default=17)
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--no-fuse-gate', action='store_true',
                   help='keep RMSNorm+swish gate as ONNX ops instead of fusing into the plugin')
    p.add_argument('--gk-in-plugin', action='store_true',
                   help='fold logsigmoid/normalizer of gk into the plugin (kernel gk_normalizer path)')
    p.add_argument('--no-static-perm', action='store_true',
                   help='keep bid_scan as Slice/Transpose instead of a Gather with a static index')
    p.add_argument('--split-qkv', action='store_true',
                   help='one GEMM per q/k/v/g/gk_low projection instead of one fused GEMM + 5 slices')
    p.add_argument('--no-fold-gk', action='store_true',
                   help='keep the rank-16 gk projection as its own GEMM instead of composing it into the fused GEMM')
    p.add_argument('--ln2d-reduce', action='store_true',
                   help='export encoder LayerNorm2d as channel-axis reductions (no Transpose pair)')
    p.add_argument('--atlas', default='none', choices=['none', 'concat', 'pad'],
                   help="det/seg MSDA value atlas: none (default; one GridSample per level - any in-graph "
                        "canvas construction hits a TensorRT 11.2 Myelin fusion bug -> NaN in some builds), "
                        "concat or pad (both affected)")
    p.add_argument('--part', default='full',
                   help="'full' | 'backbone' (ViT features) | 'encoder' (pyramid) | 'blocks:N' (patch_embed + N ViT blocks)")
    p.add_argument('--postproc', action='store_true', help='append the task postprocessor (det/seg/obb/pose)')
    p.add_argument('--uint8-input', action='store_true', help='uint8 NCHW input with on-device normalisation')
    p.add_argument('--no-simplify', action='store_true')
    p.add_argument('--no-check', action='store_true')
    p.add_argument('--meta', default=None, help='write export metadata json here')
    p.add_argument('--seed', type=int, default=0,
                   help='RNG seed for the random-init weights (verify_trt.py rebuilds the model with the same seed)')
    args = p.parse_args()

    backbone = None if args.backbone == 'config' else args.backbone
    torch.manual_seed(args.seed)
    model, post, task, (H, W) = build_model(args.config, backbone, args.resume, args.postproc)
    dtype = torch.float16 if args.dtype == 'fp16' else torch.float32
    device = torch.device('cuda')
    model = model.to(device=device, dtype=dtype).eval()
    if post is not None:
        post = post.to(device).eval()
    n_gla = prepare_for_export(model, (H, W), fuse_gate=not args.no_fuse_gate,
                               gk_in_plugin=args.gk_in_plugin, static_perm=not args.no_static_perm,
                               split_qkv=args.split_qkv, fold_gk=not args.no_fold_gk,
                               ln2d_reduce=args.ln2d_reduce, atlas=args.atlas)
    print(f'[info] task={task} eval={H}x{W} dtype={args.dtype} patched {n_gla} GLA modules '
          f'(fuse_gate={not args.no_fuse_gate}, gk_in_plugin={args.gk_in_plugin}, '
          f'static_perm={not args.no_static_perm}, split_qkv={args.split_qkv}, fold_gk={not args.no_fold_gk}, '
          f'ln2d_reduce={args.ln2d_reduce}, atlas={args.atlas}, part={args.part})')

    if args.uint8_input:
        x = torch.randint(0, 255, (args.batch, 3, H, W), dtype=torch.uint8, device=device)
    else:
        x = torch.randn(args.batch, 3, H, W, device=device, dtype=dtype)

    if args.part != 'full':
        wrapper = PartWrapper(model, args.part).to(device).eval()
        inputs = (x,)
        input_names = ['images']
        n_out = len(wrapper(x))
        output_names = [f'{args.part.replace(":", "")}_{i}' for i in range(n_out)]
    elif args.postproc:
        assert task in POSTPROC_OUTPUTS, f'--postproc not supported for task {task}'
        wrapper = PostprocWrapper(model, post, task, args.uint8_input).to(device).eval()
        sizes = torch.tensor([[W, H]] * args.batch, device=device)
        inputs = (x, sizes)
        input_names = ['images', 'orig_target_sizes']
        output_names = POSTPROC_OUTPUTS[task]
    else:
        wrapper = RawWrapper(model, task, args.uint8_input).to(device).eval()
        inputs = (x,)
        input_names = ['images']
        output_names = TASK_OUTPUTS[task]
    if args.uint8_input:
        wrapper.pre_scale.data = wrapper.pre_scale.data.to(dtype)
        wrapper.pre_bias.data = wrapper.pre_bias.data.to(dtype)

    with torch.no_grad():
        outs = wrapper(*inputs)
    print('[info] output shapes:', [tuple(o.shape) for o in outs])

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper, inputs, args.output,
            input_names=input_names, output_names=output_names,
            opset_version=args.opset, do_constant_folding=True,
            custom_opsets={CUSTOM_DOMAIN: 1}, dynamo=False,
        )
    print(f'[ok] exported {args.output}')
    stats = finalize_onnx(args.output, simplify=not args.no_simplify, check=not args.no_check)
    meta = dict(config=args.config, task=task, eval_size=[H, W], dtype=args.dtype, batch=args.batch, seed=args.seed,
                backbone=backbone or 'config', fuse_gate=not args.no_fuse_gate, gk_in_plugin=args.gk_in_plugin,
                static_perm=not args.no_static_perm, split_qkv=args.split_qkv, fold_gk=not args.no_fold_gk,
                ln2d_reduce=args.ln2d_reduce, atlas=args.atlas, part=args.part,
                postproc=args.postproc, uint8_input=args.uint8_input,
                input_names=input_names, output_names=output_names,
                output_shapes=[list(o.shape) for o in outs], **stats)
    with open(args.meta or (os.path.splitext(args.output)[0] + '.meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)


if __name__ == '__main__':
    main()
