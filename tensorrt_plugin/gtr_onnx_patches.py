"""Monkeypatches that make a GTR deploy model traceable by ``torch.onnx.export`` (TorchScript tracer).

1. ``GatedLinearAttention.forward`` is replaced by a version that emits ONE custom node
   ``custom::GatedLinearAttention`` (the TensorRT plugin in ``trt_plugin/``) instead of the
   Triton / CUDA-extension call. Everything around it (fused qkv GEMM, gk projection,
   logsigmoid gate, RMSNorm-gate, o_proj) stays standard ONNX.
2. ``GTRTransformer._build_value_atlas`` (det/seg MSDA fast path) is rebuilt without in-place
   slice assignment (which the tracer would turn into ScatterND): Concat with zero blocks by
   default, Pad+Add optionally (hits a TensorRT 11.2 fusion bug in some builds), or no atlas.
3. ``SegmentationHead`` einsum -> MatMul (TensorRT's Einsum support is narrower).
4. ``GTREncoder`` multi-stream stage fork/join is switched off (eager-only optimisation).
5. Optional: ViTAdapterSpatialSwiGLU bid_scan flip/transpose chain -> one Gather with a precomputed index.
"""

import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from engine.gtr.backbone.vit_adapter import GatedLinearAttention
from engine.gtr.backbone.vit_adapter_spatial_swiglu import VisionTransformerSpatialSwiGLU
from engine.gtr.det.decoder import GTRTransformer
from engine.gtr.det.segmentation_head import SegmentationHead

CUSTOM_DOMAIN = 'custom'
GLA_OP = 'GatedLinearAttention'

# Raw model outputs per task (order = ONNX output order).
TASK_OUTPUTS = {
    'detection': ['pred_logits', 'pred_boxes'],
    'segmentation': ['pred_logits', 'pred_boxes', 'pred_masks'],
    'obb': ['pred_logits', 'pred_boxes'],
    'pose': ['pred_logits', 'pred_keypoints'],
    'semantic_segmentation': ['pred_sem_seg'],
    'depth': ['pred_depth'],
}


# --------------------------------------------------------------------------- GLA custom op
def _ext():
    from engine.gtr.backbone.vit_adapter import _gla_cuda_ext
    return _gla_cuda_ext


def _run_kernel(q, k, v, gk, scale, gk_normalizer, g=None, w=None, rms_eps=0.0):
    """Run the real CUDA kernel so the patched model is also usable in eager mode
    (equivalence tests); falls back to zeros when the extension is unavailable."""
    ext = _ext()
    if ext is None or not q.is_cuda:
        return torch.zeros_like(v)
    args = [t.contiguous() for t in (q, k, v, gk)]
    B, T, H = q.shape[:3]
    ws = ext.make_workspace(B, T, H, q.device)
    out = torch.empty_like(args[2])
    if g is not None:
        ext.chunk_gla_run_gated(*args, g.contiguous(), w.contiguous(), scale, rms_eps, ws, out, gk_normalizer)
    else:
        ext.chunk_gla_run(*args, scale, ws, out, gk_normalizer)
    return out


class _GLAOp4(torch.autograd.Function):
    """q,k,gk [B,T,H,K]; v [B,T,H,V] -> o [B,T,H,V]. symbolic() is what lands in the graph."""

    @staticmethod
    def forward(ctx, q, k, v, gk, scale, gk_normalizer):
        return _run_kernel(q, k, v, gk, scale, gk_normalizer)

    @staticmethod
    def symbolic(g, q, k, v, gk, scale, gk_normalizer):
        out = g.op(f'{CUSTOM_DOMAIN}::{GLA_OP}', q, k, v, gk,
                   scale_f=float(scale), gk_normalizer_f=float(gk_normalizer), rms_eps_f=0.0)
        out.setType(v.type())
        return out


class _GLAOp6(torch.autograd.Function):
    """Same, plus g [B,T,H,V] and rms_w [V]: RMSNorm(o) * rms_w * silu(g) fused in the kernel."""

    @staticmethod
    def forward(ctx, q, k, v, gk, g, w, scale, gk_normalizer, rms_eps):
        return _run_kernel(q, k, v, gk, scale, gk_normalizer, g, w, rms_eps)

    @staticmethod
    def symbolic(g_, q, k, v, gk, g, w, scale, gk_normalizer, rms_eps):
        out = g_.op(f'{CUSTOM_DOMAIN}::{GLA_OP}', q, k, v, gk, g, w,
                    scale_f=float(scale), gk_normalizer_f=float(gk_normalizer), rms_eps_f=float(rms_eps))
        out.setType(v.type())
        return out


def _manual_rms_norm(x, weight, eps):
    out = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return out * weight if weight is not None else out


def _gla_forward_onnx(self, hidden_states, attention_mask=None, past_key_values=None,
                      use_cache=False, output_attentions=False, **kwargs):
    assert attention_mask is None and not self.use_short_conv and not self.qk_norm
    assert self.feature_map_fn is None and self.num_kv_groups == 1
    opts = self._onnx_gla_opts

    if getattr(self, 'fused_qkv_g_gk_low', False):
        if opts.get('fold_gk', True):
            # Precomputed in prepare_for_export: [q|k|v|g|gk] weight with the rank-16 gk
            # projection composed in (see _fold_gk_projection).
            w, bias, dims = self._onnx_w_all, self._onnx_b_all, list(self._onnx_dims)
        else:
            w, bias, dims = self.qkv_g_gkl_proj.weight, None, list(self._fused_split)
        if opts.get('split_qkv'):
            # One GEMM per projection: each output is born contiguous in its own buffer, so
            # no Split/Slice copies are needed before the plugin (trades extra small GEMMs).
            parts, off = [], 0
            for n in dims:
                parts.append(F.linear(hidden_states, w[off:off + n], None if bias is None else bias[off:off + n]))
                off += n
            q, k, v, g_pre, last = parts
        else:
            fused = F.linear(hidden_states, w, bias)
            q, k, v, g_pre, last = fused.split(dims, dim=-1)
        gk = last if opts.get('fold_gk', True) else self.gk_high_proj(last)
    else:
        q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
        gk = self.gk_proj(hidden_states)
        g_pre = self.g_proj(hidden_states) if self.use_output_gate else None

    q = rearrange(q, '... (h d) -> ... h d', d=self.head_k_dim)
    k = rearrange(k, '... (h d) -> ... h d', d=self.head_k_dim)
    gk = rearrange(gk, '... (h d) -> ... h d', d=self.head_k_dim)
    v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)

    if opts['gk_in_plugin']:
        assert self.clamp_min is None, 'gk_in_plugin requires clamp_min=None'
        gk_norm_attr = float(self.gate_logit_normalizer)
    else:
        # logsigmoid(x) = min(x, 0) - log1p(exp(-|x|)): stays finite in fp16 for any x, unlike
        # the log(sigmoid(x)) form the exporter would emit (sigmoid underflows below x=-17).
        gk = (torch.clamp_max(gk, 0.0) - torch.log1p(torch.exp(-gk.abs()))) / self.gate_logit_normalizer
        if self.clamp_min is not None:
            gk = torch.clamp_min(gk, self.clamp_min)
        gk_norm_attr = 0.0

    scale = float(self.head_k_dim ** -0.5)
    dt = hidden_states.dtype
    h16 = lambda t: t.to(torch.float16)   # plugin I/O is fp16; a no-op in fp16 graphs

    fuse_gate = opts['fuse_gate'] and self.use_output_gate and self.fuse_norm_and_gate
    if fuse_gate:
        g_ = rearrange(g_pre, '... (h d) -> ... h d', d=self.head_v_dim)
        norm = self.g_norm_swish_gate
        y = _GLAOp6.apply(h16(q), h16(k), h16(v), h16(gk), h16(g_), h16(norm.weight),
                          scale, gk_norm_attr, float(norm.eps))
        o = rearrange(y.to(dt), '... h d -> ... (h d)')
    else:
        o = _GLAOp4.apply(h16(q), h16(k), h16(v), h16(gk), scale, gk_norm_attr).to(dt)
        if self.use_output_gate:
            g_ = rearrange(g_pre, '... (h d) -> ... h d', d=self.head_v_dim)
            if self.fuse_norm_and_gate:
                norm = self.g_norm_swish_gate
                y = _manual_rms_norm(o, norm.weight, norm.eps) * F.silu(g_)
            else:
                y = _manual_rms_norm(o, self.g_norm.weight, self.g_norm.eps) * self.gate_fn(g_)
            o = rearrange(y, '... h d -> ... (h d)')
        else:
            o = rearrange(_manual_rms_norm(o, self.g_norm.weight, self.g_norm.eps), '... h d -> ... (h d)')
    return self.o_proj(o), None, past_key_values


@torch.no_grad()
def _fold_gk_projection(attn: GatedLinearAttention):
    """gk = W_high (W_low x) + b_high == (W_high W_low) x + b_high: compose the rank-16 gate
    projection into the fused [q|k|v|g] GEMM. Exact in real arithmetic; the composition is
    done in fp32 and rounded once to the model dtype. Removes one small GEMM per block."""
    w = attn.qkv_g_gkl_proj.weight                            # [key+key+val+val+low, hidden]
    dims = list(attn._fused_split)
    n_low = dims[-1]
    w_gk = (attn.gk_high_proj.weight.float() @ w[-n_low:].float()).to(w.dtype)
    w_all = torch.cat([w[:-n_low], w_gk], dim=0).contiguous()
    b_all = torch.cat([torch.zeros(sum(dims[:-1]), dtype=w.dtype, device=w.device),
                       attn.gk_high_proj.bias.to(w.dtype)]).contiguous()
    attn.register_buffer('_onnx_w_all', w_all, persistent=False)
    attn.register_buffer('_onnx_b_all', b_all, persistent=False)
    attn._onnx_dims = tuple(dims[:-1] + [w_gk.shape[0]])


# --------------------------------------------------------------------------- nn.MultiheadAttention
def _mha_forward_export(self, query, key, value, key_padding_mask=None, need_weights=False, attn_mask=None,
                        average_attn_weights=True, is_causal=False):
    """Explicit batch_first MHA (in_proj -> SDPA -> out_proj). The eval-mode fast path
    (aten::_native_multi_head_attention) has no ONNX symbolic."""
    assert self.batch_first and key_padding_mask is None and not is_causal
    B, Nq, E = query.shape
    Nk = key.shape[1]
    nh = self.num_heads
    hd = E // nh
    w_q, w_k, w_v = self.in_proj_weight.split(E, dim=0)
    if self.in_proj_bias is not None:
        b_q, b_k, b_v = self.in_proj_bias.split(E, dim=0)
    else:
        b_q = b_k = b_v = None
    q = F.linear(query, w_q, b_q).view(B, Nq, nh, hd).transpose(1, 2)
    k = F.linear(key, w_k, b_k).view(B, Nk, nh, hd).transpose(1, 2)
    v = F.linear(value, w_v, b_v).view(B, Nk, nh, hd).transpose(1, 2)
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype).masked_fill(attn_mask, float('-inf'))
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    out = self.out_proj(out.transpose(1, 2).reshape(B, Nq, E))
    return out, None


# --------------------------------------------------------------------------- encoder LayerNorm2d
def _ln2d_forward_reduce(self, x):
    """Channel LayerNorm written as reductions over axis 1 (no permutes): lets TensorRT keep
    the conv tensors in its channels-last format instead of Transpose -> LN -> Transpose ->
    reformat around every _LayerNorm2d."""
    mean = x.mean(dim=1, keepdim=True)
    xc = x - mean
    var = xc.pow(2).mean(dim=1, keepdim=True)
    y = xc * torch.rsqrt(var + self.eps)
    return y * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


# --------------------------------------------------------------------------- det/seg value atlas
def _build_value_atlas_pad(self, feats):
    dec = self.decoder
    bs = feats[0].shape[0]
    nh = self.nhead
    c = self.hidden_dim // nh
    H_A, W_A = dec._va_hw
    canvas = None
    for f, (h, w), (r0, c0) in zip(feats, dec._va_shapes, dec._va_origins):
        x = f.reshape(bs * nh, c, h, w)
        x = F.pad(x, (c0, W_A - c0 - w, r0, H_A - r0 - h))
        canvas = x if canvas is None else canvas + x
    return canvas


def _build_value_atlas_concat(self, feats):
    """Same canvas as _build_value_atlas_pad, assembled with Concat of the level tensors and
    constant zero blocks (one row band per level, then the bands). No Pad / Add: TensorRT
    11.2 fuses Pad x3 + Add + GridSample into one Myelin kernel that, in some builds
    (gtr_s / gtr_m at --builderOptimizationLevel=5, gtr_s at --maxAuxStreams=0), writes a
    corrupted canvas (non-zero gutters, inf/NaN) and every decoder output becomes NaN."""
    dec = self.decoder
    bs = feats[0].shape[0]
    nh = self.nhead
    c = self.hidden_dim // nh
    H_A, W_A = dec._va_hw
    bands, r = [], 0
    for f, (h, w), (r0, c0) in zip(feats, dec._va_shapes, dec._va_origins):
        x = f.reshape(bs * nh, c, h, w)
        if r0 > r:
            bands.append(x.new_zeros(bs * nh, c, r0 - r, W_A))
        parts = [x.new_zeros(bs * nh, c, h, c0), x, x.new_zeros(bs * nh, c, h, W_A - c0 - w)]
        bands.append(torch.cat([p for p in parts if p.shape[3] > 0], dim=3))
        r = r0 + h
    if H_A > r:
        bands.append(x.new_zeros(bs * nh, c, H_A - r, W_A))
    return torch.cat(bands, dim=2)


_ATLAS_BUILDERS = {'pad': _build_value_atlas_pad, 'concat': _build_value_atlas_concat}


# --------------------------------------------------------------------------- seg head
def _seg_head_forward_matmul(self, spatial_features, query_features, skip_blocks=False):
    assert len(query_features) == 1
    target_size = (self.image_size[0] // self.downsample_ratio, self.image_size[1] // self.downsample_ratio)
    spatial_features = F.interpolate(spatial_features, size=target_size, mode='bilinear', align_corners=False)
    if not skip_blocks:
        for block in self.blocks:
            spatial_features = block(spatial_features)
    spatial_features = self.spatial_features_proj(spatial_features)           # [B, C, H, W]
    qf = self.query_features_proj(self.query_features_block(query_features[0]))  # [B, N, C]
    B, C, H, W = spatial_features.shape
    logits = torch.matmul(qf, spatial_features.flatten(2)).view(B, -1, H, W)    # bchw,bnc->bnhw
    return [logits + self.bias]


# --------------------------------------------------------------------------- ViTAdapterSpatialSwiGLU static perm
def _install_static_perm(vit: VisionTransformerSpatialSwiGLU, grid):
    gs_h, gs_w = grid
    assert gs_h == gs_w, 'bid_scan static perm needs a square token grid'
    L = gs_h * gs_w
    device = next(vit.parameters()).device
    identity = torch.arange(L, dtype=torch.long, device=device)
    perms = []
    for i in range(vit.depth):
        p = identity.flip(0)
        if i % 2 == 1 and vit.if_quad_dir:
            p = p.view(gs_h, gs_w).t().reshape(-1)
        perms.append(p.contiguous())
    vit.register_buffer('_onnx_bid_perm', torch.stack(perms, 0).contiguous(), persistent=False)

    def _apply_bid_scan(self, x, layer_idx):
        return x.index_select(1, self._onnx_bid_perm[layer_idx])

    vit._apply_bid_scan = types.MethodType(_apply_bid_scan, vit)


# --------------------------------------------------------------------------- entry point
def prepare_for_export(model: nn.Module, eval_size, fuse_gate=True, gk_in_plugin=False, static_perm=True,
                       split_qkv=False, fold_gk=True, ln2d_reduce=False, atlas='none'):
    """Apply all patches in place. ``model`` must already be ``.deploy()``-ed.
    ``atlas``: how det/seg build the MSDA value atlas: 'pad' (Pad+Add), 'concat' (Concat with
    zero blocks) or 'none' (no atlas: one GridSample per level, as obb/pose do)."""
    from engine.gtr.hybrid_encoder import _LayerNorm2d
    n_gla = 0
    for m in model.modules():
        if ln2d_reduce and isinstance(m, _LayerNorm2d):
            m.forward = types.MethodType(_ln2d_forward_reduce, m)
        if isinstance(m, GatedLinearAttention):
            m._onnx_gla_opts = dict(fuse_gate=fuse_gate, gk_in_plugin=gk_in_plugin, split_qkv=split_qkv,
                                    fold_gk=fold_gk)
            if fold_gk and getattr(m, 'fused_qkv_g_gk_low', False):
                _fold_gk_projection(m)
            m.forward = types.MethodType(_gla_forward_onnx, m)
            n_gla += 1
        if isinstance(m, nn.MultiheadAttention):
            m.forward = types.MethodType(_mha_forward_export, m)
        if hasattr(m, '_deploy_parallel_stages'):
            m._deploy_parallel_stages = False
        if isinstance(m, GTRTransformer):
            if atlas == 'none':
                m.decoder._va_ready = False
            else:
                m._build_value_atlas = types.MethodType(_ATLAS_BUILDERS[atlas], m)
        if isinstance(m, SegmentationHead):
            m.forward = types.MethodType(_seg_head_forward_matmul, m)
        if static_perm and isinstance(m, VisionTransformerSpatialSwiGLU) and m.bid_scan:
            patch = m._model.patch_embed.convs[0].conv_bn_fused if hasattr(m._model.patch_embed.convs[0], 'conv_bn_fused') else None
            grid = (eval_size[0] // m.patch_size, eval_size[1] // m.patch_size)
            _install_static_perm(m, grid)
    return n_gla
