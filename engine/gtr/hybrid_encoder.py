"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import register
from .utils import get_activation

__all__ = ['HybridEncoder', 'GTREncoder']


class ConvNormLayer_fuse(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()
        padding = (kernel_size-1)//2 if padding is None else padding
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            groups=g,
            padding=padding,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)
        self.ch_in, self.ch_out, self.kernel_size, self.stride, self.g, self.padding, self.bias = \
            ch_in, ch_out, kernel_size, stride, g, padding, bias

    def forward(self, x):
        if hasattr(self, 'conv_bn_fused'):
            y = self.conv_bn_fused(x)
        else:
            y = self.norm(self.conv(x))
        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv_bn_fused'):
            self.conv_bn_fused = nn.Conv2d(
                self.ch_in,
                self.ch_out,
                self.kernel_size,
                self.stride,
                groups=self.g,
                padding=self.padding,
                bias=True)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv_bn_fused.weight.data = kernel
        self.conv_bn_fused.bias.data = bias
        self.__delattr__('conv')
        self.__delattr__('norm')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor()

        return kernel3x3, bias3x3

    def _fuse_bn_tensor(self):
        kernel = self.conv.weight
        running_mean = self.norm.running_mean
        running_var = self.norm.running_var
        gamma = self.norm.weight
        beta = self.norm.bias
        eps = self.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()
        padding = (kernel_size-1)//2 if padding is None else padding
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            groups=g,
            padding=padding,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class SCDown(nn.Module):
    def __init__(self, c1, c2, k, s, act=None):
        super().__init__()
        self.cv1 = ConvNormLayer_fuse(c1, c2, 1, 1)
        self.cv2 = ConvNormLayer_fuse(c2, c2, k, s, c2)

    def forward(self, x):
        return self.cv2(self.cv1(x))


class VGGBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        self.__delattr__('conv1')
        self.__delattr__('conv2')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)

        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=False,
                 act="silu",
                 bottletype=VGGBlock):
        super(CSPLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer_fuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer_fuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            bottletype(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer_fuse(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_2 = self.conv2(x)
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        return self.conv3(x_1 + x_2)

class RepNCSPELAN4(nn.Module):
    # csp-elan
    def __init__(self, c1, c2, c3, c4, n=3,
                 bias=False,
                 act="silu",
                 csp_type='csp2',
                 ):
        super().__init__()
        self.c = c3 // 2
        self.cv1 = ConvNormLayer_fuse(c1, c3, 1, 1, bias=bias, act=act)
        if csp_type == 'csp2':
            CSPLayer = CSPLayer2
        self.cv2 = nn.Sequential(CSPLayer(c3//2, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock))
        self.cv3 = nn.Sequential(CSPLayer(c4, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock))
        self.cv4 = ConvNormLayer_fuse(c3+ (2*c4), c2, 1, 1, bias=bias, act=act)

    def forward_chunk(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


# This layer is equivalent to RepC3 in YOLOs repo
class CSPLayer2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=False,
                 act="silu",
                 bottletype=VGGBlock,
                 ):
        super(CSPLayer2, self).__init__()
        hidden_channels = int(out_channels * expansion)

        self.conv1 = ConvNormLayer_fuse(in_channels, hidden_channels * 2, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            bottletype(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer_fuse(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        y = list(self.conv1(x).chunk(2, 1))
        return self.conv3(y[0] + self.bottlenecks(y[1]))

# transformer
class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="silu",
                 normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        # self.ffn = SwiGLUFFN(d_model, dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # src = self.ffn(src)
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


@register()
class HybridEncoder(nn.Module):
    __share__ = ['eval_spatial_size', ]

    def __init__(self,
                 in_channels=[192, 192, 192],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward = 1024,
                 dropout=0.0,
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None,
                 csp_type='csp',
                 fuse_op='cat',
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        self.fuse_op = fuse_op

        # encoder transformer
        encoder_layer = TransformerEncoderLayer(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=act
            )

        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))
        ])

        input_dim = hidden_dim if self.fuse_op == 'sum' else hidden_dim * 2   # deim use sum instead of cat

        Lateral_Conv = ConvNormLayer_fuse(hidden_dim, hidden_dim, 1, 1)
        SCDown_Conv = nn.Sequential(SCDown(hidden_dim, hidden_dim, 3, 2))

        c1, c2, c3, c4, num_blocks = input_dim, hidden_dim, hidden_dim*2, round(expansion * hidden_dim // 2), round(3 * depth_mult)
        Fuse_Block = RepNCSPELAN4(c1=c1, c2=c2, c3=c3, c4=c4, n=num_blocks, act=act, csp_type=csp_type)

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(copy.deepcopy(Lateral_Conv))
            self.fpn_blocks.append(copy.deepcopy(Fuse_Block))

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(copy.deepcopy(SCDown_Conv))
            self.pan_blocks.append(copy.deepcopy(Fuse_Block))

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)
                # self.register_buffer(f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        """
        """
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, proj_feats):
        assert len(proj_feats) == len(self.in_channels)

        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)

                memory :torch.Tensor = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        # broadcasting and fusion
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh
            upsample_feat = F.interpolate(feat_heigh, scale_factor=2., mode='nearest') # TODO use bilinear
            fused_feat = (upsample_feat + feat_low) \
                if self.fuse_op == 'sum' else torch.concat([upsample_feat, feat_low], dim=1)
            inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](fused_feat)
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            fused_feat = (downsample_feat + feat_height) \
                if self.fuse_op == 'sum' else torch.concat([downsample_feat, feat_height], dim=1)
            out = self.pan_blocks[idx](fused_feat)
            outs.append(out)

        return outs


# ============================================================================
# RF-DETR style encoder (LW-DETR MultiScaleProjector, copied & adapted)
# 设计：把 N 个同分辨率的 ViT 层特征，按目标尺度做 ConvTranspose 上采样 / stride-2
# 卷积下采样 / identity，沿通道 concat，再用一个 C2f (YOLOv8 风格 CSP) 融合到
# hidden_dim。每个目标尺度独立一组分支；3 个尺度即得到 1/8、1/16、1/32 的金字塔。
# ============================================================================


class _LayerNorm2d(nn.Module):
    """Channel-first LayerNorm: input/output shape [B, C, H, W]."""

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps
        self.normalized_shape = (channels,)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


class _ConvBNAct(nn.Module):
    """Conv -> BN(or LN) -> SiLU."""

    def __init__(self, ch_in, ch_out, kernel=3, stride=1, groups=1, dilation=1, act='silu', layer_norm=False,
                 force_contiguous=True):
        super().__init__()
        if not isinstance(kernel, tuple):
            kernel = (kernel, kernel)
        padding = (kernel[0] // 2, kernel[1] // 2)
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=kernel, stride=stride,
                              padding=padding, groups=groups, dilation=dilation, bias=False)
        self.norm = _LayerNorm2d(ch_out) if layer_norm else nn.BatchNorm2d(ch_out)
        self.act = get_activation(act) if act is not None else nn.Identity()
        self.force_contiguous = force_contiguous

    def forward(self, x):
        # The `.contiguous()` is training-lineage dependent, so it is a per-encoder flag
        # (GTREncoder.force_contiguous) rather than a hard-coded call:
        #   * det / seg / semseg / obb (force_contiguous=True, default): the preceding
        #     _LayerNorm2d returns a permute() view with channels_last strides; without the
        #     copy cuDNN runs NHWC kernels whose rounding differs from the NCHW path every
        #     released checkpoint of those tasks was trained/evaluated with (first visible in
        #     the stride-32 stage, then amplified by the decoder's top-k query selection).
        #   * depth (force_contiguous=False): the released GTRDepth checkpoints were trained
        #     WITHOUT the copy, so forcing it here breaks their bit-exact reproduction.
        # The copy costs a few % in the deploy graph; per-task parity with the training code wins.
        # NOTE: rewriting 1x1/s1 convs as F.linear was tried and is a no-op under
        # torch.compile (inductor already lowers 1x1 conv to the same mm — verified
        # bit-identical and equal speed standalone); it only pessimizes eager runs.
        if self.force_contiguous:
            x = x.contiguous()
        return self.act(self.norm(self.conv(x)))


class _Bottleneck(nn.Module):
    """YOLOv8-style 3x3 -> 3x3 bottleneck with optional shortcut."""

    def __init__(self, c_in, c_out, shortcut=True, e=1.0, act='silu', layer_norm=False, force_contiguous=True):
        super().__init__()
        c_hid = int(c_out * e)
        self.cv1 = _ConvBNAct(c_in, c_hid, kernel=3, stride=1, act=act, layer_norm=layer_norm, force_contiguous=force_contiguous)
        self.cv2 = _ConvBNAct(c_hid, c_out, kernel=3, stride=1, act=act, layer_norm=layer_norm, force_contiguous=force_contiguous)
        self.add = shortcut and c_in == c_out

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class _C2f(nn.Module):
    """YOLOv8 C2f (CSP with 2 convs and n bottlenecks)."""

    def __init__(self, c_in, c_out, n=3, shortcut=False, e=0.5, act='silu', layer_norm=False, force_contiguous=True):
        super().__init__()
        self.c = int(c_out * e)
        self.cv1 = _ConvBNAct(c_in, 2 * self.c, kernel=1, stride=1, act=act, layer_norm=layer_norm, force_contiguous=force_contiguous)
        self.cv2 = _ConvBNAct((2 + n) * self.c, c_out, kernel=1, stride=1, act=act, layer_norm=layer_norm, force_contiguous=force_contiguous)
        self.m = nn.ModuleList(
            _Bottleneck(self.c, self.c, shortcut=shortcut, e=1.0, act=act, layer_norm=layer_norm, force_contiguous=force_contiguous) for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), dim=1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, dim=1))


@register()
class GTREncoder(nn.Module):
    """RF-DETR / LW-DETR 风格的 MultiScaleProjector。

    输入：来自 ViT 的 N 个同分辨率特征 [B, in_dim, H, W]（H=W=img/patch_size）。
    输出：len(scale_factors) 个尺度的金字塔 [B, hidden_dim, H', W']。

    每个目标尺度独立一组：
        - 对每个输入层应用 (scale, layer) 专属重采样：
            * scale=4.0  -> 2 次 ConvTranspose2d(↑2，通道 //2)
            * scale=2.0  -> 1 次 ConvTranspose2d(↑2，通道 //2)
            * scale=1.0  -> identity
            * scale=0.5  -> stride-2 ConvBNAct(下采样 ½，通道不变)
        - 沿通道 concat
        - C2f(in_dim_after_resample -> hidden_dim) + LayerNorm2d
    """

    __share__ = ['eval_spatial_size', ]

    def __init__(
        self,
        in_channels,                 # list[int]，长度 N，ViT 各层通道数（一般都相同）
        hidden_dim=256,
        scale_factors=(2.0, 1.0, 0.5),
        feat_strides=(8, 16, 32),    # 每个 scale 对应的下游 stride，喂给 decoder
        num_blocks=3,                # C2f 内 bottleneck 个数
        layer_norm=True,             # C2f 内是否使用 LayerNorm2d (True: 与 RF-DETR 一致)
        act='silu',
        eval_spatial_size=None,      # 仅占位，与 HybridEncoder 接口对齐
        force_contiguous=True,       # _ConvBNAct 是否对 conv 输入强制 .contiguous()（见该类注释）
                                     # det/seg/semseg/obb=True；depth 发布权重训练时无此拷贝，需置 False
    ):
        super().__init__()
        in_channels = list(in_channels)
        scale_factors = list(scale_factors)
        feat_strides = list(feat_strides)
        assert len(scale_factors) == len(feat_strides), \
            f"len(scale_factors)={len(scale_factors)} != len(feat_strides)={len(feat_strides)}"

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.scale_factors = scale_factors
        self.feat_strides = feat_strides
        self.out_channels = [hidden_dim for _ in scale_factors]
        self.out_strides = feat_strides
        self.eval_spatial_size = eval_spatial_size

        # 每个目标尺度一组 per-layer 重采样模块 + 一个 C2f 融合
        stages_sampling = []
        stages = []
        for scale in scale_factors:
            samp_per_layer = []
            in_dim_after_resample = 0
            for in_dim in in_channels:
                if scale == 4.0:
                    layers = [
                        nn.ConvTranspose2d(in_dim, in_dim // 2, kernel_size=2, stride=2),
                        _LayerNorm2d(in_dim // 2),
                        nn.GELU(),
                        nn.ConvTranspose2d(in_dim // 2, in_dim // 4, kernel_size=2, stride=2),
                    ]
                    out_c_layer = in_dim // 4
                elif scale == 2.0:
                    layers = [nn.ConvTranspose2d(in_dim, in_dim // 2, kernel_size=2, stride=2)]
                    out_c_layer = in_dim // 2
                elif scale == 1.0:
                    layers = []
                    out_c_layer = in_dim
                elif scale == 0.5:
                    layers = [_ConvBNAct(in_dim, in_dim, kernel=3, stride=2, act=act, layer_norm=layer_norm, force_contiguous=force_contiguous)]
                    out_c_layer = in_dim
                else:
                    raise NotImplementedError(f"Unsupported scale_factor: {scale}")
                samp_per_layer.append(nn.Sequential(*layers) if layers else nn.Identity())
                in_dim_after_resample += out_c_layer
            stages_sampling.append(nn.ModuleList(samp_per_layer))

            stages.append(nn.Sequential(
                _C2f(in_dim_after_resample, hidden_dim, n=num_blocks, act=act, layer_norm=layer_norm, force_contiguous=force_contiguous),
                _LayerNorm2d(hidden_dim),
            ))

        self.stages_sampling = nn.ModuleList(stages_sampling)
        self.stages = nn.ModuleList(stages)

    def convert_to_deploy(self):
        # The per-scale stages only read `feats` and are mutually independent, and
        # their conv kernels are too small to fill the GPU one at a time. In deploy
        # mode run stage 0 on the current stream and fork the remaining stages onto
        # side streams; inside an outer CUDA-graph capture this records parallel
        # branches that overlap on idle SMs.
        self._deploy_parallel_stages = True
        self._stage_streams = None

    def _run_stage(self, stage_idx, feats):
        samp_list = self.stages_sampling[stage_idx]
        fuse = self.stages[stage_idx]
        resampled = [samp(feats[j]) for j, samp in enumerate(samp_list)]
        fused = resampled[0] if len(resampled) == 1 else torch.cat(resampled, dim=1)
        return fuse(fused)

    # Dynamo must not trace the stream fork/join (inductor's stream codegen chokes
    # on it); recursive=False so the per-stage compiled submodules still run their
    # compiled artifacts when invoked from this eager frame.
    @torch.compiler.disable(recursive=False)
    def _forward_parallel_stages(self, feats):
        cur = torch.cuda.current_stream()
        if self._stage_streams is None:
            self._stage_streams = [torch.cuda.Stream() for _ in range(len(self.stages) - 1)]
        outs = [None] * len(self.stages)
        for i, s in enumerate(self._stage_streams):
            s.wait_stream(cur)
            with torch.cuda.stream(s):
                outs[i + 1] = self._run_stage(i + 1, feats)
        outs[0] = self._run_stage(0, feats)
        capturing = torch.cuda.is_current_stream_capturing()
        for i, s in enumerate(self._stage_streams):
            cur.wait_stream(s)
            if not capturing:
                # Mark cross-stream use so the caching allocator does not hand the
                # block back to the side stream while the main stream still reads it.
                outs[i + 1].record_stream(cur)
        return outs

    def forward(self, feats):
        assert len(feats) == len(self.in_channels), \
            f"expect {len(self.in_channels)} ViT-layer features, got {len(feats)}"

        if (getattr(self, '_deploy_parallel_stages', False) and not self.training
                and feats[0].is_cuda and len(self.stages) > 1):
            return self._forward_parallel_stages(feats)

        outs = []
        for stage_idx, (samp_list, fuse) in enumerate(zip(self.stages_sampling, self.stages)):
            resampled = [samp(feats[j]) for j, samp in enumerate(samp_list)]
            fused = resampled[0] if len(resampled) == 1 else torch.cat(resampled, dim=1)
            outs.append(fuse(fused))
        return outs
