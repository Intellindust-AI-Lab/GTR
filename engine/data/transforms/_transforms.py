"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn

import torchvision
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F

import PIL
import PIL.Image

from typing import Any, Dict, List, Optional

from .._misc import convert_to_tv_tensor, _boxes_keys
from .._misc import Image, Video, Mask, BoundingBoxes
from .._misc import SanitizeBoundingBoxes

from ...core import register
torchvision.disable_beta_transforms_warning()


RandomPhotometricDistort = register()(T.RandomPhotometricDistort)
RandomZoomOut = register()(T.RandomZoomOut)
RandomHorizontalFlip = register()(T.RandomHorizontalFlip)
Resize = register()(T.Resize)
# ToImageTensor = register()(T.ToImageTensor)
# ConvertDtype = register()(T.ConvertDtype)
# PILToTensor = register()(T.PILToTensor)
SanitizeBoundingBoxes = register(name='SanitizeBoundingBoxes')(SanitizeBoundingBoxes)
RandomCrop = register()(T.RandomCrop)
Normalize = register()(T.Normalize)


@register()
class ResizeShortestEdge(T.Resize):
    """T.Resize with an int size = shortest-edge resize, i.e. DA-V2's keep-aspect
    'lower_bound' resize. Registered under its own name so
    YAMLConfig.reset_tranforms_size does not rewrite it to a square size."""
    pass


def _get_spatial_size(inpt):
    return F.get_size(inpt) if hasattr(F, "get_size") else F.get_spatial_size(inpt)


def _get_fill_value(fill, inpt):
    if isinstance(fill, dict):
        if type(inpt) in fill:
            return fill[type(inpt)]
        if "others" in fill:
            return fill["others"]
    return fill


@register()
class LargeScaleJitter(nn.Module):
    def __init__(self, output_size, min_scale=0.1, max_scale=2.0, fill=0, mask_fill=None,
                 cat_max_ratio=None, ignore_index=255) -> None:
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = tuple(output_size)
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.fill = fill
        # mask_fill: separate pad value for Mask tv-tensors (e.g. 255 = semantic-seg
        # ignore label, while the image pads with `fill`). None keeps the legacy
        # behavior of one scalar fill for every type.
        self.mask_fill = mask_fill
        # cat_max_ratio: mmseg RandomCrop's semantic-seg guard. Resample the crop while a
        # single class covers >= this share of the non-ignore pixels of target['seg_map'],
        # so all-road / all-sky crops do not spend a training sample on one class.
        # None disables the check and leaves every detection config bit-identical.
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index
        self.random_crop = T.RandomCrop(self.output_size)

    def _crop_balanced(self, image, target):
        """mmseg RandomCrop(cat_max_ratio=...): take the first of 10 crops that is not
        dominated by a single class, falling back to the last one when none qualifies."""
        for _ in range(10):
            cropped_image, cropped_target = self.random_crop(image, target)
            labels, counts = torch.unique(cropped_target['seg_map'], return_counts=True)
            counts = counts[labels != self.ignore_index]
            if len(counts) > 1 and counts.max() / counts.sum() < self.cat_max_ratio:
                break
        return cropped_image, cropped_target

    def forward(self, *inputs):
        sample = inputs if len(inputs) > 1 else inputs[0]

        if len(sample) == 3:
            image, target, dataset = sample
        elif len(sample) == 2:
            image, target = sample
            dataset = None
        else:
            raise ValueError('LargeScaleJitter expects (image, target) or (image, target, dataset).')

        input_h, input_w = _get_spatial_size(image)
        scale = torch.empty(1).uniform_(self.min_scale, self.max_scale).item()

        scaled_h = self.output_size[0] * scale
        scaled_w = self.output_size[1] * scale
        resize_scale = min(scaled_h / input_h, scaled_w / input_w)
        resized_h = max(1, int(round(input_h * resize_scale)))
        resized_w = max(1, int(round(input_w * resize_scale)))

        resize = T.Resize((resized_h, resized_w))
        image, target = resize(image, target)

        resized_h, resized_w = _get_spatial_size(image)
        if resized_h < self.output_size[0] or resized_w < self.output_size[1]:
            pad = PadToSize(self.output_size, fill=self.fill)
            if self.mask_fill is not None:
                # PadToSize._transform resolves fill via this module's _get_fill_value,
                # which natively understands {type: value, 'others': value} dicts;
                # setting _fill directly sidesteps torchvision's fill validation.
                pad._fill = {Mask: self.mask_fill, 'others': self.fill}
            image, target = pad(image, target)

        if self.cat_max_ratio is None:
            image, target = self.random_crop(image, target)
        else:
            image, target = self._crop_balanced(image, target)

        if dataset is None:
            return image, target

        return image, target, dataset


@register()
class StandardScaleJitter(nn.Module):
    """Random isotropic scale in [min_scale, max_scale]; boxes updated; square Resize follows in pipeline."""

    def __init__(self, min_scale: float = 0.8, max_scale: float = 1.25) -> None:
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale

    def forward(self, *inputs):
        sample = inputs if len(inputs) > 1 else inputs[0]

        if len(sample) == 3:
            image, target, dataset = sample
        elif len(sample) == 2:
            image, target = sample
            dataset = None
        else:
            raise ValueError('StandardScaleJitter expects (image, target) or (image, target, dataset).')

        input_h, input_w = _get_spatial_size(image)
        scale = torch.empty(1).uniform_(self.min_scale, self.max_scale).item()
        resized_h = max(1, int(round(input_h * scale)))
        resized_w = max(1, int(round(input_w * scale)))
        resize = T.Resize((resized_h, resized_w))
        image, target = resize(image, target)

        if dataset is None:
            return image, target
        return image, target, dataset


@register()
class EmptyTransform(T.Transform):
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs


@register()
class PadToSize(T.Pad):
    _transformed_types = (
        PIL.Image.Image,
        Image,
        Video,
        Mask,
        BoundingBoxes,
    )
    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        input_h, input_w = _get_spatial_size(flat_inputs[0])
        target_h, target_w = self.size
        # Pad only, never shrink: when one side is short and the other long
        # (e.g. LargeScaleJitter on a 2:1 image), a negative pad would silently
        # crop the right/bottom DETERMINISTICALLY and degrade the follow-up
        # RandomCrop to a no-op. Clamp so the long side is left for RandomCrop.
        pad_h = max(0, target_h - input_h)
        pad_w = max(0, target_w - input_w)
        self.padding = [0, 0, pad_w, pad_h]
        return dict(padding=self.padding)

    def __init__(self, size, fill=0, padding_mode='constant') -> None:
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        super().__init__(0, fill, padding_mode)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        fill = _get_fill_value(self._fill, inpt)
        padding = params['padding']
        return F.pad(inpt, padding=padding, fill=fill, padding_mode=self.padding_mode)  # type: ignore[arg-type]

    def __call__(self, *inputs: Any) -> Any:
        outputs = super().forward(*inputs)
        if len(outputs) > 1 and isinstance(outputs[1], dict):
            outputs[1]['padding'] = torch.tensor(self.padding)
        return outputs


@register()
class RandomIoUCrop(T.RandomIoUCrop):
    def __init__(self, min_scale: float = 0.3, max_scale: float = 1, min_aspect_ratio: float = 0.5, max_aspect_ratio: float = 2, sampler_options: Optional[List[float]] = None, trials: int = 40, p: float = 1.0):
        super().__init__(min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials)
        self.p = p

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        return super().forward(*inputs)


@register()
class ConvertBoxes(T.Transform):
    _transformed_types = (
        BoundingBoxes,
    )
    def __init__(self, fmt='', normalize=False) -> None:
        super().__init__()
        self.fmt = fmt
        self.normalize = normalize

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        spatial_size = getattr(inpt, _boxes_keys[1])
        if self.fmt:
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.fmt.lower())
            inpt = convert_to_tv_tensor(inpt, key='boxes', box_format=self.fmt.upper(), spatial_size=spatial_size)

        if self.normalize:
            inpt = inpt / torch.tensor(spatial_size[::-1]).tile(2)[None]

        return inpt


@register()
class ConvertPILImage(T.Transform):
    _transformed_types = (
        PIL.Image.Image,
    )
    def __init__(self, dtype='float32', scale=True) -> None:
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        inpt = F.pil_to_tensor(inpt)
        if self.dtype == 'float32':
            inpt = inpt.float()

        if self.scale:
            inpt = inpt / 255.

        inpt = Image(inpt)

        return inpt
