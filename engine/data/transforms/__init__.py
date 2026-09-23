"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

# torchvision v2 API compat: tv>=0.22 renamed Transform._get_params->make_params and
# _transform->transform. The custom transforms here use the legacy names. Bridge them on
# the base class, active only when the new API is present (old torchvision env is untouched,
# since its forward() calls the legacy methods directly).
import torchvision.transforms.v2 as _tv_v2
_TB = _tv_v2.Transform
if hasattr(_TB, 'make_params') and not getattr(_TB, '_gtr_tvcompat', False):
    _new_make_params, _new_transform = _TB.make_params, _TB.transform

    def _bridged_make_params(self, flat_inputs):
        for c in type(self).__mro__:
            if c is _TB:
                break
            if '_get_params' in c.__dict__:
                return self._get_params(flat_inputs)
        return _new_make_params(self, flat_inputs)

    def _bridged_transform(self, inpt, params):
        for c in type(self).__mro__:
            if c is _TB:
                break
            if '_transform' in c.__dict__:
                return self._transform(inpt, params)
        return _new_transform(self, inpt, params)

    _TB.make_params = _bridged_make_params
    _TB.transform = _bridged_transform
    _TB._gtr_tvcompat = True

from ._transforms import (
    EmptyTransform,
    LargeScaleJitter,
    StandardScaleJitter,
    RandomPhotometricDistort,
    RandomZoomOut,
    RandomIoUCrop,
    RandomHorizontalFlip,
    Resize,
    PadToSize,
    SanitizeBoundingBoxes,
    RandomCrop,
    Normalize,
    ConvertBoxes,
    ConvertPILImage,
)
from .container import Compose
from .mosaic import Mosaic
from .obb_transforms import (
    RandomOBBFlip,
    RandomOBBRotate,
    RandomOBBLargeScaleJitter,
    ConvertDOTABoxes,
)
from .pose_transforms import (
    PoseCompose,
    PoseMosaic,
    MixUpCopyPaste,
    PoseRandomZoomOut,
    PoseRandomHorizontalFlip,
    PoseColorJitter,
    PoseResize,
    PoseToTensor,
    PoseNormalize,
)