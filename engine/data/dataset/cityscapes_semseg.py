"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
"""

import os

import numpy as np
import torch
from PIL import Image as PILImage

from ...core import register
from .._misc import Mask
from ._dataset import DetDataset

__all__ = ['CityscapesSemSeg']


@register()
class CityscapesSemSeg(DetDataset):
    """Cityscapes 19-class semantic segmentation.

    Reads a two-column index file (one "img_relpath label_relpath" pair per line,
    both relative to ``data_root``) produced by prepare_cityscapes.sh. Labels are
    ``*_gtFine_labelTrainIds.png``: uint8 maps already collapsed to trainIds
    {0..18} with 255 = ignore, so nothing is remapped at load time.

    ``target['seg_map']`` is a (1, H, W) Mask tv-tensor: torchvision-v2 geometric
    ops (Resize / RandomCrop / flip / pad) transform it in lockstep with the image
    using nearest interpolation, while photometric ops and Normalize skip it.

    ``keep_native_label=True`` (val): the seg map is popped before the transform
    pipeline and re-attached untouched, so the image can be squashed to the square
    the backbone requires while mIoU is still measured against the native-resolution
    ground truth (SemSegPostProcessor resizes the logits back instead).
    """
    __inject__ = ['transforms']

    CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
               'traffic light', 'traffic sign', 'vegetation', 'terrain',
               'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
               'motorcycle', 'bicycle')
    PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
               [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
               [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
               [0, 80, 100], [0, 0, 230], [119, 11, 32]]

    def __init__(self, data_root, ann_file, transforms,
                 ignore_index=255, keep_native_label=False):
        self.data_root = data_root
        self.ann_file = ann_file
        self._transforms = transforms
        self.ignore_index = ignore_index
        self.keep_native_label = keep_native_label

        index_path = os.path.join(data_root, ann_file)
        self.items = []
        with open(index_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                img_rel, lbl_rel = line.split()
                self.items.append((os.path.join(data_root, img_rel),
                                   os.path.join(data_root, lbl_rel)))
        assert self.items, f'Empty index file: {index_path}'

    def __len__(self):
        return len(self.items)

    def load_item(self, idx):
        img_path, lbl_path = self.items[idx]
        image = PILImage.open(img_path).convert('RGB')
        seg = torch.from_numpy(np.array(PILImage.open(lbl_path), dtype=np.uint8))
        w, h = image.size
        target = {
            'image_id': torch.tensor([idx]),
            # [w, h], matching the CocoDetection/PostProcessor convention.
            'orig_size': torch.tensor([w, h]),
            'seg_map': Mask(seg[None]),  # (1, H, W) uint8
        }
        return image, target

    def __getitem__(self, idx):
        image, target = self.load_item(idx)
        if self._transforms is not None:
            if self.keep_native_label:
                seg = target.pop('seg_map')
                image, target, _ = self._transforms(image, target, self)
                target['seg_map'] = seg
            else:
                image, target, _ = self._transforms(image, target, self)
        return image, target

    def extra_repr(self) -> str:
        s = f' data_root: {self.data_root}\n ann_file: {self.ann_file}\n'
        s += f' len: {len(self.items)}  keep_native_label: {self.keep_native_label}\n'
        if self._transforms is not None:
            s += f' transforms:\n   {repr(self._transforms)}'
        return s
