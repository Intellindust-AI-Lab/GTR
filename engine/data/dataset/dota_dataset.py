"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
DOTA dataset (mmrotate split format): images/*.png + annfiles/*.txt, each line
"x1 y1 x2 y2 x3 y3 x4 y4 class_name difficult".
"""

import os

import torch
from PIL import Image

from ...core import register
from ._dataset import DetDataset

__all__ = ['DOTADetection', 'DOTA_CLASSES']

DOTA_CLASSES = (
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
    'harbor', 'swimming-pool', 'helicopter')

DOTA_NAME2LABEL = {name: i for i, name in enumerate(DOTA_CLASSES)}


@register()
class DOTADetection(DetDataset):
    __inject__ = ['transforms', ]

    def __init__(self, img_folder, ann_folder, transforms=None, filter_empty=False, img_ext='.png'):
        """img_folder / ann_folder: a directory or a list of directories."""
        super().__init__()
        self.img_folder = img_folder
        self.ann_folder = ann_folder
        self._transforms = transforms
        self.transforms = transforms
        self.filter_empty = filter_empty
        self.img_ext = img_ext

        img_folders = [img_folder] if isinstance(img_folder, str) else list(img_folder)
        ann_folders = [ann_folder] if isinstance(ann_folder, str) else list(ann_folder)
        assert len(img_folders) == len(ann_folders), 'img_folder/ann_folder count mismatch'

        # Parse every annfile once up front (annfiles live on a network FS; caching
        # ~5MB of parsed boxes avoids per-sample txt reads during training).
        total = 0
        self.entries, self.annotations = [], []   # entries: (img_dir, name)
        for imf, anf in zip(img_folders, ann_folders):
            for name in sorted(os.path.splitext(f)[0] for f in os.listdir(anf) if f.endswith('.txt')):
                total += 1
                ann = self._parse_file(os.path.join(anf, name + '.txt'))
                if filter_empty and len(ann[1]) == 0:
                    continue
                self.entries.append((imf, name))
                self.annotations.append(ann)
        if filter_empty:
            print(f'DOTADetection: kept {len(self.entries)}/{total} images with at least one instance')

    @staticmethod
    def _parse_line(line):
        parts = line.split()
        if len(parts) < 9 or parts[8] not in DOTA_NAME2LABEL:
            return None
        try:
            poly = [float(v) for v in parts[:8]]
        except ValueError:
            return None
        difficult = int(parts[9]) if len(parts) > 9 else 0
        return poly, DOTA_NAME2LABEL[parts[8]], difficult

    @classmethod
    def _parse_file(cls, path):
        polys, labels, difficult = [], [], []
        with open(path) as f:
            for line in f:
                parsed = cls._parse_line(line)
                if parsed is None:
                    continue
                polys.append(parsed[0])
                labels.append(parsed[1])
                difficult.append(parsed[2])
        return (torch.tensor(polys, dtype=torch.float32).reshape(-1, 8),
                torch.tensor(labels, dtype=torch.int64),
                torch.tensor(difficult, dtype=torch.int64))

    def load_annotation(self, idx):
        """(polys [N,8] float32, labels [N], difficult [N]) from the in-memory cache."""
        return self.annotations[idx]

    def load_item(self, idx):
        img_dir, name = self.entries[idx]
        image = Image.open(os.path.join(img_dir, name + self.img_ext)).convert('RGB')
        polys, labels, difficult = self.load_annotation(idx)
        w, h = image.size
        target = {
            'image_id': torch.tensor([idx]),
            'polys': polys.clone(),
            'labels': labels.clone(),
            'difficult': difficult.clone(),
            'orig_size': torch.tensor([int(w), int(h)]),
            'idx': torch.tensor([idx]),
        }
        return image, target

    def __len__(self):
        return len(self.entries)

    def extra_repr(self) -> str:
        s = f' img_folder: {self.img_folder}\n ann_folder: {self.ann_folder}\n'
        s += f' num_images: {len(self.entries)}\n'
        if self._transforms is not None:
            s += f' transforms:\n   {repr(self._transforms)}'
        return s
