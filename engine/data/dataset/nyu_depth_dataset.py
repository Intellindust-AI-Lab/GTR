"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
NYU Depth V2 (labeled subset) for monocular metric depth estimation.
Data protocol follows Depth-Anything-V2 metric_depth (https://github.com/DepthAnything/Depth-Anything-V2).
"""

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ...core import register
from .._misc import Mask
from ._dataset import DetDataset

__all__ = ['NYUDepthV2']

# Official labeled subset is 1449 pairs with the standard 795/654 split
# (train_test_split from the NYU toolbox). Used when `splits_file` is absent.
_NYU_TRAIN_COUNT = 795


@register()
class NYUDepthV2(DetDataset):
    """Reads nyu_depth_v2_labeled.mat (MATLAB v7.3 / HDF5) directly via h5py.

    Layout in the .mat (h5py view, row-major reversed from MATLAB):
        images: (N, 3, W, H) uint8 RGB
        depths: (N, W, H)    float32, in-painted metric depth in meters

    split='train': target['depth'] is a float Mask tv-tensor [1, H, W] so the
    geometric train transforms (flip / resize, nearest for Mask) stay aligned
    with the image. split='test': the image goes through the val transforms
    while target['depth'] keeps the original resolution (plain tensor, ignored
    by torchvision v2 ops) for DA-V2-style full-resolution evaluation.
    """

    __inject__ = ['transforms', ]

    def __init__(self, mat_file, transforms, split='train', splits_file=None):
        super().__init__()
        assert split in ('train', 'test'), f"split must be 'train' or 'test', got {split}"
        self.mat_file = str(mat_file)
        self.splits_file = str(splits_file) if splits_file else None
        self.split = split
        self._transforms = transforms
        self._h5 = None
        self._h5_pid = None

        self.indexes = self._build_split()

    # ------------------------------------------------------------------ split
    def _build_split(self):
        with self._open_mat() as h5:
            num_samples = h5['images'].shape[0]

        if self.splits_file and Path(self.splits_file).exists():
            import scipy.io as sio
            mat = sio.loadmat(self.splits_file)
            key = 'trainNdxs' if self.split == 'train' else 'testNdxs'
            # MATLAB indices are 1-based column vectors
            indexes = np.asarray(mat[key]).reshape(-1).astype(np.int64) - 1
            print(f'NYUDepthV2[{self.split}]: official split from {self.splits_file}, {len(indexes)} samples')
        else:
            rng = np.random.default_rng(0)
            perm = rng.permutation(num_samples)
            n_train = _NYU_TRAIN_COUNT if num_samples == 1449 else int(round(num_samples * 0.8))
            indexes = np.sort(perm[:n_train]) if self.split == 'train' else np.sort(perm[n_train:])
            print(f'NYUDepthV2[{self.split}]: splits_file not found, deterministic fallback split '
                  f'(seed=0), {len(indexes)}/{num_samples} samples')

        assert len(indexes) > 0, f'Empty {self.split} split for {self.mat_file}'
        return indexes

    # ------------------------------------------------------------- h5 handling
    def _open_mat(self):
        import h5py
        if not Path(self.mat_file).exists():
            raise FileNotFoundError(
                f'NYU mat file not found: {self.mat_file}. '
                'Download nyu_depth_v2_labeled.mat and set dataset.mat_file accordingly.')
        return h5py.File(self.mat_file, 'r')

    @property
    def h5(self):
        # One handle per process: h5py handles cannot be pickled to spawn workers
        # nor shared across processes.
        pid = os.getpid()
        if self._h5 is None or self._h5_pid != pid:
            self._h5 = self._open_mat()
            self._h5_pid = pid
        return self._h5

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_h5'] = None
        state['_h5_pid'] = None
        return state

    # ---------------------------------------------------------------- loading
    def __len__(self):
        return len(self.indexes)

    def _read_pair(self, mat_index):
        img = np.transpose(self.h5['images'][mat_index], (2, 1, 0))      # -> (H, W, 3) RGB
        depth = np.transpose(self.h5['depths'][mat_index], (1, 0))       # -> (H, W) meters
        return Image.fromarray(np.ascontiguousarray(img)), np.ascontiguousarray(depth, dtype=np.float32)

    def load_item(self, index):
        mat_index = int(self.indexes[index])
        image, depth = self._read_pair(mat_index)
        depth = torch.from_numpy(depth)

        target = {'idx': torch.tensor([mat_index])}
        if self.split == 'train':
            target['depth'] = Mask(depth[None])       # [1, H, W], transformed with the image
        else:
            target['depth'] = depth                   # [H, W] original resolution, untouched
        return image, target

    def __getitem__(self, index):
        img, target = self.load_item(index)
        if self._transforms is not None:
            img, target, _ = self._transforms(img, target, self)
        return img, target

    def extra_repr(self) -> str:
        s = f' mat_file: {self.mat_file}\n split: {self.split} ({len(self.indexes)} samples)\n'
        if self._transforms is not None:
            s += f' transforms:\n   {repr(self._transforms)}'
        return s
