"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Ultralytics-format depth datasets packed in LMDB, for mixed metric-depth pretraining.

Logical layout follows the Ultralytics depth dataset convention
(https://docs.ultralytics.com/tasks/depth): an image key
`images/<split>/<name>.<ext>` holds the original encoded image bytes, and its
depth key — derived by swapping the leading `images/` for `depth/` and the file
extension for `.npy`, exactly the Ultralytics loader rule — holds a float32
metric-depth array in meters with invalid pixels set to 0 (<= 0 is invalid),
stored as zlib-compressed .npy bytes.  `__index__/<split>` holds the JSON list
of image keys for the split; `__meta__` records provenance.

The physical packing is LMDB rather than loose files: cold-reading ~1M small
files from network storage would starve training.
"""

import io
import json
import os
import zlib

import numpy as np
import torch
from PIL import Image

from ...core import register
from .._misc import Mask
from ._dataset import DetDataset

Image.MAX_IMAGE_PIXELS = None

__all__ = ['UltraDepthLmdb', 'derive_depth_key', 'encode_depth', 'decode_depth']


def derive_depth_key(image_key):
    """images/<split>/<name>.<ext> -> depth/<split>/<name>.npy (Ultralytics rule)."""
    assert image_key.startswith('images/'), image_key
    stem = image_key[len('images/'):].rsplit('.', 1)[0]
    return f'depth/{stem}.npy'


def encode_depth(depth):
    """float array (meters, invalid=0) -> zlib-compressed .npy bytes (float32)."""
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(depth, dtype=np.float32))
    return zlib.compress(buf.getvalue(), 1)


def decode_depth(blob):
    return np.load(io.BytesIO(zlib.decompress(blob)))


@register()
class UltraDepthLmdb(DetDataset):
    """Concat-style reader over one or more Ultralytics-format depth LMDBs.

    Mirrors the NYUDepthV2 target contract: split='train' puts the depth map in
    target['depth'] as a float Mask tv-tensor [1, H, W] so geometric transforms
    (flip / resize / crop, nearest for Mask) stay aligned with the image;
    split='val' keeps the native-resolution depth as a plain tensor for
    full-resolution evaluation.

    `repeats` oversamples small member datasets by an integer factor to balance
    the mix (e.g. SUN RGB-D's 10k pairs against TartanAir's 300k).
    """

    __inject__ = ['transforms', ]

    def __init__(self, lmdb_paths, transforms, split='train', repeats=None):
        super().__init__()
        assert split in ('train', 'val'), \
            f"split must be 'train' or 'val', got {split}"
        if isinstance(lmdb_paths, str):
            lmdb_paths = [lmdb_paths]
        self.lmdb_paths = [str(p) for p in lmdb_paths]
        repeats = repeats if repeats is not None else [1] * len(self.lmdb_paths)
        assert len(repeats) == len(self.lmdb_paths), \
            f'repeats ({len(repeats)}) must match lmdb_paths ({len(self.lmdb_paths)})'
        self.repeats = [int(r) for r in repeats]
        self.split = split
        self._transforms = transforms
        self._envs = None
        self._envs_pid = None

        # (db_index, image_key) pairs; only keys live in memory.
        self.index = []
        for i, path in enumerate(self.lmdb_paths):
            keys = self._read_index(path, split)
            assert keys, f'Empty {split} index in {path}'
            for _ in range(max(self.repeats[i], 1)):
                self.index.extend((i, k) for k in keys)
            print(f'UltraDepthLmdb[{split}]: {os.path.basename(path)} '
                  f'{len(keys)} samples x{self.repeats[i]}')
        assert self.index, f'Empty combined {split} index for {self.lmdb_paths}'

    # ------------------------------------------------------------------ lmdb
    # lmdb forbids opening the same file twice within one process, and the train
    # and val datasets naturally share LMDBs — so read-only envs are cached and
    # shared per process. A forked DataLoader worker inherits the parent's handles,
    # which are useless there yet still block a reopen; the pid guard drops them
    # first so the worker opens its own.
    _ENV_CACHE = {}
    _ENV_PID = None

    @classmethod
    def _open_env(cls, path):
        import gc
        import lmdb
        pid = os.getpid()
        if cls._ENV_PID != pid:
            cls._ENV_CACHE.clear()      # release inherited handles before reopening
            gc.collect()
            cls._ENV_PID = pid
        env = cls._ENV_CACHE.get(path)
        if env is None:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f'Depth LMDB not found: {path} (expected an Ultralytics-layout depth LMDB, '
                    f'see the module docstring).')
            env = lmdb.open(path, readonly=True, lock=False, readahead=False,
                            max_readers=4096, subdir=os.path.isdir(path))
            cls._ENV_CACHE[path] = env
        return env

    @classmethod
    def _read_index(cls, path, split):
        with cls._open_env(path).begin() as txn:
            blob = txn.get(f'__index__/{split}'.encode())
            return json.loads(blob.decode()) if blob else []

    @property
    def envs(self):
        pid = os.getpid()
        if self._envs is None or self._envs_pid != pid:
            self._envs = None       # drop this instance's inherited handles first
            self._envs = [self._open_env(p) for p in self.lmdb_paths]
            self._envs_pid = pid
        return self._envs

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_envs'] = None
        state['_envs_pid'] = None
        return state

    # --------------------------------------------------------------- loading
    def __len__(self):
        return len(self.index)

    def load_item(self, index):
        db_i, key = self.index[index]
        with self.envs[db_i].begin() as txn:
            img_blob = txn.get(key.encode())
            depth_blob = txn.get(derive_depth_key(key).encode())
        if img_blob is None or depth_blob is None:
            raise KeyError(f'Missing pair for {key} in {self.lmdb_paths[db_i]}')

        image = Image.open(io.BytesIO(img_blob)).convert('RGB')
        depth = torch.from_numpy(np.ascontiguousarray(decode_depth(depth_blob),
                                                      dtype=np.float32))
        assert depth.shape == (image.height, image.width), \
            f'{key}: depth {tuple(depth.shape)} vs image {(image.height, image.width)}'

        target = {'idx': torch.tensor([index])}
        if self.split != 'val':
            target['depth'] = Mask(depth[None])       # [1, H, W], transformed with the image
        else:
            target['depth'] = depth                   # [H, W] native resolution, untouched
        return image, target

    def __getitem__(self, index):
        img, target = self.load_item(index)
        if self._transforms is not None:
            img, target, _ = self._transforms(img, target, self)
        return img, target

    def extra_repr(self):
        s = f' split: {self.split} ({len(self.index)} samples)\n'
        for p, r in zip(self.lmdb_paths, self.repeats):
            s += f' lmdb: {p} (x{r})\n'
        if self._transforms is not None:
            s += f' transforms:\n   {repr(self._transforms)}'
        return s
