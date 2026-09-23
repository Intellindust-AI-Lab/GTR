"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Transforms for oriented boxes carried as 8-point polygons in target['polys']
(pixel domain). The final ConvertDOTABoxes converts polygons into normalized
(cx, cy, w, h, a) rboxes for the model, a = theta / (pi/2), theta in [0, pi/2).
"""

import math
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2.functional as VF
from PIL import Image
from torchvision.transforms import InterpolationMode

from ...core import register

__all__ = ['RandomOBBFlip', 'RandomOBBRotate', 'RandomOBBLargeScaleJitter', 'ConvertDOTABoxes']


def _unpack(inputs):
    sample = inputs if len(inputs) > 1 else inputs[0]
    if len(sample) == 3:
        return sample[0], sample[1], sample[2]
    if len(sample) == 2:
        return sample[0], sample[1], None
    raise ValueError('expects (image, target) or (image, target, dataset)')


def _pack(image, target, dataset):
    if dataset is None:
        return image, target
    return image, target, dataset


@register()
class RandomOBBFlip(nn.Module):
    """Random horizontal / vertical / diagonal flip on image + target['polys'].

    With prob p one of the given directions is drawn uniformly (mmrotate RandomFlip).
    """

    def __init__(self, p=0.75, directions=('horizontal', 'vertical', 'diagonal')):
        super().__init__()
        self.p = p
        self.directions = list(directions)

    def forward(self, *inputs):
        image, target, dataset = _unpack(inputs)
        if random.random() < self.p:
            direction = random.choice(self.directions)
            w, h = image.size
            polys = target['polys'].reshape(-1, 4, 2).clone()
            if direction == 'horizontal':
                image = VF.hflip(image)
                polys[..., 0] = w - polys[..., 0]
            elif direction == 'vertical':
                image = VF.vflip(image)
                polys[..., 1] = h - polys[..., 1]
            else:  # diagonal: horizontal + vertical together (mmrotate 'diagonal')
                image = image.transpose(Image.Transpose.ROTATE_180)
                polys[..., 0] = w - polys[..., 0]
                polys[..., 1] = h - polys[..., 1]
            target['polys'] = polys.reshape(-1, 8)
        return _pack(image, target, dataset)


@register()
class ConvertDOTABoxes(nn.Module):
    """polys (N, 8) pixels -> target['boxes'] (N, 5) normalized sigmoid-domain rboxes."""

    def __init__(self, min_size=1.0):
        super().__init__()
        self.min_size = min_size

    def forward(self, *inputs):
        image, target, dataset = _unpack(inputs)
        size = image.shape[-2:] if isinstance(image, torch.Tensor) else image.size[::-1]
        h, w = int(size[0]), int(size[1])
        polys = target.pop('polys').numpy().astype(np.float32)

        boxes = np.zeros((len(polys), 5), dtype=np.float32)
        for i, poly in enumerate(polys):
            (cx, cy), (bw, bh), angle = cv2.minAreaRect(poly.reshape(4, 2))
            # long-edge regularization (paper width_longer=True, start_angle=0):
            # w is the longer side, theta in [0, pi)
            theta = math.radians(angle) % math.pi
            if bw < bh:
                bw, bh = bh, bw
                theta = (theta + math.pi / 2) % math.pi
            boxes[i] = (cx, cy, bw, bh, theta)

        keep = (boxes[:, 2] >= self.min_size) & (boxes[:, 3] >= self.min_size)
        boxes = boxes[keep]
        boxes[:, 0] /= w
        boxes[:, 1] /= h
        boxes[:, 2] /= w
        boxes[:, 3] /= h
        boxes[:, 4] = np.clip(boxes[:, 4] / math.pi, 0.0, 1.0 - 1e-6)

        target['boxes'] = torch.from_numpy(np.clip(boxes, 0.0, 1.0))
        keep_t = torch.from_numpy(keep)
        target['labels'] = target['labels'][keep_t]
        if 'difficult' in target:
            target['difficult'] = target['difficult'][keep_t]
        return _pack(image, target, dataset)


@register()
class RandomOBBLargeScaleJitter(nn.Module):
    """Large scale jittering for OBB polys (pixel domain, PIL image).

    With prob p, resize the image by a uniform scale in [min_scale, max_scale];
    scale < 1 pads bottom-right back to the input size (polys unchanged after
    scaling), scale > 1 random-crops back (polys whose center leaves the crop
    are dropped, same rule as RandomOBBRotate). The torchvision-v2 based
    LargeScaleJitter in _transforms.py must NOT be used on OBB data: it only
    updates tv_tensor boxes and would leave target['polys'] untouched.

    stop_epoch (read from dataset.epoch, mp-shared across workers) turns the op
    into an identity from that epoch on, independently of the Compose-level
    stop_epoch that gates flip/rotate.
    """

    def __init__(self, p=0.5, min_scale=0.1, max_scale=2.0, stop_epoch=None, fill=0):
        super().__init__()
        self.p = p
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.stop_epoch = stop_epoch
        self.fill = fill

    def forward(self, *inputs):
        image, target, dataset = _unpack(inputs)
        if self.stop_epoch is not None and dataset is not None and dataset.epoch >= self.stop_epoch:
            return _pack(image, target, dataset)
        if random.random() >= self.p:
            return _pack(image, target, dataset)

        w, h = image.size
        scale = random.uniform(self.min_scale, self.max_scale)
        new_w = max(1, round(w * scale))
        new_h = max(1, round(h * scale))
        image = VF.resize(image, [new_h, new_w], interpolation=InterpolationMode.BILINEAR)

        polys = target['polys'].reshape(-1, 4, 2).clone()
        polys[..., 0] *= new_w / w
        polys[..., 1] *= new_h / h

        pad_r, pad_b = max(0, w - new_w), max(0, h - new_h)
        if pad_r or pad_b:
            image = VF.pad(image, [0, 0, pad_r, pad_b], fill=self.fill)
        if new_w > w or new_h > h:
            left = random.randint(0, max(0, new_w - w))
            top = random.randint(0, max(0, new_h - h))
            image = VF.crop(image, top, left, h, w)
            polys[..., 0] -= left
            polys[..., 1] -= top
            centers = polys.mean(dim=1)
            keep = (centers[:, 0] >= 0) & (centers[:, 0] < w) & (centers[:, 1] >= 0) & (centers[:, 1] < h)
            polys = polys[keep]
            target['labels'] = target['labels'][keep]
            if 'difficult' in target:
                target['difficult'] = target['difficult'][keep]
        target['polys'] = polys.reshape(-1, 8)
        return _pack(image, target, dataset)


@register()
class RandomOBBRotate(nn.Module):
    """Random rotation around the image center (mmrotate RandomRotate).

    With prob p rotate by a uniform angle in [-angle_range, angle_range) degrees;
    if the image contains any label from rect_obj_labels (square-like classes such as
    storage-tank / roundabout), the angle is drawn from {90, 180, -90, -180}. Boxes
    whose center leaves the canvas are dropped. Bilinear interpolation as in mmcv.
    """

    def __init__(self, p=0.5, angle_range=180, rect_obj_labels=(9, 11), fill=0):
        super().__init__()
        assert 0 < angle_range <= 180
        self.p = p
        self.angle_range = angle_range
        self.rect_obj_labels = set(rect_obj_labels)
        self.horizontal_angles = [90, 180, -90, -180]
        self.fill = fill

    def forward(self, *inputs):
        image, target, dataset = _unpack(inputs)
        if random.random() >= self.p:
            return _pack(image, target, dataset)

        labels = target['labels']
        if len(labels) > 0 and any(int(l) in self.rect_obj_labels for l in labels.tolist()):
            angle = float(random.choice(self.horizontal_angles))
        else:
            angle = self.angle_range * (2 * random.random() - 1)

        w, h = image.size
        image = VF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR, fill=self.fill)

        # VF.rotate(angle) rotates content counter-clockwise on screen; in the
        # y-down pixel frame points transform as p' = c + R(-a) (p - c) with
        # R the mathematically ccw matrix, i.e. [[cos, sin], [-sin, cos]].
        a = math.radians(angle)
        cos_a, sin_a = math.cos(a), math.sin(a)
        rot = torch.tensor([[cos_a, sin_a], [-sin_a, cos_a]], dtype=torch.float32)
        center = torch.tensor([(w - 1) / 2, (h - 1) / 2], dtype=torch.float32)

        polys = target['polys'].reshape(-1, 4, 2)
        polys = (polys - center) @ rot.T + center
        centers = polys.mean(dim=1)
        keep = (centers[:, 0] >= 0) & (centers[:, 0] < w) & (centers[:, 1] >= 0) & (centers[:, 1] < h)
        target['polys'] = polys[keep].reshape(-1, 8)
        target['labels'] = target['labels'][keep]
        if 'difficult' in target:
            target['difficult'] = target['difficult'][keep]
        return _pack(image, target, dataset)
