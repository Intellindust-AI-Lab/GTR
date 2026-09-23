"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 D-FINE authors. All Rights Reserved.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as T

from ...core import GLOBAL_CONFIG, register
from ._transforms import EmptyTransform

torchvision.disable_beta_transforms_warning()
import random


@register()
class Compose(T.Compose):
    def __init__(self, ops, policy=None, remove_ops=None, mosaic_epoch=-1, mosaic_prob=-1, stop_epoch=None, start_epoch=0) -> None:
        transforms = []
        if ops is not None:
            for op in ops:
                if isinstance(op, dict):
                    name = op.pop('type')
                    transform = getattr(GLOBAL_CONFIG[name]['_pymodule'], GLOBAL_CONFIG[name]['_name'])(**op)
                    transforms.append(transform)
                    op['type'] = name

                elif isinstance(op, nn.Module):
                    transforms.append(op)

                else:
                    raise ValueError('')
        else:
            transforms =[EmptyTransform(), ]

        super().__init__(transforms=transforms)

        self.mosaic_prob = mosaic_prob
        if policy is None:
            policy = 'default'
        self.global_samples = 0
        self.policy = policy
        
        self.strong_augmentation = remove_ops
        self.mosaic_epoch = mosaic_epoch
        self.stop_epoch = stop_epoch
        self.start_epoch = start_epoch  # epoch < start_epoch: warmup NoAug (DEIMv2 policy_epoch[0])

    def state_dict(self):
        return {
            'global_samples': self.global_samples,
            'transforms': [
                transform.state_dict() if hasattr(transform, 'state_dict') else None
                for transform in self.transforms
            ],
        }

    def load_state_dict(self, state):
        if not state:
            return

        self.global_samples = state.get('global_samples', self.global_samples)
        transform_states = state.get('transforms', [])
        for transform, transform_state in zip(self.transforms, transform_states):
            if transform_state is not None and hasattr(transform, 'load_state_dict'):
                transform.load_state_dict(transform_state)

    def forward(self, *inputs: Any) -> Any:
        return self.get_forward(self.policy)(*inputs)

    def get_forward(self, name):
        forwards = {
            'default': self.default_forward,
            'stop_epoch': self.stop_epoch_forward,
            'stop_sample': self.stop_sample_forward,
        }
        return forwards[name]

    def default_forward(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def stop_epoch_forward(self, *inputs: Any):
        sample = inputs if len(inputs) > 1 else inputs[0]  # image, target, dataset
        dataset = sample[-1]
        cur_epoch = dataset.epoch


        if self.mosaic_prob > 0 and self.start_epoch <= cur_epoch < self.mosaic_epoch:
            with_mosaic = random.random() <= self.mosaic_prob
        else:
            with_mosaic = False

        for transform in self.transforms:
            # Removing strong augmentation outside [start_epoch, stop_epoch): warmup NoAug + no-aug tail
            if type(transform).__name__ in self.strong_augmentation and (cur_epoch >= self.stop_epoch or cur_epoch < self.start_epoch):
                pass
             # Using Mosaic for [policy_epoch[0], policy_epoch[1]] with probability
            elif (type(transform).__name__ == 'Mosaic' and not with_mosaic):      
                pass
            # 本样本已做 Mosaic 时不再叠加大尺度抖动（epoch<mosaic_epoch 且抽到 mosaic 时）
            elif (type(transform).__name__ == 'LargeScaleJitter' and with_mosaic):
                pass
            # Mosaic and Zoomout/IoUCrop can not be co-existed in the same sample
            elif (type(transform).__name__ == 'RandomZoomOut' or type(transform).__name__ == 'RandomIoUCrop') and with_mosaic:      
                pass
            else:
                sample = transform(sample)

        return sample


    def stop_sample_forward(self, *inputs: Any):
        sample = inputs if len(inputs) > 1 else inputs[0]
        dataset = sample[-1]

        cur_epoch = dataset.epoch
        policy_ops = self.policy['ops']
        policy_sample = self.policy['sample']

        for transform in self.transforms:
            if type(transform).__name__ in policy_ops and self.global_samples >= policy_sample:
                pass
            else:
                sample = transform(sample)

        self.global_samples += 1

        return sample
