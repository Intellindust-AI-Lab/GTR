"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import multiprocessing as mp

import torch
import torch.utils.data as data


class DetDataset(data.Dataset):
    def _get_stateful_transforms(self):
        transforms = getattr(self, '_transforms', None)
        if transforms is None:
            transforms = getattr(self, 'transforms', None)
        return transforms

    def state_dict(self):
        state = {
            'epoch': self.epoch,
        }
        transforms = self._get_stateful_transforms()
        if hasattr(transforms, 'state_dict'):
            state['transforms'] = transforms.state_dict()
        return state

    def load_state_dict(self, state):
        if not state:
            return

        if 'epoch' in state:
            self._epoch = state['epoch']

        transforms = self._get_stateful_transforms()
        if 'transforms' in state and hasattr(transforms, 'load_state_dict'):
            transforms.load_state_dict(state['transforms'])

    def __getitem__(self, index):
        img, target = self.load_item(index)
        if self.transforms is not None:
            img, target, _ = self.transforms(img, target, self)
        return img, target

    def load_item(self, index):
        raise NotImplementedError("Please implement this function to return item before `transforms`.")

    def _ensure_epoch_state(self):
        state = getattr(self, '_shared_epoch_state', None)
        if state is None:
            state = mp.get_context('spawn').Value('i', -1)
            self._shared_epoch_state = state
        return state

    def set_epoch(self, epoch) -> None:
        self._epoch = epoch
        self._ensure_epoch_state().value = epoch

    @property
    def epoch(self):
        state = getattr(self, '_shared_epoch_state', None)
        if state is not None:
            return state.value
        return self._epoch if hasattr(self, '_epoch') else -1
