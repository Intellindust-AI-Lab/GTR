"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
"""

import atexit
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np

import torch
import torch.nn as nn

from ..core import BaseConfig
from ..misc import dist_utils


def to(m: nn.Module, device: str):
    if m is None:
        return None
    return m.to(device)


def remove_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def _state_dict_shape_filter(module: nn.Module, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Keep only tensors that match the current module shapes.

    Used when eval_spatial_size differs from training (e.g. 640 vs 1024): decoder
    anchors / valid_mask buffers are resolution-dependent and must stay model-init.
    """
    current = dist_utils.de_model(module).state_dict()
    filtered = {}
    skipped = []
    for key, value in state_dict.items():
        if key not in current:
            continue
        if current[key].shape != value.shape:
            skipped.append(key)
            continue
        filtered[key] = value
    if skipped:
        print(
            f'Skip loading {len(skipped)} checkpoint key(s) with shape mismatch '
            f'(keep current module values): {skipped}'
        )
    return filtered


_RUNTIME_FINGERPRINT_KEYS = (
    'GTR_FAST_MAL_LOSS',
    'GTR_FAST_L1_COST',
    'GTR_ELEMENTWISE_BOX_IOU',
    'GTR_FAST_GO_INDICES',
    'GTR_GPU_LSAP',
    'GTR_SYNC_FREE_DDF',
    'GTR_TENSOR_LOSS_NORM',
    'GTR_SHARE_MATCHED_INDICES',
    'GTR_SANITIZE_LOSSES',
    'GTR_TF32',
    'USE_AMP',
    'GTR_AMP_DTYPE',
    'GTR_AMP_SAFE_MODE',
    'DDP_STATIC_GRAPH',
    'DDP_GRADIENT_AS_BUCKET_VIEW',
    'DDP_BUCKET_CAP_MB',
)


def _runtime_fingerprint() -> Dict[str, str]:
    return {key: os.getenv(key, '<unset>') for key in _RUNTIME_FINGERPRINT_KEYS}


def _warn_runtime_fingerprint_mismatch(saved: Dict[str, str]):
    if not saved:
        return
    current = _runtime_fingerprint()
    mismatches = {
        key: (saved.get(key, '<unset>'), current.get(key, '<unset>'))
        for key in _RUNTIME_FINGERPRINT_KEYS
        if saved.get(key, '<unset>') != current.get(key, '<unset>')
    }
    if mismatches and dist_utils.is_main_process():
        print(
            'Warning: resume runtime fingerprint differs from checkpoint; '
            'verify this is intentional before comparing training curves.'
        )
        for key, (old, new) in mismatches.items():
            print(f'  {key}: checkpoint={old} current={new}')


def freeze_unused_params(model, criterion, device, size=640):
    """Run ONE synthetic fwd+bwd, freeze params that receive no gradient, so DDP can run
    with find_unused_parameters=False (which skips the per-iteration unused-param graph traversal —
    a free speedup on the backward). Numerically neutral: these params get no grad in the training
    graph (the optimizer was already a no-op on them; EMA still averages them via the full
    state_dict, and the optimizer builder filters by requires_grad so they drop out cleanly).
    The dead set is structural (data-independent) and deterministic, so every DDP rank freezes the
    same params. For this finetune config it is 11 params (distillation-only mask_token + final
    norm; group_detr>1 enc-head templates). When the criterion trains masks (segmentation), the
    synthetic targets carry box-shaped rectangle masks so the mask losses run and the
    SegmentationHead keeps its gradients instead of being frozen."""
    rng = torch.get_rng_state()
    crng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    was_training = model.training
    model.train(); criterion.train()
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 3, size, size, generator=g).to(device)
    with_masks = 'masks' in getattr(criterion, 'losses', ())
    targets = []
    for _ in range(2):
        cxcy = torch.rand(10, 2, generator=g) * 0.6 + 0.2
        wh = torch.rand(10, 2, generator=g) * 0.3 + 0.05
        target = {'boxes': torch.cat([cxcy, wh], 1).to(device),
                  'labels': torch.randint(0, 80, (10,), generator=g).to(device)}
        if with_masks:
            xyxy = (torch.cat([cxcy - wh / 2, cxcy + wh / 2], 1).clamp_(0, 1) * size).long()
            masks = torch.zeros(10, size, size, dtype=torch.uint8)
            for j, (x0, y0, x1, y1) in enumerate(xyxy.tolist()):
                masks[j, y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)] = 1
            target['masks'] = masks.to(device)
        targets.append(target)
    out = model(x, targets=targets)
    loss = sum(criterion(out, targets, epoch=0, step=0, global_step=0, epoch_step=1).values())
    loss.backward()
    dead = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    for _, p in model.named_parameters():
        if p.requires_grad and p.grad is None:
            p.requires_grad_(False)
    model.zero_grad(set_to_none=True)
    if not was_training:
        model.eval()
    torch.set_rng_state(rng)
    if crng is not None:
        torch.cuda.set_rng_state_all(crng)
    if dist_utils.is_main_process():
        print(f'[freeze_unused_params] froze {len(dead)} unused param(s) so '
              f'find_unused_parameters=False is safe: {dead}')
    return dead


class BaseSolver(object):
    def __init__(self, cfg: BaseConfig) -> None:
        self.cfg = cfg
        self.obj365_ids = [
            0, 46, 5, 58, 114, 55, 116, 65, 21, 40, 176, 127, 249, 24, 56, 139, 92, 78, 99, 96,
            144, 295, 178, 180, 38, 39, 13, 43, 120, 219, 148, 173, 165, 154, 137, 113, 145, 146,
            204, 8, 35, 10, 88, 84, 93, 26, 112, 82, 265, 104, 141, 152, 234, 143, 150, 97, 2,
            50, 25, 75, 98, 153, 37, 73, 115, 132, 106, 61, 163, 134, 277, 81, 133, 18, 94, 30,
            169, 70, 328, 226
        ]
    def _setup(self):
        """Avoid instantiating unnecessary classes"""
        cfg = self.cfg
        if cfg.device:
            device = torch.device(cfg.device)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = cfg.model

        # NOTE: Must load_tuning_state before EMA instance building
        if self.cfg.tuning:
            print(f'Tuning checkpoint from {self.cfg.tuning}')
            self.load_tuning_state(self.cfg.tuning)

        # With find_unused_parameters=False, DDP skips the per-iteration unused-param graph traversal
        # (a free FP32-exact speedup). But this model has structurally-dead params (distillation-only
        # mask_token/norm, group_detr>1 enc-head templates) that would make DDP hang waiting for grads
        # that never arrive — so auto-detect and freeze them first.
        if cfg.find_unused_parameters is False:
            _sz = (cfg.yaml_cfg.get('eval_spatial_size') or [640, 640])[0]
            freeze_unused_params(self.model.to(device), cfg.criterion.to(device), device, size=_sz)

        self.model = dist_utils.warp_model(
            self.model.to(device), sync_bn=cfg.sync_bn, find_unused_parameters=cfg.find_unused_parameters,
            static_graph=cfg.ddp_static_graph,
            gradient_as_bucket_view=cfg.ddp_gradient_as_bucket_view,
            bucket_cap_mb=cfg.ddp_bucket_cap_mb,
            compile=cfg.compile, compile_mode=cfg.compile_mode
        )

        self.criterion = self.to(cfg.criterion, device)
        self.postprocessor = self.to(cfg.postprocessor, device)

        self.ema = self.to(cfg.ema, device)
        self.scaler = cfg.scaler

        self.device = device
        self.last_epoch = self.cfg.last_epoch

        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.writer = cfg.writer

        if self.writer:
            atexit.register(self.writer.close)
            if dist_utils.is_main_process():
                self.writer.add_text('config', '{:s}'.format(cfg.__repr__()), 0)

    def cleanup(self):
        if self.writer:
            atexit.register(self.writer.close)

    @staticmethod
    def _load_checkpoint(path: str):
        if path.startswith('http'):
            return torch.hub.load_state_dict_from_url(path, map_location='cpu')

        try:
            return torch.load(path, map_location='cpu', weights_only=True)
        except Exception:
            return torch.load(path, map_location='cpu', weights_only=False)

    def extra_state_dict(self):
        return {}

    def load_extra_state_dict(self, state):
        return

    def train(self):
        self._setup()
        self.optimizer = self.cfg.optimizer
        self._optimizer_base_lrs = [group['lr'] for group in self.optimizer.param_groups]
        for group, base_lr in zip(self.optimizer.param_groups, self._optimizer_base_lrs):
            group['initial_lr'] = base_lr
        self.lr_scheduler = self.cfg.lr_scheduler
        self.lr_warmup_scheduler = self.cfg.lr_warmup_scheduler

        self.train_dataloader = dist_utils.warp_loader(
            self.cfg.train_dataloader, shuffle=self.cfg.train_dataloader.shuffle
        )
        self.val_dataloader = dist_utils.warp_loader(
            self.cfg.val_dataloader, shuffle=self.cfg.val_dataloader.shuffle
        )

        self.evaluator = self.cfg.evaluator

        # NOTE: Instantiating order
        if self.cfg.resume:
            print(f'Resume checkpoint from {self.cfg.resume}')
            self.load_resume_state(self.cfg.resume)

    def eval(self):
        self._setup()

        self.val_dataloader = dist_utils.warp_loader(
            self.cfg.val_dataloader, shuffle=self.cfg.val_dataloader.shuffle
        )

        self.evaluator = self.cfg.evaluator

        if self.cfg.resume:
            print(f'Resume checkpoint from {self.cfg.resume}')
            self.load_resume_state(self.cfg.resume, restore_rng_state=False)

    def to(self, module, device):
        return module.to(device) if hasattr(module, 'to') else module

    @staticmethod
    def _pack_numpy_rng_state(state) -> Dict[str, Any]:
        return {
            'bit_generator': state[0],
            'keys': state[1].tolist(),
            'pos': state[2],
            'has_gauss': state[3],
            'cached_gaussian': state[4],
        }

    @staticmethod
    def _unpack_numpy_rng_state(state: Dict[str, Any]):
        return (
            state['bit_generator'],
            np.asarray(state['keys'], dtype=np.uint32),
            state['pos'],
            state['has_gauss'],
            state['cached_gaussian'],
        )

    def capture_rng_state(self, distributed: bool=False):
        state = {
            'python': random.getstate(),
            'numpy': self._pack_numpy_rng_state(np.random.get_state()),
            'torch': torch.get_rng_state().cpu(),
        }
        if torch.cuda.is_available():
            state['cuda'] = [rng.cpu() for rng in torch.cuda.get_rng_state_all()]

        if distributed and dist_utils.is_dist_available_and_initialized():
            return {'by_rank': dist_utils.all_gather(state)}

        return state

    def restore_rng_state(self, state):
        if not state:
            return

        if 'by_rank' in state:
            states = state['by_rank']
            rank = dist_utils.get_rank()
            if rank < len(states):
                state = states[rank]
            elif len(states) == 1:
                state = states[0]
            else:
                print(f'Warning: rank {rank} has no dedicated RNG state, falling back to rank 0 state.')
                state = states[0]

        if 'python' in state:
            random.setstate(state['python'])
        if 'numpy' in state:
            np.random.set_state(self._unpack_numpy_rng_state(state['numpy']))
        if 'torch' in state:
            torch.set_rng_state(state['torch'])
        if 'cuda' in state and torch.cuda.is_available():
            cuda_states = state['cuda']
            device_count = torch.cuda.device_count()
            if len(cuda_states) != device_count:
                print(
                    f'Warning: CUDA RNG state count mismatch: checkpoint={len(cuda_states)} '
                    f'current={device_count}; restoring first {min(len(cuda_states), device_count)}.'
                )
            for idx, cuda_state in enumerate(cuda_states[:device_count]):
                torch.cuda.set_rng_state(cuda_state, idx)

    def state_dict(self):
        """State dict, train/eval"""
        state = {}
        state['date'] = datetime.now().isoformat()

        # For resume
        state['last_epoch'] = self.last_epoch
        state['rng_state'] = self.capture_rng_state(
            distributed=dist_utils.is_dist_available_and_initialized()
        )

        for k, v in self.__dict__.items():
            if hasattr(v, 'state_dict'):
                v = dist_utils.de_model(v)
                state[k] = v.state_dict()

        extra_state = self.extra_state_dict()
        state['__meta__'] = {
            'solver': extra_state,
            'runtime_fingerprint': _runtime_fingerprint(),
            # last_epoch and the data/LR schedule are chunk-scale dependent
            # (engine/data/chunked_sampler.py); guarded on resume.
            'epoch_chunks': getattr(getattr(self, 'train_dataloader', None), 'epoch_chunks', 1),
        }

        return state

    def load_state_dict(self, state, restore_rng_state: bool=True, skip_keys=None):
        """Load state dict, train/eval"""
        skip_keys = set(skip_keys or [])
        if 'last_epoch' in state:
            self.last_epoch = state['last_epoch']
            print(f'Load last_epoch {self.last_epoch}')

        for k, v in self.__dict__.items():
            if k in skip_keys:
                continue
            if hasattr(v, 'load_state_dict') and k in state:
                v = dist_utils.de_model(v)
                if k == 'model':
                    sd = _state_dict_shape_filter(v, state[k])
                    v.load_state_dict(sd, strict=False)
                elif k == 'ema':
                    ema_state = state[k]
                    mod_sd = _state_dict_shape_filter(v.module, ema_state['module'])
                    v.module.load_state_dict(mod_sd, strict=False)
                    if 'updates' in ema_state:
                        v.updates = ema_state['updates']
                else:
                    try:
                        v.load_state_dict(state[k])
                    except ValueError as exc:
                        if k != 'optimizer' or 'parameter group' not in str(exc):
                            raise
                        print(
                            'Skip optimizer.state_dict due to parameter-group mismatch. '
                            'This can happen when find_unused_parameters=False freezes '
                            f'structurally unused params before building the optimizer: {exc}'
                        )
                        continue
                    if k == 'optimizer':
                        base_lrs = getattr(self, '_optimizer_base_lrs', None)
                        if base_lrs is not None:
                            for group, base_lr in zip(v.param_groups, base_lrs):
                                group['initial_lr'] = base_lr
                print(f'Load {k}.state_dict')

            if hasattr(v, 'load_state_dict') and k not in state:
                if k == 'ema':
                    model = getattr(self, 'model', None)
                    if model is not None:
                        ema = dist_utils.de_model(v)
                        model_state_dict = remove_module_prefix(model.state_dict())
                        ema.load_state_dict({'module': model_state_dict})
                        print(f'Load {k}.state_dict from model.state_dict')
                else:
                    print(f'Not load {k}.state_dict')

        meta_state = state.get('__meta__', {})
        _warn_runtime_fingerprint_mismatch(meta_state.get('runtime_fingerprint', {}))
        saved_chunks = meta_state.get('epoch_chunks')
        loader = getattr(self, 'train_dataloader', None)
        if saved_chunks is not None and loader is not None:
            current_chunks = getattr(loader, 'epoch_chunks', 1)
            if saved_chunks != current_chunks:
                raise RuntimeError(
                    f'epoch_chunks mismatch: checkpoint was trained with epoch_chunks='
                    f'{saved_chunks}, config now has {current_chunks}. last_epoch and the '
                    f'data/LR schedule are chunk-scale dependent, so resuming would silently '
                    f'misalign them. Restore epoch_chunks (and the proportionally scaled '
                    f'epochs/flat_epoch/stop_epoch/val_freq/checkpoint_freq) or start a '
                    f'fresh output_dir.'
                )
        solver_state = meta_state.get('solver', {})
        if solver_state:
            self.load_extra_state_dict(solver_state)
            print('Load solver extra state')

        self._loaded_rng_state = False
        if restore_rng_state and 'rng_state' in state:
            self.restore_rng_state(state['rng_state'])
            self._loaded_rng_state = True
            print('Load rng_state')

    def load_resume_state(self, path: str, restore_rng_state: bool=True, skip_keys=None):
        """Load resume"""
        state = self._load_checkpoint(path)

        # state['model'] = remove_module_prefix(state['model'])
        self.load_state_dict(state, restore_rng_state=restore_rng_state, skip_keys=skip_keys)

    def load_tuning_state(self, path: str):
        """Load model for tuning and adjust mismatched head parameters"""
        if path.startswith('http'):
            state = torch.hub.load_state_dict_from_url(path, map_location='cpu', weights_only=True)
        else:
            state = self._load_checkpoint(path)

        module = dist_utils.de_model(self.model)

        # Load the appropriate state dict
        if 'ema' in state:
            pretrain_state_dict = state['ema']['module']
        else:
            pretrain_state_dict = state['model']

        # Adjust head parameters between datasets
        try:
            adjusted_state_dict = self._adjust_head_parameters(module.state_dict(), pretrain_state_dict)
            stat, infos = self._matched_state(module.state_dict(), adjusted_state_dict)
        except Exception:
            stat, infos = self._matched_state(module.state_dict(), pretrain_state_dict)

        module.load_state_dict(stat, strict=False)
        print(f'Load model.state_dict, {infos}')

    @staticmethod
    def _matched_state(state: Dict[str, torch.Tensor], params: Dict[str, torch.Tensor]):
        missed_list = []
        unmatched_list = []
        matched_state = {}
        for k, v in state.items():
            if k in params:
                if v.shape == params[k].shape:
                    matched_state[k] = params[k]
                else:
                    unmatched_list.append(k)
            else:
                missed_list.append(k)

        return matched_state, {'missed': missed_list, 'unmatched': unmatched_list}

    def _adjust_head_parameters(self, cur_state_dict, pretrain_state_dict):
        """Adjust head parameters between datasets."""
        # List of parameters to adjust
        if pretrain_state_dict['decoder.denoising_class_embed.weight'].size() != \
                cur_state_dict['decoder.denoising_class_embed.weight'].size():
            del pretrain_state_dict['decoder.denoising_class_embed.weight']

        head_param_names = [
            'decoder.enc_score_head.weight',
            'decoder.enc_score_head.bias'
        ]
        for i in range(8):
            head_param_names.append(f'decoder.dec_score_head.{i}.weight')
            head_param_names.append(f'decoder.dec_score_head.{i}.bias')

        adjusted_params = []

        for param_name in head_param_names:
            if param_name in cur_state_dict and param_name in pretrain_state_dict:
                cur_tensor = cur_state_dict[param_name]
                pretrain_tensor = pretrain_state_dict[param_name]
                adjusted_tensor = self.map_class_weights(cur_tensor, pretrain_tensor)
                if adjusted_tensor is not None:
                    pretrain_state_dict[param_name] = adjusted_tensor
                    adjusted_params.append(param_name)
                else:
                    print(f"Cannot adjust parameter '{param_name}' due to size mismatch.")

        return pretrain_state_dict

    def map_class_weights(self, cur_tensor, pretrain_tensor):
        """Map class weights from pretrain model to current model based on class IDs."""
        if pretrain_tensor.size() == cur_tensor.size():
            return pretrain_tensor

        adjusted_tensor = cur_tensor.clone()
        adjusted_tensor.requires_grad = False

        if pretrain_tensor.size() > cur_tensor.size():
            for coco_id, obj_id in enumerate(self.obj365_ids):
                adjusted_tensor[coco_id] = pretrain_tensor[obj_id+1]
        else:
            for coco_id, obj_id in enumerate(self.obj365_ids):
                adjusted_tensor[obj_id+1] = pretrain_tensor[coco_id]

        return adjusted_tensor

    def fit(self):
        raise NotImplementedError('')

    def val(self):
        raise NotImplementedError('')
