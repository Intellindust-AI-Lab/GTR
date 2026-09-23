"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
reference
- https://github.com/pytorch/vision/blob/main/references/detection/utils.py
- https://github.com/facebookresearch/detr/blob/master/util/misc.py#L406

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import atexit
import os
import random
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn
import torch.distributed
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

# from torch.utils.data.dataloader import DataLoader
from ..data import DataLoader
from ..data.chunked_sampler import ChunkedDistributedSampler


def setup_distributed(print_rank: int=0, print_method: str='builtin', seed: int=None, ):
    """
    env setup
    args:
        print_rank,
        print_method, (builtin, rich)
        seed,
    """
    try:
        # https://pytorch.org/docs/stable/elastic/run.html
        RANK = int(os.getenv('RANK', -1))
        LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
        WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
        local_rank = LOCAL_RANK if LOCAL_RANK >= 0 else RANK

        if torch.cuda.is_available() and local_rank >= 0:
            torch.cuda.set_device(local_rank)

        # torch.distributed.init_process_group(backend=backend, init_method='env://')
        if torch.cuda.is_available() and local_rank >= 0:
            try:
                torch.distributed.init_process_group(
                    init_method='env://',
                    device_id=torch.device(f'cuda:{local_rank}'),
                )
            except TypeError:
                torch.distributed.init_process_group(init_method='env://')
            try:
                torch.distributed.barrier(device_ids=[local_rank])
            except TypeError:
                torch.distributed.barrier()
        else:
            torch.distributed.init_process_group(init_method='env://')
            torch.distributed.barrier()

        rank = torch.distributed.get_rank()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        enabled_dist = True
        if get_rank() == print_rank:
            print('Initialized distributed mode...')

    except Exception:
        enabled_dist = False
        print('Not init distributed mode.')

    setup_print(get_rank() == print_rank, method=print_method)
    if seed is not None:
        setup_seed(seed)
    else:
        configure_tf32_from_env()

    return enabled_dist


def configure_tf32_from_env():
    value = os.getenv('GTR_TF32')
    if value is None or value == '':
        return

    enabled = value.lower() in {'1', 'true', 'yes', 'on'}
    if hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = enabled
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = enabled
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high' if enabled else 'highest')

    if get_rank() == 0:
        print(
            '[tf32] '
            f'GTR_TF32={value} '
            f'matmul_allow_tf32={getattr(torch.backends.cuda.matmul, "allow_tf32", None)} '
            f'cudnn_allow_tf32={getattr(torch.backends.cudnn, "allow_tf32", None)} '
            f'float32_matmul_precision={torch.get_float32_matmul_precision() if hasattr(torch, "get_float32_matmul_precision") else "<unknown>"}',
            flush=True,
        )


def setup_print(is_main, method='builtin'):
    """This function disables printing when not in master process
    """
    import builtins as __builtin__

    if method == 'builtin':
        builtin_print = __builtin__.print

    elif method == 'rich':
        import rich
        builtin_print = rich.print

    else:
        raise AttributeError('')

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_main or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_available_and_initialized():
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True


@atexit.register
def cleanup():
    """cleanup distributed environment
    """
    if is_dist_available_and_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def get_rank():
    if not is_dist_available_and_initialized():
        return 0
    return torch.distributed.get_rank()


def get_world_size():
    if not is_dist_available_and_initialized():
        return 1
    return torch.distributed.get_world_size()


def is_main_process():
    return get_rank() == 0


def atomic_torch_save(obj, save_path, **kwargs):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # Use a unique tmp name so concurrent runs do not clobber each other.
    tmp_path = save_path.with_name(
        f'.{save_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp'
    )

    try:
        torch.save(obj, tmp_path, **kwargs)
        os.replace(tmp_path, save_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def save_on_master(*args, **kwargs):
    if not is_main_process():
        return

    if len(args) >= 2 and isinstance(args[1], (str, os.PathLike)):
        obj, save_path = args[:2]
        remaining_args = args[2:]
        if not remaining_args:
            atomic_torch_save(obj, save_path, **kwargs)
            return

    torch.save(*args, **kwargs)



def warp_model(
    model: torch.nn.Module,
    sync_bn: bool=False,
    dist_mode: str='ddp',
    find_unused_parameters: bool=False,
    static_graph: bool=False,
    gradient_as_bucket_view: bool=False,
    bucket_cap_mb: int=None,
    compile: bool=False,
    compile_mode: str='reduce-overhead',
    **kwargs
):
    if is_dist_available_and_initialized():
        device_id = torch.cuda.current_device() if torch.cuda.is_available() else None
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model) if sync_bn else model
        if dist_mode == 'dp':
            if device_id is None:
                model = DP(model)
            else:
                model = DP(model, device_ids=[device_id], output_device=device_id)
        elif dist_mode == 'ddp':
            ddp_kwargs = dict(
                find_unused_parameters=find_unused_parameters,
                gradient_as_bucket_view=gradient_as_bucket_view,
                static_graph=static_graph,
            )
            if bucket_cap_mb is not None:
                ddp_kwargs['bucket_cap_mb'] = bucket_cap_mb
            if device_id is None:
                model = DDP(model, **ddp_kwargs)
            else:
                model = DDP(
                    model,
                    device_ids=[device_id],
                    output_device=device_id,
                    **ddp_kwargs,
                )
            print(
                '[ddp] '
                f'find_unused_parameters={find_unused_parameters} '
                f'static_graph={static_graph} '
                f'gradient_as_bucket_view={gradient_as_bucket_view} '
                f'bucket_cap_mb={bucket_cap_mb}'
            )
        else:
            raise AttributeError('')

    if compile:
        # NOTE: do NOT force dynamic=True — the GLA backbone uses einops.rearrange with '...'
        # which dynamo cannot trace under forced-dynamic shapes. Default (auto) lets the static
        # backbone compile and the decoder's variable denoising dims recompile-to-dynamic.
        model = torch.compile(model, mode=compile_mode)

    return model

def de_model(model):
    return de_parallel(de_complie(model))


def warp_loader(loader, shuffle=False):
    if is_dist_available_and_initialized():
        sampler_seed = getattr(loader, 'seed', 0)
        epoch_chunks = getattr(loader, 'epoch_chunks', 1)
        if epoch_chunks > 1:
            # Anti-preemption short epochs: one segment of the base permutation
            # per epoch (see engine/data/chunked_sampler.py).
            sampler = ChunkedDistributedSampler(
                loader.dataset, epoch_chunks, shuffle=shuffle, seed=sampler_seed
            )
        else:
            try:
                sampler = DistributedSampler(loader.dataset, shuffle=shuffle, seed=sampler_seed)
            except TypeError:
                sampler = DistributedSampler(loader.dataset, shuffle=shuffle)
        worker_seed = sampler_seed + get_rank() * 1000
        generator = torch.Generator()
        generator.manual_seed(worker_seed)
        loader = DataLoader(loader.dataset,
                            loader.batch_size,
                            sampler=sampler,
                            drop_last=loader.drop_last,
                            collate_fn=loader.collate_fn,
                            pin_memory=True,
                            num_workers=loader.num_workers,
                            persistent_workers=loader.num_workers > 0,
                            multiprocessing_context='spawn' if loader.num_workers > 0 else None,
                            worker_init_fn=_seed_worker,
                            generator=generator)
        # Keep the chunking marker visible to the solver (warmup-truncation guard).
        loader.epoch_chunks = epoch_chunks
    return loader



def is_parallel(model) -> bool:
    # Returns True if model is of type DP or DDP
    return type(model) in (torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel)


def de_parallel(model) -> nn.Module:
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model


def reduce_dict(data, avg=True):
    """
    Args
        data dict: input, {k: v, ...}
        avg bool: true
    """
    world_size = get_world_size()
    if world_size < 2:
        return data

    with torch.no_grad():
        keys, values = [], []
        for k in sorted(data.keys()):
            keys.append(k)
            v = data[k]
            if not torch.is_tensor(v):
                v = torch.as_tensor(v)
            if v.numel() == 1:
                v = v.reshape(())
            values.append(v)

        values = torch.stack(values, dim=0)
        torch.distributed.all_reduce(values)

        if avg is True:
            values /= world_size

        return {k: v for k, v in zip(keys, values)}


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    torch.distributed.all_gather_object(data_list, data)
    return data_list


def sync_time():
    """sync_time
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return time.time()



def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def setup_seed(seed: int, deterministic: bool=False):
    """setup_seed for reproducibility
    torch.manual_seed(3407) is all you need. https://arxiv.org/abs/2109.08203
    """
    seed = seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available() and deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)

        if hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = False
    else:
        configure_tf32_from_env()


# for torch.compile
def check_compile():
    import warnings

    import torch
    gpu_ok = False
    if torch.cuda.is_available():
        device_cap = torch.cuda.get_device_capability()
        if device_cap in ((7, 0), (8, 0), (9, 0)):
            gpu_ok = True
    if not gpu_ok:
        warnings.warn(
            "GPU is not NVIDIA V100, A100, or H100. Speedup numbers may be lower "
            "than expected."
        )
    return gpu_ok

def is_compile(model):
    import torch._dynamo
    return type(model) in (torch._dynamo.OptimizedModule, )

def de_complie(model):
    return model._orig_mod if is_compile(model) else model
