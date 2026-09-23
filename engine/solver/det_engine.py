"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""

import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

import torch
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter

from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils
from ..optim import ModelEMA


def train_one_epoch(self_lr_scheduler, lr_scheduler, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('img_s', SmoothedValue(window_size=20, fmt='{avg:.1f}'))
    header = 'Epoch: [{}]'.format(epoch)

    print_freq = kwargs.get('print_freq', 10)
    writer :SummaryWriter = kwargs.get('writer', None)

    ema :ModelEMA = kwargs.get('ema', None)
    scaler :GradScaler = kwargs.get('scaler', None)
    lr_warmup_scheduler = kwargs.get('lr_warmup_scheduler', None)
    perf_log = os.getenv('GTR_PERF_LOG', '1') != '0'
    profile_times = os.getenv('GTR_PROFILE_TIMES', '0') == '1'
    flops_per_image = float(os.getenv('GTR_FLOPS_PER_IMAGE', '0') or 0)
    peak_tflops_per_gpu = float(os.getenv('GTR_PEAK_TFLOPS_PER_GPU', '0') or 0)
    reduce_loss_every = max(1, int(os.getenv('GTR_REDUCE_LOSS_EVERY', '1') or 1))
    zero_grad_set_to_none = os.getenv('GTR_ZERO_GRAD_SET_TO_NONE', '0') == '1'
    grad_accum_steps = max(1, int(os.getenv('GTR_GRAD_ACCUM_STEPS', '1') or 1))
    world_size = dist_utils.get_world_size()
    rank = dist_utils.get_rank()
    output_dir = kwargs.get('output_dir', None)
    torch_profiler = _maybe_create_torch_profiler(output_dir, rank)
    amp_dtype = _get_amp_dtype(device)
    amp_safe_mode = os.getenv('GTR_AMP_SAFE_MODE', 'full' if scaler is not None else 'off')
    use_amp = scaler is not None
    use_grad_scaler = use_amp and amp_dtype is torch.float16

    if profile_times:
        metric_logger.add_meter('fwd_ms', SmoothedValue(window_size=20, fmt='{avg:.1f}'))
        metric_logger.add_meter('crit_ms', SmoothedValue(window_size=20, fmt='{avg:.1f}'))
        metric_logger.add_meter('bwd_opt_ms', SmoothedValue(window_size=20, fmt='{avg:.1f}'))
    if flops_per_image > 0:
        metric_logger.add_meter('tflops_gpu', SmoothedValue(window_size=20, fmt='{avg:.2f}'))
        if peak_tflops_per_gpu > 0:
            metric_logger.add_meter('mfu', SmoothedValue(window_size=20, fmt='{avg:.2f}'))

    effective_steps_per_epoch = math.ceil(len(data_loader) / grad_accum_steps)
    if perf_log and dist_utils.is_main_process():
        dataset_len = len(getattr(data_loader, 'dataset', []))
        print(
            '[train-infra] '
            f'epoch={epoch} world={world_size} rank_batch={data_loader.batch_size} '
            f'micro_global_batch={data_loader.batch_size * world_size} '
            f'grad_accum_steps={grad_accum_steps} '
            f'effective_global_batch={data_loader.batch_size * world_size * grad_accum_steps} '
            f'workers/rank={data_loader.num_workers} micro_steps/epoch={len(data_loader)} '
            f'optimizer_steps/epoch={effective_steps_per_epoch} '
            f'dataset_len/rank_view={dataset_len} amp={use_amp} amp_dtype={_amp_dtype_name(amp_dtype)} '
            f'amp_safe_mode={amp_safe_mode} '
            f'share_matched_indices={os.getenv("GTR_SHARE_MATCHED_INDICES", "0")} '
            f'fast_mal_loss={os.getenv("GTR_FAST_MAL_LOSS", "0")} '
            f'fast_l1_cost={os.getenv("GTR_FAST_L1_COST", "1")} '
            f'elementwise_box_iou={os.getenv("GTR_ELEMENTWISE_BOX_IOU", "1")} '
            f'fast_go_indices={os.getenv("GTR_FAST_GO_INDICES", "1")} '
            f'gpu_lsap={os.getenv("GTR_GPU_LSAP", "0")} '
            f'sync_free_ddf={os.getenv("GTR_SYNC_FREE_DDF", "1")} '
            f'tensor_loss_norm={os.getenv("GTR_TENSOR_LOSS_NORM", "1")} '
            f'grad_scaler={use_grad_scaler} '
            f'tf32_matmul={getattr(torch.backends.cuda.matmul, "allow_tf32", None)} '
            f'tf32_cudnn={getattr(torch.backends.cudnn, "allow_tf32", None)} '
            f'profile_times={profile_times} torch_profiler={torch_profiler is not None} '
            f'reduce_loss_every={reduce_loss_every} '
            f'zero_grad_set_to_none={zero_grad_set_to_none}',
            flush=True,
        )
        if flops_per_image > 0:
            print(
                '[train-infra] '
                f'flops_per_image={flops_per_image:.4e} '
                f'peak_tflops_per_gpu={peak_tflops_per_gpu:.2f}',
                flush=True,
            )

    cur_iters = epoch * effective_steps_per_epoch

    if torch_profiler is not None:
        torch_profiler.start()

    try:
        for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            optimizer = _train_step(
                self_lr_scheduler,
                lr_scheduler,
                model,
                criterion,
                data_loader,
                optimizer,
                device,
                epoch,
                max_norm,
                samples,
                targets,
                i,
                cur_iters,
                metric_logger,
                writer,
                ema,
                scaler,
                lr_warmup_scheduler,
                world_size,
                profile_times,
                flops_per_image,
                peak_tflops_per_gpu,
                amp_dtype,
                use_grad_scaler,
                print_freq,
                reduce_loss_every,
                zero_grad_set_to_none,
                grad_accum_steps,
            )
            if torch_profiler is not None:
                torch_profiler.step()
    finally:
        if torch_profiler is not None:
            torch_profiler.stop()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def _parse_rank_set(value: str):
    ranks = set()
    for part in value.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            start, end = part.split('-', 1)
            ranks.update(range(int(start), int(end) + 1))
        else:
            ranks.add(int(part))
    return ranks


def _maybe_create_torch_profiler(output_dir, rank: int):
    if os.getenv('GTR_TORCH_PROFILER', '0') != '1':
        return None
    rank_spec = os.getenv('GTR_TORCH_PROFILE_RANKS', '0')
    if rank not in _parse_rank_set(rank_spec):
        return None

    trace_dir = os.getenv('GTR_TORCH_PROFILE_DIR', '')
    if trace_dir:
        trace_root = Path(trace_dir)
    else:
        if output_dir is None:
            trace_root = Path('profiler_traces')
        else:
            trace_root = Path(output_dir) / 'profiler_traces'
    trace_root.mkdir(parents=True, exist_ok=True)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    wait = int(os.getenv('GTR_TORCH_PROFILE_WAIT', '2'))
    warmup = int(os.getenv('GTR_TORCH_PROFILE_WARMUP', '2'))
    active = int(os.getenv('GTR_TORCH_PROFILE_ACTIVE', '4'))
    repeat = int(os.getenv('GTR_TORCH_PROFILE_REPEAT', '1'))
    if dist_utils.is_main_process():
        print(
            '[torch-profiler] '
            f'enabled ranks={rank_spec} trace_dir={trace_root} '
            f'schedule=wait{wait}/warmup{warmup}/active{active}/repeat{repeat}',
            flush=True,
        )

    return torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=wait,
            warmup=warmup,
            active=active,
            repeat=repeat,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            str(trace_root),
            worker_name=f'rank{rank}',
        ),
        record_shapes=os.getenv('GTR_TORCH_PROFILE_SHAPES', '0') == '1',
        profile_memory=os.getenv('GTR_TORCH_PROFILE_MEMORY', '1') == '1',
        with_stack=os.getenv('GTR_TORCH_PROFILE_STACK', '0') == '1',
    )


def _amp_dtype_name(dtype):
    if dtype is torch.bfloat16:
        return 'bf16'
    if dtype is torch.float16:
        return 'fp16'
    return 'none'


def _get_amp_dtype(device):
    if str(device).startswith('cuda') and os.getenv('GTR_AMP_DTYPE', 'bf16').lower() in {'bf16', 'bfloat16'}:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if dist_utils.is_main_process():
            print('[amp] bf16 requested but unsupported; falling back to fp16 autocast', flush=True)
    return torch.float16


def _float_outputs_for_loss(value):
    if isinstance(value, torch.Tensor):
        return value.float() if value.is_floating_point() else value
    if isinstance(value, dict):
        return {k: _float_outputs_for_loss(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_float_outputs_for_loss(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_float_outputs_for_loss(v) for v in value)
    return value


def _target_lengths(targets, key):
    lengths = []
    for target in targets:
        value = target.get(key)
        if isinstance(value, torch.Tensor):
            lengths.append(int(value.shape[0]))
    return lengths


def _tensor_shape(value):
    if isinstance(value, torch.Tensor):
        return tuple(value.shape)
    tensors = getattr(value, 'tensors', None)
    if isinstance(tensors, torch.Tensor):
        return tuple(tensors.shape)
    return type(value).__name__


def _cuda_mem_summary_mb():
    if not torch.cuda.is_available():
        return 'cuda=unavailable'
    MB = 1024.0 * 1024.0
    return (
        f'allocated={torch.cuda.memory_allocated() / MB:.0f}MB '
        f'reserved={torch.cuda.memory_reserved() / MB:.0f}MB '
        f'max_allocated={torch.cuda.max_memory_allocated() / MB:.0f}MB '
        f'max_reserved={torch.cuda.max_memory_reserved() / MB:.0f}MB'
    )


def _log_batch_memory_diag(samples, targets, epoch, step, global_step, phase, error=None):
    rank = dist_utils.get_rank()
    target_counts = _target_lengths(targets, 'labels')
    box_counts = _target_lengths(targets, 'boxes')
    if target_counts:
        target_summary = (
            f'targets_sum={sum(target_counts)} targets_max={max(target_counts)} '
            f'targets_min={min(target_counts)}'
        )
    else:
        target_summary = 'targets_sum=NA targets_max=NA targets_min=NA'
    if box_counts and box_counts != target_counts:
        target_summary += (
            f' boxes_sum={sum(box_counts)} boxes_max={max(box_counts)} '
            f'boxes_min={min(box_counts)}'
        )
    message = (
        f'[mem-diag] rank={rank} phase={phase} epoch={epoch} step={step} '
        f'global_step={global_step} sample_shape={_tensor_shape(samples)} '
        f'local_batch={len(targets)} {target_summary} {_cuda_mem_summary_mb()}'
    )
    if error is not None:
        message += f' error={type(error).__name__}: {error}'
    print(message, flush=True)


def _is_cuda_oom(error):
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(error, RuntimeError) and 'CUDA out of memory' in str(error)


def _train_step(
    self_lr_scheduler,
    lr_scheduler,
    model,
    criterion,
    data_loader,
    optimizer,
    device,
    epoch,
    max_norm,
    samples,
    targets,
    i,
    cur_iters,
    metric_logger,
    writer,
    ema,
    scaler,
    lr_warmup_scheduler,
    world_size,
    profile_times,
    flops_per_image,
    peak_tflops_per_gpu,
    amp_dtype,
    use_grad_scaler,
    print_freq,
    reduce_loss_every,
    zero_grad_set_to_none,
    grad_accum_steps,
):
    iter_start = time.time()
    accum_group_start = (i // grad_accum_steps) * grad_accum_steps
    accum_group_end = min(accum_group_start + grad_accum_steps, len(data_loader))
    accum_denom = accum_group_end - accum_group_start
    is_accum_start = i == accum_group_start
    is_accum_end = (i + 1) == accum_group_end
    update_step = i // grad_accum_steps
    sync_context = (
        model.no_sync()
        if hasattr(model, 'no_sync') and not is_accum_end
        else nullcontext()
    )
    with torch.profiler.record_function('gtr/h2d_batch'):
        samples = samples.to(device, non_blocking=True)
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
    global_step = cur_iters + update_step
    metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))
    mem_diag_every = int(os.getenv('GTR_MEM_DIAG_EVERY', '0') or 0)
    if mem_diag_every > 0 and (i % mem_diag_every == 0 or i == len(data_loader) - 1):
        _log_batch_memory_diag(samples, targets, epoch, i, global_step, 'sample')
    if is_accum_start:
        with torch.profiler.record_function('gtr/zero_grad'):
            optimizer.zero_grad(set_to_none=zero_grad_set_to_none)
        # Set this update's LR BEFORE optimizer.step(). FlatCosineLRScheduler is STATELESS
        # -- it writes param_groups[i]['lr'] from the iteration handed to it -- so PyTorch's
        # "step the scheduler after the optimizer" convention does not apply to it. Calling
        # it afterwards made update j run at f(j-1), and update 0 run at the optimizer's
        # constructor LR (the full base LR, warmup entirely bypassed) because no call had
        # happened yet. It also made resume differ from uninterrupted training, since
        # det_solver pre-seeds f(start_iter) on resume only.
        if self_lr_scheduler:
            with torch.profiler.record_function('gtr/lr_scheduler'):
                optimizer = lr_scheduler.step(global_step, optimizer)

    if scaler is not None:
        if profile_times and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        with sync_context:
            with torch.profiler.record_function('gtr/forward'), torch.autocast(
                device_type=str(device),
                dtype=amp_dtype,
                cache_enabled=True,
            ):
                outputs = model(samples, targets=targets)
            if profile_times and torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.time()

            with torch.profiler.record_function('gtr/nan_check'):
                # pose models predict keypoints instead of boxes; depth models predict dense maps
                if 'pred_boxes' in outputs:
                    pred_reg = outputs['pred_boxes']
                elif 'pred_keypoints' in outputs:
                    pred_reg = outputs['pred_keypoints']
                else:
                    pred_reg = outputs['pred_depth']
                if torch.isnan(pred_reg).any() or torch.isinf(pred_reg).any():
                    print(pred_reg)
                    state = model.state_dict()
                    new_state = {}
                    for key, value in model.state_dict().items():
                        new_key = key.replace('module.', '')
                        state[new_key] = value
                    new_state['model'] = state
                    dist_utils.save_on_master(new_state, "./NaN.pth")

            with torch.profiler.record_function('gtr/criterion'), torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(_float_outputs_for_loss(outputs), targets, **metas)
            if profile_times and torch.cuda.is_available():
                torch.cuda.synchronize()
            t2 = time.time()

            loss = sum(loss_dict.values())
            backward_loss = loss / accum_denom
            if use_grad_scaler:
                with torch.profiler.record_function('gtr/backward'):
                    try:
                        scaler.scale(backward_loss).backward()
                    except Exception as error:
                        if os.getenv('GTR_OOM_DIAG', '1') != '0' and _is_cuda_oom(error):
                            _log_batch_memory_diag(samples, targets, epoch, i, global_step, 'backward', error)
                        raise

                if is_accum_end:
                    if max_norm > 0:
                        with torch.profiler.record_function('gtr/grad_clip'):
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, foreach=True)

                    with torch.profiler.record_function('gtr/optimizer_step'):
                        scaler.step(optimizer)
                        scaler.update()
            else:
                with torch.profiler.record_function('gtr/backward'):
                    try:
                        backward_loss.backward()
                    except Exception as error:
                        if os.getenv('GTR_OOM_DIAG', '1') != '0' and _is_cuda_oom(error):
                            _log_batch_memory_diag(samples, targets, epoch, i, global_step, 'backward', error)
                        raise

                if is_accum_end:
                    if max_norm > 0:
                        with torch.profiler.record_function('gtr/grad_clip'):
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, foreach=True)

                    with torch.profiler.record_function('gtr/optimizer_step'):
                        optimizer.step()
        if profile_times and torch.cuda.is_available():
            torch.cuda.synchronize()
        t3 = time.time()

    else:
        if profile_times and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        with sync_context:
            with torch.profiler.record_function('gtr/forward'):
                outputs = model(samples, targets=targets)
            if profile_times and torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.time()
            with torch.profiler.record_function('gtr/criterion'):
                loss_dict = criterion(outputs, targets, **metas)
            if profile_times and torch.cuda.is_available():
                torch.cuda.synchronize()
            t2 = time.time()

            loss: torch.Tensor = sum(loss_dict.values())
            backward_loss = loss / accum_denom
            with torch.profiler.record_function('gtr/backward'):
                try:
                    backward_loss.backward()
                except Exception as error:
                    if os.getenv('GTR_OOM_DIAG', '1') != '0' and _is_cuda_oom(error):
                        _log_batch_memory_diag(samples, targets, epoch, i, global_step, 'backward', error)
                    raise

            if is_accum_end:
                if max_norm > 0:
                    with torch.profiler.record_function('gtr/grad_clip'):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, foreach=True)

                with torch.profiler.record_function('gtr/optimizer_step'):
                    optimizer.step()
        if profile_times and torch.cuda.is_available():
            torch.cuda.synchronize()
        t3 = time.time()

    if ema is not None and is_accum_end:
        with torch.profiler.record_function('gtr/ema_update'):
            ema.update(model)

    if is_accum_end:
        # The self_lr_scheduler branch moved to before optimizer.step() (see is_accum_start
        # above). lr_warmup_scheduler is a stateful torch scheduler, so it keeps the
        # step-after-the-optimizer convention it was written for.
        if not self_lr_scheduler and lr_warmup_scheduler is not None:
            with torch.profiler.record_function('gtr/lr_scheduler'):
                lr_warmup_scheduler.step()

    detailed_loss_log = (
        reduce_loss_every <= 1
        or i % reduce_loss_every == 0
        or i % print_freq == 0
        or i == len(data_loader) - 1
    )
    with torch.profiler.record_function('gtr/loss_reduce_logging'):
        if detailed_loss_log:
            loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
            loss_value = sum(loss_dict_reduced.values())
        else:
            loss_dict_reduced = None
            loss_value = loss.detach()

    if not math.isfinite(loss_value):
        print("Loss is {}, stopping training".format(loss_value))
        print(loss_dict if loss_dict_reduced is None else loss_dict_reduced)
        sys.exit(1)

    if loss_dict_reduced is not None:
        metric_logger.update(loss=loss_value, **loss_dict_reduced)
    else:
        metric_logger.update(loss=loss_value)
    metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    iter_elapsed = max(time.time() - iter_start, 1e-9)
    global_img_s = samples.shape[0] * world_size / iter_elapsed
    metric_logger.update(img_s=global_img_s)
    if flops_per_image > 0:
        tflops_gpu = flops_per_image * global_img_s / world_size / 1e12
        metric_logger.update(tflops_gpu=tflops_gpu)
        if peak_tflops_per_gpu > 0:
            metric_logger.update(mfu=100.0 * tflops_gpu / peak_tflops_per_gpu)
    if profile_times:
        metric_logger.update(
            fwd_ms=(t1 - t0) * 1000,
            crit_ms=(t2 - t1) * 1000,
            bwd_opt_ms=(t3 - t2) * 1000,
        )

    if writer and dist_utils.is_main_process() and global_step % 10 == 0:
        with torch.profiler.record_function('gtr/tensorboard_write'):
            writer.add_scalar('Loss/total', loss_value.item(), global_step)
            writer.add_scalar('Perf/img_s', global_img_s, global_step)
            if flops_per_image > 0:
                writer.add_scalar('Perf/tflops_gpu', tflops_gpu, global_step)
                if peak_tflops_per_gpu > 0:
                    writer.add_scalar('Perf/mfu', 100.0 * tflops_gpu / peak_tflops_per_gpu, global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
            if loss_dict_reduced is not None:
                for k, v in loss_dict_reduced.items():
                    writer.add_scalar(f'Loss/{k}', v.item(), global_step)
            if profile_times:
                writer.add_scalar('Perf/fwd_ms', (t1 - t0) * 1000, global_step)
                writer.add_scalar('Perf/crit_ms', (t2 - t1) * 1000, global_step)
                writer.add_scalar('Perf/bwd_opt_ms', (t3 - t2) * 1000, global_step)

    return optimizer


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessor, data_loader, coco_evaluator: CocoEvaluator, device):
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessor.keys())
    iou_types = coco_evaluator.iou_types
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        results = postprocessor(outputs, orig_target_sizes)

        # if 'segm' in postprocessor.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessor['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator.labels is not None and 'bbox' in iou_types:

        import numpy as np
        from tabulate import tabulate

        precisions = coco_evaluator.coco_eval['bbox'].eval['precision']   # (T, R, K, A, M)        
        ap = np.mean(precisions[..., 0, -1], axis=(0, 1)) * 100   # (K,)
        ap_50 = np.mean(precisions[0, :, :, 0, -1], axis=0) * 100   # (K,)

        table_data = [
            (name, f'{ap[k]:.2f}', f'{ap_50[k]:.2f}')
            for k, name in enumerate(coco_evaluator.labels)]
        print(tabulate(table_data, headers=['class', 'AP', 'AP50'], tablefmt='pretty'))
        
    
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
        if 'keypoints' in iou_types:
            stats['coco_eval_keypoints'] = coco_evaluator.coco_eval['keypoints'].stats.tolist()

    return stats, coco_evaluator
