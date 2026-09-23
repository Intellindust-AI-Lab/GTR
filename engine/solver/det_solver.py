"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 D-FINE authors. All Rights Reserved.
"""

import datetime
import json
import math
import os
import time
from pathlib import Path

import torch

from ..misc import dist_utils, stats
from ..optim.lr_scheduler import FlatCosineLRScheduler
from ._solver import BaseSolver
from .det_engine import evaluate, train_one_epoch


class DetSolver(BaseSolver):
    @staticmethod
    def _metric_value(value):
        if isinstance(value, (list, tuple)):
            return value[0]
        return value

    @staticmethod
    def _primary_metric_key(stat):
        if not stat:
            return None
        # Segmentation runs select best.pth by mask AP (GTR semantics);
        # detection runs have no masks key and keep using bbox AP.
        if 'coco_eval_masks' in stat:
            return 'coco_eval_masks'
        if 'coco_eval_bbox' in stat:
            return 'coco_eval_bbox'
        for key in stat:
            if key != 'epoch':
                return key
        return None

    @classmethod
    def _summarize_eval_stats(cls, epoch, test_stats):
        stat = {'epoch': epoch}
        for key, value in test_stats.items():
            stat[key] = cls._metric_value(value)
        return stat

    @classmethod
    def _is_better_stat(cls, candidate, reference):
        key = cls._primary_metric_key(candidate) or cls._primary_metric_key(reference)
        if key is None or candidate is None:
            return False
        if reference is None or key not in reference:
            return True
        return candidate[key] > reference[key]

    def state_dict(self):
        state = super().state_dict()
        if hasattr(self, 'best_stat') and isinstance(self.best_stat, dict):
            state['best_stat'] = dict(self.best_stat)
        return state

    def load_state_dict(self, state, restore_rng_state: bool=True, skip_keys=None):
        super().load_state_dict(
            state,
            restore_rng_state=restore_rng_state,
            skip_keys=skip_keys,
        )
        self._loaded_best_stat = 'best_stat' in state
        if self._loaded_best_stat:
            self.best_stat = dict(state['best_stat'])
            print(f'Load best_stat {self.best_stat}')

    def _evaluate_current_model(self):
        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(
            module,
            self.criterion,
            self.postprocessor,
            self.val_dataloader,
            self.evaluator,
            self.device
        )
        current_stat = self._summarize_eval_stats(self.last_epoch, test_stats)
        return current_stat, test_stats, coco_evaluator

    def _maybe_probe_legacy_best(self, resume_path):
        if getattr(self, '_loaded_best_stat', False) or not resume_path:
            return
        best_path = self.output_dir / 'best.pth'
        if not best_path.is_file():
            return

        resume_path = Path(resume_path).resolve()
        if best_path.resolve() == resume_path:
            return

        print(f'Legacy resume detected, probing {best_path} to recover historical best metric...')
        self.load_resume_state(str(best_path))
        best_stat, _, _ = self._evaluate_current_model()
        if self._is_better_stat(best_stat, self.best_stat):
            self.best_stat = best_stat
        self.load_resume_state(str(resume_path))

    def fit(self, ):
        self.train()
        args = self.cfg
        self.best_stat = dict(getattr(self, 'best_stat', {'epoch': -1}))
        profiler_rng_state = self.capture_rng_state()
        try:
            n_parameters, model_stats = stats(self.cfg)
        except Exception as e:
            print(f"Warning: FLOPs profiling failed: {e}, skipping.")
            n_parameters = sum(p.numel() for p in self.model.parameters())
            model_stats = None
        finally:
            self.restore_rng_state(profiler_rng_state)
        # n_parameters, model_stats = stats(self.cfg)
        print(model_stats)
        print("-"*42 + "Start training" + "-"*43)

        stop_aug_epoch = self.train_dataloader.dataset._transforms.stop_epoch  # epoch to stop augmentation
        if args.lrsheduler is not None:
            no_aug_epochs = args.epochs - stop_aug_epoch
            flat_epochs = self.train_dataloader.dataset._transforms.mosaic_epoch if args.flat_epoch is None else args.flat_epoch
            grad_accum_steps = max(1, int(os.getenv('GTR_GRAD_ACCUM_STEPS', '1') or 1))
            iter_per_epoch = math.ceil(len(self.train_dataloader) / grad_accum_steps)
            warmup_iter = min(args.warmup_iter, 3 * iter_per_epoch)
            # Chunked short epochs shrink iter_per_epoch; refuse to silently truncate
            # warmup.
            if getattr(self.train_dataloader, 'epoch_chunks', 1) > 1:
                assert warmup_iter == args.warmup_iter, (
                    f'warmup_iter {args.warmup_iter} would be truncated to {warmup_iter} '
                    f'(3 * iter_per_epoch = {3 * iter_per_epoch}); lower warmup_iter or epoch_chunks'
                )

            min_lr_override = getattr(args, 'min_lr', None)
            print(
                f'FlatCosineLRScheduler with flat_epochs: {flat_epochs}, '
                f'no_aug_epochs: {no_aug_epochs}, warmup_iter: {args.warmup_iter}, '
                f'min_lr: {min_lr_override}, grad_accum_steps: {grad_accum_steps}, '
                f'optimizer_steps_per_epoch: {iter_per_epoch}'
            )
            self.lr_scheduler = FlatCosineLRScheduler(self.optimizer, args.lr_gamma, iter_per_epoch, total_epochs=args.epochs,
                                                warmup_iter=warmup_iter, flat_epochs=flat_epochs, no_aug_epochs=no_aug_epochs,
                                                min_lr=min_lr_override)
            self.self_lr_scheduler = True
            if self.last_epoch >= 0:
                start_iter = (self.last_epoch + 1) * iter_per_epoch
                self.optimizer = self.lr_scheduler.step(start_iter, self.optimizer)
                print(f'Initialize optimizer lr for resumed start_iter: {start_iter}')
        else:
            self.self_lr_scheduler = False

        if self.last_epoch > 0:
            resume_rng_state = self.capture_rng_state()
            if not getattr(self, '_loaded_best_stat', False):
                current_stat, _, _ = self._evaluate_current_model()
                if self._is_better_stat(current_stat, self.best_stat):
                    self.best_stat = current_stat
                self._maybe_probe_legacy_best(args.resume)
            self.restore_rng_state(resume_rng_state)
            print(f'best_stat: {self.best_stat}')
        start_time = time.time()
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, args.epochs):

            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
                
            if epoch == stop_aug_epoch:
                if dist_utils.is_dist_available_and_initialized():
                    torch.distributed.barrier()
                best_epoch = self.best_stat.get('epoch', -1)
                best_path = self.output_dir / 'best.pth'
                if not args.no_aug_load_best:
                    print(f'Keep the epoch-{epoch - 1} weights for the no-aug stage (no_aug_load_best=False).')
                elif best_epoch >= 0 and best_epoch != self.last_epoch and best_path.is_file():
                    print(f'Load best checkpoint before no-aug stage from {best_path}')
                    self.load_resume_state(
                        str(best_path),
                        restore_rng_state=False,
                        skip_keys={'train_dataloader', 'val_dataloader'},
                    )
                else:
                    print('Current checkpoint is already the best one before no-aug stage.')
                self.last_epoch = epoch - 1
                # Refresh EMA decay at the no-aug boundary (DEIMv2 ema_restart).
                if self.ema is not None and hasattr(self.train_dataloader.collate_fn, 'ema_restart_decay'):
                    self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
                    print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')

            train_stats = train_one_epoch(
                self.self_lr_scheduler,
                self.lr_scheduler,
                self.model, 
                self.criterion, 
                self.train_dataloader, 
                self.optimizer, 
                self.device, 
                epoch, 
                max_norm=args.clip_max_norm, 
                print_freq=args.print_freq, 
                ema=self.ema, 
                scaler=self.scaler, 
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer,
                output_dir=self.output_dir,
            )

            if not self.self_lr_scheduler:  # update by epoch 
                if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                    self.lr_scheduler.step()

            self.last_epoch += 1

            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'last.pth']
                # Keep an epoch checkpoint periodically so auto-resume can fall back
                # if the rolling last.pth is unavailable after a preemption.
                if (epoch + 1) % args.checkpoint_freq == 0 or (epoch + 1) == args.epochs:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            # Evaluate every args.val_freq epochs (the final epoch is always evaluated).
            do_eval = (epoch + 1) % args.val_freq == 0 or (epoch + 1) == args.epochs
            test_stats = {}
            coco_evaluator = None
            if do_eval:
                current_stat, test_stats, coco_evaluator = self._evaluate_current_model()

                for k in test_stats:
                    if self.writer and dist_utils.is_main_process():
                        for i, v in enumerate(test_stats[k]):
                            self.writer.add_scalar(f'Test/{k}_{i}'.format(k), v, epoch)

                is_new_best = self._is_better_stat(current_stat, self.best_stat)
                if is_new_best:
                    self.best_stat = current_stat
                    if self.output_dir:
                        dist_utils.save_on_master(self.state_dict(), self.output_dir / 'last.pth')
                        dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best.pth')

                print(f'best_stat: {self.best_stat}')  # global best

            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    # bbox keeps the legacy filenames; other iou types (segm) get a suffix
                    # so seg runs also serialize the mask evaluator for offline analysis.
                    for iou_type, coco_eval in coco_evaluator.coco_eval.items():
                        suffix = '' if iou_type == 'bbox' else f'_{iou_type}'
                        filenames = [f'latest{suffix}.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}{suffix}.pth')
                        for name in filenames:
                            torch.save(coco_eval.eval, self.output_dir / "eval" / name)
            if torch.cuda.is_available():  # Just for clearing up GPU memory. You can remove it if you have enough GPU memory.
                torch.cuda.empty_cache()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    def val(self, ):
        self.eval()

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, self.evaluator, self.device)

        if self.output_dir:
            for iou_type, coco_eval in coco_evaluator.coco_eval.items():
                suffix = '' if iou_type == 'bbox' else f'_{iou_type}'
                dist_utils.save_on_master(coco_eval.eval, self.output_dir / f"eval{suffix}.pth")

        return
