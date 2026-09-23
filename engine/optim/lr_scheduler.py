"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
"""

import math
from functools import partial


def flat_cosine_schedule(total_iter, warmup_iter, flat_iter, no_aug_iter, current_iter, init_lr, min_lr):
    """
    Computes the learning rate using a warm-up, flat, and cosine decay schedule.

    Args:
        total_iter (int): Total number of iterations.
        warmup_iter (int): Number of iterations for warm-up phase.
        flat_iter (int): Number of iterations for flat phase.
        no_aug_iter (int): Number of iterations for no-augmentation phase.
        current_iter (int): Current iteration.
        init_lr (float): Initial learning rate.
        min_lr (float): Minimum learning rate.

    Returns:
        float: Calculated learning rate.
    """
    if current_iter <= warmup_iter:
        return init_lr * (current_iter / float(warmup_iter)) ** 2 if warmup_iter > 0 else init_lr
    elif warmup_iter < current_iter <= flat_iter:
        return init_lr
    elif current_iter >= total_iter - no_aug_iter:
        return min_lr
    else:
        cosine_decay = 0.5 * (1 + math.cos(math.pi * (current_iter - flat_iter) /
                                           (total_iter - flat_iter - no_aug_iter)))
        return min_lr + (init_lr - min_lr) * cosine_decay


class FlatCosineLRScheduler:
    """
    Learning rate scheduler with warm-up, optional flat phase, and cosine decay following RTMDet.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer instance.
        lr_gamma (float): Scaling factor for the minimum learning rate
            (used only when ``min_lr`` is not supplied).
        iter_per_epoch (int): Number of iterations per epoch.
        total_epochs (int): Total number of training epochs.
        warmup_epochs (int): Number of warm-up epochs.
        flat_epochs (int): Number of flat epochs (for flat-cosine scheduler).
        no_aug_epochs (int): Number of no-augmentation epochs.
        min_lr (float | list[float] | None): Optional absolute minimum learning rate
            applied uniformly (scalar) or per param-group (list). When provided, it
            overrides the ``base_lr * lr_gamma`` computation.
    """
    def __init__(self, optimizer, lr_gamma, iter_per_epoch, total_epochs,
                 warmup_iter, flat_epochs, no_aug_epochs, min_lr=None):
        self.base_lrs = [group.get("initial_lr", group["lr"]) for group in optimizer.param_groups]
        if min_lr is None:
            self.min_lrs = [base_lr * lr_gamma for base_lr in self.base_lrs]
        elif isinstance(min_lr, (list, tuple)):
            assert len(min_lr) == len(self.base_lrs), (
                f"min_lr list length {len(min_lr)} != num param groups {len(self.base_lrs)}"
            )
            self.min_lrs = [float(v) for v in min_lr]
        else:
            self.min_lrs = [float(min_lr) for _ in self.base_lrs]

        total_iter = int(iter_per_epoch * total_epochs)
        no_aug_iter = int(iter_per_epoch * no_aug_epochs)
        flat_iter = int(iter_per_epoch * flat_epochs)

        
        self.lr_func = partial(flat_cosine_schedule, total_iter, warmup_iter, flat_iter, no_aug_iter)

    def step(self, current_iter, optimizer):
        """
        Updates the learning rate of the optimizer at the current iteration.

        Args:
            current_iter (int): Current iteration.
            optimizer (torch.optim.Optimizer): Optimizer instance.
        """
        for i, group in enumerate(optimizer.param_groups):
            group["lr"] = self.lr_func(current_iter, self.base_lrs[i], self.min_lrs[i])
        return optimizer
