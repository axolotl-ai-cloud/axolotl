"""Module for custom LRScheduler class"""
import math
from functools import partial

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler


class InterpolatingLogScheduler(LRScheduler):
    """
    A scheduler that interpolates learning rates in a logarithmic fashion
    """

    def __init__(self, optimizer, num_steps, min_lr, max_lr, last_epoch=-1):
        """A scheduler that interpolates learning rates in a logarithmic fashion

        Args:
        - optimizer: pytorch optimizer
        - num_steps: int, the number of steps over which to increase from the min_lr to the max_lr
        - min_lr: float, the minimum learning rate
        - max_lr: float, the maximum learning rate

        Usage:
            fc = nn.Linear(1,1)
            optimizer = optim.Adam(fc.parameters())
            lr_scheduler = InterpolatingLogScheduler(optimizer, num_steps=400, min_lr=1e-6, max_lr=1e-4)
        """
        self.num_steps = num_steps
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.q = (max_lr / min_lr) ** (  # pylint: disable=invalid-name
            1 / (num_steps - 1)
        )
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch <= 0:
            lrs = [self.min_lr for base_lr in self.base_lrs]
        elif self.last_epoch < self.num_steps:
            lrs = [
                self.min_lr * (self.q ** (self.last_epoch - 1))
                for base_lr in self.base_lrs
            ]
        else:
            lrs = [self.max_lr for base_lr in self.base_lrs]

        return lrs


def _get_cosine_schedule_with_quadratic_warmup_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float
):
    if current_step < num_warmup_steps:
        return (float(current_step) / float(max(1, num_warmup_steps))) ** 2
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    return max(
        0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
    )


def get_cosine_schedule_with_quadratic_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_cosine_schedule_with_quadratic_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)
