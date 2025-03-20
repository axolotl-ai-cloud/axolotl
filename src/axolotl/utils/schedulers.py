"""Module for custom LRScheduler class"""

import math
from functools import partial

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler


class RexLR(LRScheduler):
    """
    Reflected Exponential (REX) learning rate scheduler.

    - Original implementation: https://github.com/IvanVassi/REX_LR
    - Original license: Apache 2.0
    - Based on: https://arxiv.org/abs/2107.04197

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to schedule the learning rate for.
        max_lr (float): The maximum learning rate.
        min_lr (float): The minimum learning rate.
        total_steps (int): The total number of training steps.
        num_warmup_steps (int): The number of warmup steps.
        last_step (int): The index of last step.
    """

    def __init__(
        self, optimizer, max_lr, min_lr, total_steps=0, num_warmup_steps=0, last_step=0
    ):
        if min_lr > max_lr:
            raise ValueError(
                f'Value of "min_lr" should be less than value of "max_lr". Got min_lr={min_lr} and max_lr={max_lr}'
            )
        if num_warmup_steps > total_steps:
            raise ValueError(
                f"num_warmup_steps ({num_warmup_steps}) must be less than or equal to total_steps ({total_steps})."
            )

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.num_warmup_steps = num_warmup_steps
        self.last_step = last_step - 1

        # Ensure each parameter group has an "initial_lr" key to avoid issues when resuming.
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])

        # Pass self.last_step as last_epoch to the parent.
        super().__init__(optimizer, last_epoch=self.last_step)

    @property
    def last_step(self):
        return self.last_epoch

    @last_step.setter
    def last_step(self, value):
        self.last_epoch = value

    def get_lr(self):
        # Warmup phase: if defined, increase lr linearly from 0 to max_lr.
        if 1 <= self.last_step <= self.num_warmup_steps:
            return [
                base_lr * self.last_step / self.num_warmup_steps
                for base_lr in self.base_lrs
            ]

        # Post-warmup phase: adjust step relative to the end of warmup.
        step_after = self.last_step - self.num_warmup_steps
        remaining_steps = self.total_steps - self.num_warmup_steps

        # Avoid LR spiking
        if step_after >= remaining_steps or step_after == -1 or remaining_steps <= 0:
            return [self.min_lr for _ in self.base_lrs]

        mod_iter = step_after % remaining_steps
        z = (remaining_steps - mod_iter) / remaining_steps
        rex_factor = self.min_lr / self.max_lr + (1.0 - self.min_lr / self.max_lr) * (
            z / (0.1 + 0.9 * z)
        )
        return [base_lr * rex_factor for base_lr in self.base_lrs]


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
    num_cycles: float,
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


def _get_cosine_schedule_with_min_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float,
):
    # Warm up
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))

    # Cosine learning rate decay
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    scaling = 0.5 * (1.0 + math.cos(math.pi * progress))
    return (1 - min_lr_ratio) * scaling + min_lr_ratio


def get_cosine_schedule_with_min_lr(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
):
    """
    Create a learning rate schedule which has:
        - linear warmup from 0 -> `max_lr` over `num_warmup_steps`
        - cosine learning rate annealing from `max_lr` -> `min_lr` over `num_training_steps`
    """

    lr_lambda = partial(
        _get_cosine_schedule_with_min_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr_ratio=min_lr_ratio,
    )
    return LambdaLR(optimizer, lr_lambda)


def _get_cosine_schedule_with_warmup_decay_constant_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    constant_lr_ratio: float,
    min_lr_ratio: float,
    num_cycles: float,
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))

    num_constant_steps = int(num_training_steps * constant_lr_ratio)
    current_step = min(current_step, num_constant_steps)

    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_constant_steps - num_warmup_steps)
    )

    return (
        max(
            0,
            (1 - min_lr_ratio)
            * 0.5
            * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
        )
        + min_lr_ratio
    )


def get_cosine_schedule_with_warmup_decay_constant(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    constant_lr_ratio: float,
    min_lr_ratio: float,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Implementation of Continual Pre-Training of Large Language Models: How to (re)warm your model? (https://arxiv.org/pdf/2308.04014.pdf)
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to min_lr_ratio until num_training_steps * constant_lr_ratio, after constant_rate returns constant value of min_rate
    , after a warmup period during which it increases linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        constant_lr_ratio: (`float`):
            The ratio of num_training_steps to decrease by cosine function.
        min_lr_ratio: (`float):
            The ratio of maximum learning rate for cosine function to decay to minimum learning rate.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_decay_constant_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        constant_lr_ratio=constant_lr_ratio,
        min_lr_ratio=min_lr_ratio,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)
