"""Module for Axolotl trainer scheduler mixin"""

import logging

import torch
from torch.optim.lr_scheduler import OneCycleLR
from transformers.trainer import Trainer

from axolotl.utils.schedulers import (
    RexLR,
    get_cosine_schedule_with_min_lr,
    get_cosine_schedule_with_quadratic_warmup,
    get_cosine_schedule_with_warmup_decay_constant,
)

LOG = logging.getLogger(__name__)


class SchedulerMixin(Trainer):
    """
    Mixin class for scheduler setup in CausalTrainer.
    """

    args = None  # type: "AxolotlTrainingArguments"  # type: ignore[name-defined]

    def create_scheduler(
        self, num_training_steps: int, optimizer: torch.optim.Optimizer = None
    ):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
            optimizer (torch.optim.Optimizer): The training optimizer
        """
        use_cosine_quadratic = (
            self.args.lr_scheduler_type == "cosine"
            and self.args.lr_quadratic_warmup is True
        )

        use_cosine_min_lr = (
            self.args.lr_scheduler_type == "cosine"
            and self.args.cosine_min_lr_ratio is not None
        )

        # fmt: off
        if self.lr_scheduler is None:  # type: ignore  # pylint: disable=access-member-before-definition
            # fmt: on
            if self.args.alternate_lr_scheduler_type == "one_cycle":
                num_warmup_steps = self.args.get_warmup_steps(num_training_steps)
                pct_start = num_warmup_steps / num_training_steps
                extra_lr_kwargs = {}
                if "pct_start" not in self.args.lr_scheduler_kwargs:
                    extra_lr_kwargs["pct_start"] = pct_start
                if "anneal_strategy" not in self.args.lr_scheduler_kwargs:
                    extra_lr_kwargs["anneal_strategy"] = "cos"

                self.lr_scheduler = OneCycleLR(
                    optimizer,
                    max_lr=self.args.learning_rate,
                    total_steps=num_training_steps,
                    **extra_lr_kwargs,
                    **self.args.lr_scheduler_kwargs,
                )
            elif self.args.alternate_lr_scheduler_type == "rex":
                if use_cosine_min_lr:
                    assert 0 <= self.args.cosine_min_lr_ratio <= 1.0, "cosine_min_lr_ratio must be between 0.0 and 1.0"

                self.lr_scheduler = RexLR(
                    optimizer=optimizer,
                    max_lr=self.args.learning_rate,
                    min_lr=0 if not use_cosine_min_lr else (self.args.learning_rate * self.args.cosine_min_lr_ratio),
                    total_steps=num_training_steps,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                )
            elif use_cosine_quadratic:
                if use_cosine_min_lr:
                    LOG.warning("Both cosine quadratic warmup and min lr detected. Using quadratic warmup.")

                self.lr_scheduler = get_cosine_schedule_with_quadratic_warmup(  # pylint: disable=attribute-defined-outside-init
                    optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                )
            elif self.args.cosine_min_lr_ratio and self.args.cosine_constant_lr_ratio and use_cosine_min_lr:
                assert 0 <= self.args.cosine_min_lr_ratio <= 1.0, "cosine_min_lr_ratio must be between 0.0 and 1.0"
                assert 0 <= self.args.cosine_constant_lr_ratio <= 1.0, "cosine_constant_lr_ratio must be between 0.0 and 1.0"
                self.lr_scheduler = get_cosine_schedule_with_warmup_decay_constant(  # pylint: disable=attribute-defined-outside-init
                    optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                    min_lr_ratio=self.args.cosine_min_lr_ratio,
                    constant_lr_ratio=self.args.cosine_constant_lr_ratio,
                )
            elif self.args.cosine_min_lr_ratio and use_cosine_min_lr:
                assert 0 <= self.args.cosine_min_lr_ratio <= 1.0, "cosine_min_lr_ratio must be between 0.0 and 1.0"
                self.lr_scheduler = get_cosine_schedule_with_min_lr(  # pylint: disable=attribute-defined-outside-init
                    optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                    min_lr_ratio=self.args.cosine_min_lr_ratio,
                )
            else:
                return super().create_scheduler(num_training_steps, optimizer=optimizer)
        else:
            if use_cosine_quadratic:
                LOG.warning("axolotl's cosine scheduler with quadratic warmup not used (e.g., because of deepspeed).")

            if use_cosine_min_lr:
                LOG.warning("axolotl's cosine scheduler with min lr not used (e.g., because of deepspeed).")

        return self.lr_scheduler
