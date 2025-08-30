"""Module for Axolotl trainer scheduler mixin"""

import torch
from torch.optim.lr_scheduler import LRScheduler, OneCycleLR
from transformers.trainer import Trainer

from axolotl.integrations.base import PluginManager
from axolotl.utils.logging import get_logger
from axolotl.utils.schedulers import (
    JaggedLRRestartScheduler,
    RexLR,
    get_cosine_schedule_with_min_lr,
    get_cosine_schedule_with_quadratic_warmup,
    get_cosine_schedule_with_warmup_decay_constant,
)

LOG = get_logger(__name__)


class SchedulerMixin(Trainer):
    """
    Mixin class for scheduler setup in CausalTrainer.
    """

    args = None  # type: "AxolotlTrainingArguments"  # type: ignore[name-defined]

    def create_scheduler(
        self, num_training_steps: int, optimizer: torch.optim.Optimizer = None
    ) -> LRScheduler:
        """
        Set up the scheduler. The optimizer of the trainer must have been set up either before this method is called or
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
        if self.lr_scheduler is None:  # type: ignore
            # fmt: on
            plugin_manager = PluginManager.get_instance()
            lr_scheduler: LRScheduler | None = plugin_manager.create_lr_scheduler(
                trainer=self,
                optimizer=optimizer,
                num_training_steps=num_training_steps
            )
            if lr_scheduler is not None:
                LOG.info(f"Using plugin-created lr_scheduler: {lr_scheduler}")
                self.lr_scheduler = lr_scheduler
            elif self.args.alternate_lr_scheduler_type == "one_cycle":
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
                    min_lr=0 if not use_cosine_min_lr else (
                        self.args.learning_rate * self.args.cosine_min_lr_ratio),
                    total_steps=num_training_steps,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                )
            elif use_cosine_quadratic:
                if use_cosine_min_lr:
                    LOG.warning(
                        "Both cosine quadratic warmup and min lr detected. Using quadratic warmup.")

                self.lr_scheduler = get_cosine_schedule_with_quadratic_warmup(
                    optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                )
            elif self.args.cosine_min_lr_ratio and self.args.cosine_constant_lr_ratio and use_cosine_min_lr:
                assert 0 <= self.args.cosine_min_lr_ratio <= 1.0, "cosine_min_lr_ratio must be between 0.0 and 1.0"
                assert 0 <= self.args.cosine_constant_lr_ratio <= 1.0, "cosine_constant_lr_ratio must be between 0.0 and 1.0"
                self.lr_scheduler = get_cosine_schedule_with_warmup_decay_constant(
                    optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                    min_lr_ratio=self.args.cosine_min_lr_ratio,
                    constant_lr_ratio=self.args.cosine_constant_lr_ratio,
                )
            elif self.args.cosine_min_lr_ratio and use_cosine_min_lr:
                assert 0 <= self.args.cosine_min_lr_ratio <= 1.0, "cosine_min_lr_ratio must be between 0.0 and 1.0"
                self.lr_scheduler = get_cosine_schedule_with_min_lr(
                    optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                    min_lr_ratio=self.args.cosine_min_lr_ratio,
                )
            else:
                super().create_scheduler(num_training_steps, optimizer=optimizer)
        else:
            if use_cosine_quadratic:
                LOG.warning(
                    "axolotl's cosine scheduler with quadratic warmup not used (e.g., because of deepspeed).")

            if use_cosine_min_lr:
                LOG.warning(
                    "axolotl's cosine scheduler with min lr not used (e.g., because of deepspeed).")

        if self.args.jagged_restart_steps:
            warmup_steps = (
                self.args.jagged_restart_warmup_steps or 10
            )
            anneal_steps = (
                self.args.jagged_restart_anneal_steps or 1
            )
            if not self.lr_scheduler:
                super().create_scheduler(num_training_steps, optimizer)
            self.lr_scheduler = JaggedLRRestartScheduler(
                optimizer,
                self.lr_scheduler,
                self.args.jagged_restart_steps,
                warmup_steps,
                anneal_steps,
                min_lr_scale=self.args.cosine_min_lr_ratio or 0.001,
            )

        return self.lr_scheduler  # type: ignore
