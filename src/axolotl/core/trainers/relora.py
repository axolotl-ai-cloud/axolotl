"""Module for ReLoRA trainer"""

import torch

from axolotl.core.trainers.base import AxolotlTrainer
from axolotl.monkeypatch.relora import ReLoRAScheduler


class ReLoRATrainer(AxolotlTrainer):
    """Trainer subclass that uses the `OneCycleLR` scheduler"""

    tag_names = ["axolotl", "relora"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr_scheduler = None

    def create_scheduler(
        self,
        num_training_steps: int,
        optimizer: torch.optim.Optimizer | None = None,
    ):
        optimizer = self.optimizer if optimizer is None else optimizer
        lr_scheduler = super().create_scheduler(num_training_steps, optimizer)

        if self.args.relora_steps:
            warmup_steps = (
                self.args.relora_warmup_steps if self.args.relora_warmup_steps else 10
            )
            anneal_steps = (
                self.args.relora_anneal_steps if self.args.relora_anneal_steps else 1
            )
            self.lr_scheduler = ReLoRAScheduler(
                optimizer,
                lr_scheduler,
                self.args.relora_steps,
                anneal_steps,
                warmup_steps,
            )
        else:
            self.lr_scheduler = lr_scheduler

        return self.lr_scheduler
