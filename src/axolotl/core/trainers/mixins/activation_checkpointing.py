"""
Trainer mixin for activation checkpointing w offloading
"""

import contextlib

from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from transformers import GradientCheckpointingLayer, Trainer
from trl.models.activation_offloading import get_act_offloading_ctx_manager


class ActivationOffloadingMixin(Trainer):
    """
    Trainer mixin class for activation checkpointing w offloading
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.args.activation_offloading:
            self.activation_offload_context = get_act_offloading_ctx_manager(self.model)
        else:
            self.activation_offload_context = contextlib.nullcontext()

    def training_step(self, *args, **kwargs):
        with self.activation_offload_context:
            return super().training_step(*args, **kwargs)


def ac_wrap_hf_model(model: nn.Module, **kwargs):
    auto_wrap_policy = ModuleWrapPolicy(set((GradientCheckpointingLayer,)))
    apply_activation_checkpointing(model, auto_wrap_policy=auto_wrap_policy, **kwargs)
