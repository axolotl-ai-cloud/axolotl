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

    def compute_loss_context_manager(self):
        """
        A helper wrapper to group together context managers.
        """
        ctx_stack = contextlib.ExitStack()

        autocast_ctx = self.autocast_smart_context_manager()
        if not isinstance(autocast_ctx, contextlib.nullcontext):
            ctx_stack.enter_context(autocast_ctx)

        if self.args.activation_offloading:
            activations_handling_ctx = get_act_offloading_ctx_manager(self.model)
            ctx_stack.enter_context(activations_handling_ctx)

        return ctx_stack


def ac_wrap_hf_model(model: nn.Module, **kwargs):
    auto_wrap_policy = ModuleWrapPolicy(set((GradientCheckpointingLayer,)))
    apply_activation_checkpointing(model, auto_wrap_policy=auto_wrap_policy, **kwargs)
