"""
Sparsity mask plugin for Axolotl - enables handling of sparse neural networks
by maintaining masks for zero weights during training.
"""

import logging

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from ..base import BasePlugin
from llmcompressor import initialize
from llmcompressor.recipe import Recipe
from llmcompressor.core import callbacks as session_callbacks

LOG = logging.getLogger("axolotl.integrations.llmcompressor")


class SFTCallbacks(TrainerCallback):
    """
    TrainerCallback for triggering CompressionSession callbacks in the training loop.
    Used to update the model reference (for running with FSDP) and trigger the post-
    optimizer callbacks in each modifier.

    Args:
        trainer: LLM Compressor trainer that will call back into this object
        *args: Arguments to be passed to base TrainerCallback
        **kwargs: Keyword arguments to be passed to base TrainerCallback
    """

    def __init__(self, trainer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainer = trainer

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the beginning of training. Updates the session reference to the
        model, as it will have changed to a wrapper if FSDP is enabled.
        """
        super().on_train_begin(args, state, control, **kwargs)
        initialize(
            model=self.trainer.model,
            optimizer=self.trainer.optimizer,
            start=state.epoch,
        )

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        super().on_step_begin(args, state, control, **kwargs)
        session_callbacks.batch_start()

    def on_pre_optimizer_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        super().on_pre_optimizer_step(args, state, control, **kwargs)
        session_callbacks.loss_calculated()
        session_callbacks.optim_pre_step()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the end of a training step. If using gradient accumulation,
        one training step might take several inputs.

        Triggers optimizer post_step and batch_end in the active CompressionSession.
        """
        super().on_step_end(args, state, control, **kwargs)
        session_callbacks.optim_post_step()
        session_callbacks.batch_end()

    def on_substep_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the end of a substep during gradient accumulation.

        Triggers optimizer post_step and batch_end in the active CompressionSession.
        """
        super().on_substep_end(args, state, control, **kwargs)
        session_callbacks.optim_post_step()


class SFTPlugin(BasePlugin):
    """
    Plugin for Sparse Fine-tuning integration with llm-compressor.
    """

    def get_input_args(self) -> str:
        return "axolotl.integrations.llmcompressor.LLMCompressorArgs"

    def add_callbacks_post_trainer(self, cfg, trainer):
        LOG.info("Adding SparsityMask callback to the trainer")
        initialize(recipe=Recipe.model_validate(cfg.recipe), model=trainer.model)
        return [SFTCallbacks(trainer)]