"""
Sparse Finetuning plugin for Axolotl - enables handling of sparse neural networks
by maintaining masks for zero weights during training.
"""

import logging
from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl
from transformers.training_args import TrainingArguments

from ..base import BasePlugin
from .args import LLMCompressorArgs  # pylint: disable=unused-import. # noqa: F401
from llmcompressor import initialize
from llmcompressor.core import callbacks as session_callbacks
from llmcompressor.recipe import Recipe

LOG = logging.getLogger("axolotl.integrations.llmcompressor_sft")

class SFTCallbackHandler(TrainerCallback):
    """
    Transformer trainer callback for Sparse Finetuning.
    Maintains sparsity patterns during training by applying masks after optimization steps.
    This ensures that optimizer updates to zero weights are canceled out.
    """

    def __init__(self, trainer: object, recipe: object):
        """
        Initialize the callback handler.
        
        Args:
            trainer (object): The trainer instance.
            recipe (object): The sparse finetuning recipe to be applied.
        """
        super().__init__()
        self.trainer = trainer
        self.recipe = Recipe.model_validate(recipe)

        if hasattr(self.trainer, "compute_loss"):
            self.trainer.compute_loss = compute_loss_wrapper(self.trainer.compute_loss)

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event triggered at the beginning of training.
        Updates the session reference to the model, accommodating changes due to wrappers like FSDP.
        """
        super().on_train_begin(args, state, control, **kwargs)
        initialize(
            model=self.trainer.model,
            optimizer=self.trainer.optimizer,
            start=state.epoch,
            recipe=self.recipe,
        )

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event triggered at the beginning of a training step.
        Calls batch_start in the active CompressionSession.
        """
        super().on_step_begin(args, state, control, **kwargs)
        session_callbacks.batch_start()

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event triggered at the end of a training step.
        Calls optimizer pre-step, post-step, and batch_end callbacks.
        """
        super().on_step_end(args, state, control, **kwargs)
        session_callbacks.optim_pre_step()
        session_callbacks.optim_post_step()
        session_callbacks.batch_end()

    def on_substep_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event triggered at the end of a substep during gradient accumulation.
        Calls batch_end in the active CompressionSession.
        """
        super().on_substep_end(args, state, control, **kwargs)
        session_callbacks.batch_end()
    
    # def on_prediction_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    #     super().on_prediction_step(args, state, control, **kwargs)
    #     session_callbacks.loss_calculated()

class SFTPlugin(BasePlugin):
    """
    Plugin for Sparse Finetuning integration with Axolotl.
    """

    def get_input_args(self) -> str:
        """
        Returns the input argument path for the plugin.
        """
        return "axolotl.integrations.llmcompressor_sft.LLMCompressorArgs"

    def add_callbacks_post_trainer(self, cfg, trainer):
        """
        Adds Sparse Finetuning callback to the trainer.
        
        Args:
            cfg (object): Configuration object containing the recipe.
            trainer (object): Trainer instance to which the callback is added.
        
        Returns:
            list: A list containing the Sparse Finetuning callback.
        """
        LOG.info("Adding Sparse Finetuning callback to the trainer")
        callback = SFTCallbackHandler(
            trainer=trainer,
            recipe=cfg.recipe,
        )
        return [callback]


def compute_loss_wrapper(compute_loss_func):
    """
    Wraps the loss computation function to integrate with the active CompressionSession.
    
    Args:
        compute_loss_func (function): The original loss computation function.
    
    Returns:
        function: Wrapped function that reports the computed loss.
    """
    def wrapper(*args, **kwargs):
        loss = compute_loss_func(*args, **kwargs)
        session_callbacks.loss_calculated(loss=loss)
        # take the mean across multiple GPUs
        # this is done outside the compute_loss function in the parent
        loss = loss.mean()
        return loss
    return wrapper