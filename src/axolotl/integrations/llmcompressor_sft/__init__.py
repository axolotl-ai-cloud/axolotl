"""
Sparse Finetuning plugin for Axolotl — enables handling of sparse neural networks
by maintaining masks for zero weights during training.
"""

import logging
from functools import wraps
from typing import Callable, TypeVar, ParamSpec, Any

from transformers.trainer import Trainer
from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl
from transformers.training_args import TrainingArguments

from ..base import BasePlugin
from llmcompressor import active_session
from llmcompressor.core import callbacks as session_callbacks
from llmcompressor.recipe import Recipe

P = ParamSpec("P")  # Params for generic function signatures
R = TypeVar("R")    # Return type for generic function signatures

LOG = logging.getLogger("axolotl.integrations.llmcompressor_sft")


class SFTCallbackHandler(TrainerCallback):
    """
    Trainer callback for Sparse Finetuning.
    Maintains sparsity patterns during training by applying masks after optimization steps,
    ensuring zero-weight updates are canceled out.
    """

    def __init__(self, trainer: Trainer, recipe: Any):
        """
        Initialize the Sparse Finetuning callback handler.

        Args:
            trainer (Trainer): Huggingface Trainer instance.
            recipe (Recipe | dict): Sparse finetuning recipe to apply.
        """
        super().__init__()
        self.trainer = trainer
        self.recipe = Recipe.model_validate(recipe) if not isinstance(recipe, Recipe) else recipe
        self.trainer.compute_loss = compute_loss_wrapper(self.trainer.compute_loss)

    def on_train_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ) -> None:
        """
        Called at the beginning of training. Initializes the compression session.

        Args:
            args (TrainingArguments): Training arguments.
            state (TrainerState): Trainer state.
            control (TrainerControl): Trainer control.
        """
        super().on_train_begin(args, state, control, **kwargs)
        session = active_session()
        session.initialize(
            model=self.trainer.model,
            optimizer=self.trainer.optimizer,
            start=state.epoch,
            recipe=self.recipe,
        )

    def on_step_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ) -> None:
        """
        Called at the beginning of a training step. Triggers batch_start callback.
        """
        super().on_step_begin(args, state, control, **kwargs)
        session_callbacks.batch_start()

    def on_step_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ) -> None:
        """
        Called at the end of a training step. Triggers optimizer and batch_end callbacks.
        """
        super().on_step_end(args, state, control, **kwargs)
        session_callbacks.optim_pre_step()
        session_callbacks.optim_post_step()
        session_callbacks.batch_end()

    def on_train_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ) -> None:
        """
        Called at the end of training. Finalizes the compression session.
        """
        super().on_train_end(args, state, control, **kwargs)
        session = active_session()
        session.finalize()


class SFTPlugin(BasePlugin):
    """
    Sparse Finetuning plugin for Axolotl integration.
    """

    def get_input_args(self) -> str:
        """
        Returns the path to the plugin's argument definition.

        Returns:
            str: Dotted path to the LLMCompressorArgs class.
        """
        return "axolotl.integrations.llmcompressor_sft.args.LLMCompressorArgs"

    def add_callbacks_post_trainer(self, cfg: Any, trainer: Trainer) -> list:
        """
        Adds Sparse Finetuning callback to the Trainer instance.

        Args:
            cfg (Any): Configuration object containing the sparse recipe.
            trainer (Trainer): Huggingface Trainer instance.

        Returns:
            list: List containing the configured callback instances.
        """
        LOG.info("Adding Sparse Finetuning callback to the trainer")
        callback = SFTCallbackHandler(
            trainer=trainer,
            recipe=cfg.llmcompressor.recipe,
        )
        return [callback]


def compute_loss_wrapper(compute_loss_func: Callable[P, R]) -> Callable[P, R]:
    """
    Wraps the loss computation function to trigger the loss_calculated callback.

    Args:
        compute_loss_func (Callable): Original loss computation function.

    Returns:
        Callable: Wrapped function that also invokes the loss_calculated callback.
    """

    @wraps(compute_loss_func)
    def compute_and_notify(*args: P.args, **kwargs: P.kwargs) -> R:
        loss = compute_loss_func(*args, **kwargs)
        session_callbacks.loss_calculated(loss=loss)
        return loss

    return compute_and_notify
