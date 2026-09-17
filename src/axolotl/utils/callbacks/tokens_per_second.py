"""A callback for calculating tokens per second during training."""

import json
import os

import torch
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from axolotl.core.trainers.constants import TOKENS_STATE_FILE
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class TokensPerSecondCallback(TrainerCallback):
    """Restore the cumulative token counters when resuming from a checkpoint.

    Throughput itself is computed in the trainer's ``log()`` from deltas of the
    cumulative ``trainable`` counter, so it is unaffected by
    gradient_accumulation_steps and logging_steps.
    """

    def __init__(self, resume_from_checkpoint=None):
        super().__init__()
        self.resume_from_checkpoint = resume_from_checkpoint

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):  # pylint: disable=unused-argument
        """Restore total_tokens state when resuming from checkpoint."""
        if not isinstance(self.resume_from_checkpoint, str):
            return
        tokens_state_path = os.path.join(self.resume_from_checkpoint, TOKENS_STATE_FILE)
        if os.path.isfile(tokens_state_path):
            with open(tokens_state_path, "r", encoding="utf-8") as f:
                tokens_state = json.load(f)
            state.tokens = {
                "total": torch.tensor(tokens_state.get("total", 0)),
                "trainable": torch.tensor(tokens_state.get("trainable", 0)),
            }
            LOG.info(f"Restored total_tokens: {state.tokens['total']}")
