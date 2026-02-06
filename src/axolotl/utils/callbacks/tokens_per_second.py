"""A callback for calculating tokens per second during training."""

import json
import os
import time

import torch
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

TOKENS_STATE_FILE = "tokens_state.json"


class TokensPerSecondCallback(TrainerCallback):
    """
    A callback to measure and log tokens per second during training.
    Also handles saving/restoring total_tokens state across checkpoint resumes.
    """

    def __init__(
        self, tensor_parallel_size, context_parallel_size, resume_from_checkpoint=None
    ):
        super().__init__()
        self.step_time = 0.0
        self.start_time = 0.0
        self.non_data_parallel_size = 1
        self.resume_from_checkpoint = resume_from_checkpoint
        if tensor_parallel_size is not None:
            self.non_data_parallel_size *= tensor_parallel_size
        if context_parallel_size is not None:
            self.non_data_parallel_size *= context_parallel_size

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

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):  # pylint: disable=unused-argument
        if not hasattr(state, "tokens"):
            state.tokens = {"trainable": torch.zeros(1), "total": torch.zeros(1)}
        self.start_time = time.perf_counter()
        state.last_tokens_per_second = torch.zeros(1)

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):  # pylint: disable=unused-argument
        tokens = getattr(state, "tokens", None)
        if tokens and "trainable_tokens" in tokens:
            step_time = time.perf_counter() - self.start_time
            num_tokens_per_device = tokens["trainable_tokens"].clone()
            # non data parallel groups have duplicated tokens, so we avoid double-counting
            num_tokens_per_device = num_tokens_per_device / self.non_data_parallel_size
            state.last_tokens_per_second = num_tokens_per_device / step_time

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs,
    ):  # pylint: disable=unused-argument
        # after logging, clear the running metrics
        if hasattr(state, "last_tokens_per_second"):
            logs["tokens/train_per_sec_per_gpu"] = state.last_tokens_per_second.item()
            state.last_tokens_per_second.zero_()
        tokens = getattr(state, "tokens", None)
        # Clear per-step tokens after logging
        if tokens and "trainable_tokens" in tokens:
            tokens["trainable_tokens"] = torch.zeros_like(tokens["trainable_tokens"])
