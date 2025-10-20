"""A callback for calculating tokens per second during training."""

import time

import torch
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)


class TokensPerSecondCallback(TrainerCallback):
    """
    A callback to measure and log tokens per second during training.
    """

    def __init__(self, tensor_parallel_size, context_parallel_size):
        super().__init__()
        self.step_time = 0.0
        self.start_time = 0.0
        self.non_data_parallel_size = 1
        if tensor_parallel_size is not None:
            self.non_data_parallel_size *= tensor_parallel_size
        if context_parallel_size is not None:
            self.non_data_parallel_size *= context_parallel_size

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):  # pylint: disable=unused-argument
        self.start_time = time.perf_counter()
        state.last_tokens_per_second = torch.zeros(1)

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):  # pylint: disable=unused-argument
        if hasattr(state, "num_tokens"):
            step_time = time.perf_counter() - self.start_time
            num_tokens_per_device = state.num_tokens.clone()
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
            state.last_tokens_per_second.zero_()
            state.num_tokens = torch.zeros(1)
