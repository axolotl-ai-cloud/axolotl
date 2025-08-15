import time
import torch
from axolotl.utils.distributed import is_distributed
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

    def __init__(self):
        super().__init__()
        self.step_time = 0.0
        self.start_time = 0.0

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the beginning of a training step.
        """
        self.start_time = time.perf_counter()
        self.last_tokens_per_second = 0.0

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the end of a training step.
        """
        step_time = time.perf_counter() - self.start_time
        num_tokens_per_device = state.num_tokens
        if is_distributed():
            # non data parallel groups have duplicated tokens, so we avoid double-counting
            num_tokens_per_device /= self.state.parallelism_config.non_data_parallel_size

        self.last_tokens_per_second = num_tokens_per_device / step_time

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs,
    ):
        """
        Event called when logging is done.
        """
        logs["tokens_per_second_per_gpu"] = round(self.last_tokens_per_second, 2)
        self.last_tokens_per_second = 0.0
        state.num_tokens = 0
