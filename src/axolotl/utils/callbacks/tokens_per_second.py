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
        total_num_tokens = state.num_tokens
        if is_distributed():
            total_num_tokens = total_num_tokens.clone()
            torch.distributed.all_reduce(total_num_tokens)
            world_size = torch.distributed.get_world_size()
            total_num_tokens = total_num_tokens.item() / world_size

        self.last_tokens_per_second = total_num_tokens / (time.perf_counter() - self.start_time)

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
