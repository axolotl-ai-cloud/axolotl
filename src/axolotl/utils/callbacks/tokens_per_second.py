import time

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
        self.last_tokens_per_second = state.num_tokens / (time.perf_counter() - self.start_time)

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
