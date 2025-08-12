import time

from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)


class StepTimingCallback(TrainerCallback):
    """
    A callback to measure and log the duration of each training step.
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
        self.step_time = time.perf_counter() - self.start_time

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
        logs["tokens_per_second"] = self.step_time
