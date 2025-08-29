"""
HF Trainer callback for creating pytorch profiling snapshots
"""

from pathlib import Path
from pickle import dump  # nosec B403

import torch
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)


class PytorchProfilerCallback(TrainerCallback):
    """
    PyTorch Profiler callback to create snapshots of GPU memory usage at specified steps.
    """

    def __init__(self, steps_to_profile: int = 5, profiler_steps_start: int = 0):
        # steps are 0 indexed, so to start at 0-th step, we start at beginning of first step,
        # and finish at end of last step, so 5 steps_to_profile is steps [0, 1, 2, 3, 4]
        self.profiler_steps_end = profiler_steps_start + steps_to_profile - 1
        if profiler_steps_start == 0:
            # start recording memory allocations before everything is allocated, because if we start
            # at the beginning of step 0, we won't have any memory allocations in the traces
            torch.cuda.memory._record_memory_history(enabled="all")
            profiler_steps_start = -1
        self.profiler_steps_start = profiler_steps_start

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.global_step == self.profiler_steps_start:
            torch.cuda.memory._record_memory_history(enabled="all")

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.global_step == self.profiler_steps_end:
            snapshot = torch.cuda.memory._snapshot()
            with open(Path(args.output_dir) / "snapshot.pickle", "wb") as fout:
                dump(snapshot, fout)

            # tell CUDA to stop recording memory allocations now
            torch.cuda.memory._record_memory_history(enabled=None)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # make sure to record if we happen to have more steps than steps to profile
        if (
            state.global_step >= self.profiler_steps_start
            and state.global_step < self.profiler_steps_end
        ):
            snapshot = torch.cuda.memory._snapshot()
            with open(Path(args.output_dir) / "snapshot.pickle", "wb") as fout:
                dump(snapshot, fout)

            # tell CUDA to stop recording memory allocations now
            torch.cuda.memory._record_memory_history(enabled=None)
