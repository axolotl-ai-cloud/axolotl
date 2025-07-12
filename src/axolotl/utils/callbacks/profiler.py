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
        self.profiler_steps_start = profiler_steps_start
        self.profiler_steps_end = profiler_steps_start + steps_to_profile

    def on_step_begin(  # pylint: disable=unused-argument
        self,
        args: TrainingArguments,  # pylint: disable=unused-argument
        state: TrainerState,
        control: TrainerControl,  # pylint: disable=unused-argument
        **kwargs,  # pylint: disable=unused-argument
    ):
        if state.global_step == self.profiler_steps_start:
            torch.cuda.memory._record_memory_history(  # pylint: disable=protected-access
                enabled="all"
            )

    def on_step_end(  # pylint: disable=unused-argument
        self,
        args: TrainingArguments,  # pylint: disable=unused-argument
        state: TrainerState,
        control: TrainerControl,  # pylint: disable=unused-argument
        **kwargs,  # pylint: disable=unused-argument
    ):
        if state.global_step == self.profiler_steps_end:
            snapshot = torch.cuda.memory._snapshot()  # pylint: disable=protected-access
            with open(Path(args.output_dir) / "snapshot.pickle", "wb") as fout:
                dump(snapshot, fout)

            # tell CUDA to stop recording memory allocations now
            torch.cuda.memory._record_memory_history(  # pylint: disable=protected-access
                enabled=None
            )
