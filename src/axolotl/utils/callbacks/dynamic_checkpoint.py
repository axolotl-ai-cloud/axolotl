import signal
import os
from pathlib import Path
from typing import Optional
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from axolotl.utils.distributed import (
    is_main_process,
    barrier,
    is_distributed,
)
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

DEFAULT_TRIGGER_FILENAME = ".axolotl_save_checkpoint"


class DynamicCheckpointCallback(TrainerCallback):
    """
    Callback to save checkpoints on-demand during training via:
    1. File-based trigger (works everywhere, rank 0 checks file)
    2. Signal-based trigger (Unix/Linux only, rank 0 sets flag)

    Thread-safe for multi-GPU distributed training.
    Addresses issue: https://github.com/axolotl-ai-cloud/axolotl/issues/3169

    Usage:
        # File-based:
        touch /path/to/output_dir/.axolotl_save_checkpoint
        # Signal-based (Unix/Linux):
        kill -SIGUSR1 <training_pid>
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.enabled = cfg.dynamic_checkpoint.enabled

        if not self.enabled:
            return

        self.trigger_filename = (
            cfg.dynamic_checkpoint.trigger_file_path or DEFAULT_TRIGGER_FILENAME
        )

        self.check_interval = cfg.dynamic_checkpoint.check_interval
        self.enable_signal = cfg.dynamic_checkpoint.enable_signal

        self.should_save_checkpoint = False

        if self.enable_signal and hasattr(signal, "SIGUSR1") and is_main_process():
            signal.signal(signal.SIGUSR1, self._signal_handler)
            LOG.info(
                f"Dynamic checkpoint: Signal handler registered on rank 0 (SIGUSR1). "
                f"Trigger with: kill -SIGUSR1 {os.getpid()}",
                main_process_only=True,
            )

        LOG.info(
            f"Dynamic checkpoint enabled. To trigger checkpoint save:\n"
            f"  • File: touch {cfg.output_dir}/{self.trigger_filename}\n"
            f"  • Signal: kill -SIGUSR1 <pid> {'(enabled)' if self.enable_signal else '(disabled)'}\n"
            f"  • Check interval: every {self.check_interval} steps",
            main_process_only=True,
        )

    def _signal_handler(self, signum, frame):
        """Handle SIGUSR1 signal to trigger checkpoint (rank 0 only)"""
        self.should_save_checkpoint = True
        LOG.info(
            "Dynamic checkpoint triggered via SIGUSR1 signal", main_process_only=True
        )

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **_kwargs,
    ) -> TrainerControl:
        """
        Check for checkpoint triggers at the end of each step.
        ONLY rank 0 checks the file, then all ranks synchronize.
        """
        if not self.enabled:
            return control

        trigger_detected = False

        if state.global_step % self.check_interval == 0:
            if is_main_process():
                trigger_path = Path(args.output_dir) / self.trigger_filename

                if trigger_path.exists():
                    trigger_detected = True
                    try:
                        trigger_path.unlink()  # Delete the trigger file
                        LOG.info(
                            f"Dynamic checkpoint triggered via file '{self.trigger_filename}' "
                            f"at step {state.global_step}",
                            main_process_only=True,
                        )
                    except Exception as exc:
                        LOG.warning(
                            f"Failed to delete trigger file: {exc}",
                            main_process_only=True,
                        )

                if self.should_save_checkpoint:
                    trigger_detected = True
                    self.should_save_checkpoint = False  # Reset flag

            if is_distributed():
                import torch
                import torch.distributed as dist

                trigger_tensor = torch.tensor(
                    1 if trigger_detected else 0,
                    dtype=torch.long,
                    device=f"cuda:{torch.cuda.current_device()}",
                )

                dist.broadcast(trigger_tensor, src=0)

                trigger_detected = bool(trigger_tensor.item())

                barrier()

        if trigger_detected:
            control.should_save = True
            LOG.info(
                f"Saving dynamic checkpoint at step {state.global_step}",
                main_process_only=False,  # Log on all ranks for debugging
            )
        return control
