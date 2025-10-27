import os
import signal
from pathlib import Path

from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from axolotl.utils.distributed import (
    barrier,
    is_distributed,
    is_main_process,
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

    def _get_config_value(self, config, key, default=None):
        """Helper to get config value from dict or object."""
        if isinstance(config, dict):
            return config.get(key, default)
        return getattr(config, key, default)

    def __init__(self, cfg):
        self.cfg = cfg
        if not cfg.dynamic_checkpoint or not cfg.dynamic_checkpoint.enabled:
            self.enabled = False
            return

        self.enabled = True
        dc_config = cfg.dynamic_checkpoint

        trigger_path = self._get_config_value(dc_config, "trigger_file_path")
        self.trigger_filename = (
            trigger_path if trigger_path else DEFAULT_TRIGGER_FILENAME
        )

        check_interval = self._get_config_value(dc_config, "check_interval")
        self.check_interval = check_interval if check_interval is not None else 100

        enable_signal = self._get_config_value(dc_config, "enable_signal")
        self.enable_signal = enable_signal if enable_signal is not None else False
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

    def _signal_handler(self, _signum, _frame):
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
                    except OSError as exc:
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

                if hasattr(args, "device"):
                    device = args.device
                else:
                    device = (
                        torch.device("cuda", torch.cuda.current_device())
                        if torch.cuda.is_available()
                        else torch.device("cpu")
                    )
                trigger_tensor = torch.tensor(
                    1 if trigger_detected else 0,
                    dtype=torch.long,
                    device=device,
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
