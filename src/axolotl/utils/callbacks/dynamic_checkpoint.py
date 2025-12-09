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

DEFAULT_TRIGGER_FILENAME = "axolotl_checkpoint.save"


class DynamicCheckpointCallback(TrainerCallback):
    """
    Callback to save checkpoints on-demand during training via:
    1. File-based trigger (works everywhere, rank 0 checks file)

    Thread-safe for multi-GPU distributed training.

    Usage:
        # File-based:
        touch /path/to/output_dir/axolotl_checkpoint.save
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

        trigger_file_path = self._get_config_value(dc_config, "trigger_file_path")
        self.trigger_filename = (
            trigger_file_path if trigger_file_path else DEFAULT_TRIGGER_FILENAME
        )

        check_interval = self._get_config_value(dc_config, "check_interval")
        self.check_interval = check_interval if check_interval is not None else 100
        self.should_save_checkpoint = False

        LOG.info(
            f"Dynamic checkpoint enabled. To trigger checkpoint save:\n"
            f"  • File: touch {cfg.output_dir}/{self.trigger_filename}\n"
            f"  • Check interval: every {self.check_interval} steps",
            main_process_only=True,
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

                device = getattr(
                    args,
                    "device",
                    torch.device("cuda" if torch.cuda.is_available() else "cpu"),
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
                main_process_only=True,
            )
        return control
