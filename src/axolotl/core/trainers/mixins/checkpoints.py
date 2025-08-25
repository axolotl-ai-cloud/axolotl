"""Custom handling to not fail training if fsdp optimizer is not savable"""

from transformers import Trainer

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class CheckpointSaveMixin(Trainer):
    """Mixin to handle saving the optimizer and scheduler if they are not savable."""

    def _save_optimizer_and_scheduler(self, output_dir):
        try:
            super()._save_optimizer_and_scheduler(output_dir)
        except (NotImplementedError, KeyError) as exc:
            # TODO: fix fsdp2 optimizer saving
            LOG.warning_once(
                f"Trainer does not support saving optimizer and scheduler:  {exc}\n"
                "Optimizer and scheduler states were not saved - resuming from checkpoints "
                "for this training run will not be possible.",
                main_process_only=True,
            )
