"""Checkpoint save handling: unsavable FSDP optimizers and retries after disk errors."""

import contextlib
import functools

from transformers import Trainer

from axolotl.core.checkpoint_retry import retry_disk_writes
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
            )


def with_checkpoint_write_retry(method):
    """Run a ``_save_checkpoint`` override inside the trainer's write-retry scope."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        scope = getattr(self, "checkpoint_write_retry", None)
        with scope() if callable(scope) else contextlib.nullcontext():
            return method(self, *args, **kwargs)

    return wrapper


class CheckpointWriteRetryMixin(Trainer):
    """Retry checkpoint file writes that fail with a full disk, quota or I/O error."""

    def checkpoint_write_retry(self):
        wait = getattr(self.args, "checkpoint_save_retry_wait", None)
        if wait is None:
            return contextlib.nullcontext()
        return retry_disk_writes(
            wait_seconds=wait,
            retries=getattr(self.args, "checkpoint_save_retries", 1),
        )

    def _save_checkpoint(self, *args, **kwargs):
        with self.checkpoint_write_retry():
            return super()._save_checkpoint(*args, **kwargs)
