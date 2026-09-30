"""A callback for calculating tokens per second during training."""

import json
import os

import torch
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from axolotl.core.trainers.constants import TOKENS_STATE_FILE
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class TokensPerSecondCallback(TrainerCallback):
    """Restore the cumulative token counters when resuming from a checkpoint.

    Throughput itself is computed in the trainer's ``log()`` from deltas of the
    cumulative ``trainable`` counter, so it is unaffected by
    gradient_accumulation_steps and logging_steps.
    """

    def __init__(self, resume_from_checkpoint=None, cfg=None):
        super().__init__()
        self.resume_from_checkpoint = resume_from_checkpoint
        self.cfg = cfg

    def _resolve_resume_from_checkpoint(self):
        # auto-resume fills in cfg.resume_from_checkpoint only after the callbacks
        # are built, so the config has to be re-read here rather than snapshotted
        if isinstance(self.resume_from_checkpoint, str):
            return self.resume_from_checkpoint
        if self.cfg is not None:
            return self.cfg.resume_from_checkpoint
        return None

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):  # pylint: disable=unused-argument
        """Restore total_tokens state when resuming from checkpoint."""
        resume_from_checkpoint = self._resolve_resume_from_checkpoint()
        if not isinstance(resume_from_checkpoint, str):
            return
        tokens_state_path = os.path.join(resume_from_checkpoint, TOKENS_STATE_FILE)
        if os.path.isfile(tokens_state_path):
            try:
                with open(tokens_state_path, "r", encoding="utf-8") as f:
                    tokens_state = json.load(f)
            except (json.JSONDecodeError, OSError):
                LOG.warning(f"Ignoring unreadable token state at {tokens_state_path}")
                return
            state.tokens = {
                "total": torch.tensor(tokens_state.get("total", 0)),
                "trainable": torch.tensor(tokens_state.get("trainable", 0)),
            }
            LOG.info(f"Restored total_tokens: {state.tokens['total']}")
