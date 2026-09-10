"""Distinguish normal training completion from a weights-only SIGINT exit."""

import json
import os
from pathlib import Path

from transformers import TrainerCallback

from axolotl.integrations.base import BasePlugin


class NebiusCompletionCallback(TrainerCallback):
    """Record Trainer completion on the saving process."""

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            Path(os.environ["AXOLOTL_NEBIUS_COMPLETION_FILE"]).write_text(
                json.dumps(
                    {"global_step": state.global_step, "max_steps": state.max_steps}
                ),
                encoding="utf-8",
            )


class NebiusCompletionPlugin(BasePlugin):
    """Attach the completion callback to the remote trainer."""

    def add_callbacks_pre_trainer(self, cfg, model):
        return [NebiusCompletionCallback()]
