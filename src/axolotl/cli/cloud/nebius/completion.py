"""Distinguish normal training completion from a weights-only SIGINT exit."""

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from transformers import TrainerCallback

if TYPE_CHECKING or __package__:
    from .storage import publish
else:
    from nebius_storage import publish

from axolotl.integrations.base import BasePlugin


class NebiusCompletionCallback(TrainerCallback):
    """Record Trainer completion on the saving process."""

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            name = f"checkpoint-{state.global_step}"
            publish(
                Path(args.output_dir) / name,
                Path(os.environ["AXOLOTL_NEBIUS_EXPORT_DIR"]) / name,
                metadata={"global_step": state.global_step},
            )

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
