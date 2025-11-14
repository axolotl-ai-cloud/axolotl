"""Trainer callback for invoking MuonClip controller logic."""

from __future__ import annotations

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from axolotl.muonclip.controller import MuonClipController
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class MuonClipCallback(TrainerCallback):
    """Calls into the MuonClip controller after each optimizer step."""

    def __init__(self, controller: MuonClipController):
        self.controller = controller

    def on_optimizer_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        try:
            self.controller.post_optimizer_step()
        except Exception as exc:  # pragma: no cover - defensive logging
            LOG.exception("MuonClip controller failed during optimizer step: %s", exc)
            raise
        return control
