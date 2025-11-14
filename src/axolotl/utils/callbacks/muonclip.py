"""Trainer callback for invoking MuonClip controller logic."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from axolotl.muonclip.controller import MuonClipController
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

STATE_FILENAME_TEMPLATE = "muonclip_state_rank{rank}.pt"


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
            optimizer = kwargs.get("optimizer")
            self.controller.post_optimizer_step(optimizer=optimizer)
        except Exception:  # pragma: no cover - defensive logging
            LOG.exception("MuonClip controller failed during optimizer step")
            raise
        return control

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        checkpoint_dir = Path(args.output_dir) / f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        buffers = self.controller.state_dict()
        if not buffers:
            return control

        process_index = getattr(state, "process_index", 0)
        state_path = self._state_file(checkpoint_dir, process_index)
        torch.save(buffers, state_path)
        return control

    def load_state_from_checkpoint(
        self,
        checkpoint_dir: str | Path,
        *,
        process_index: int = 0,
    ) -> None:
        """
        Load Muon optimizer buffers from a checkpoint directory.
        """

        state_path = self._state_file(Path(checkpoint_dir), process_index)
        if not state_path.exists():
            LOG.debug("MuonClip state file %s not found; skipping restore", state_path)
            return

        try:
            buffers = torch.load(state_path, map_location="cpu")
        except OSError as exc:
            LOG.warning("Failed loading MuonClip state from %s: %s", state_path, exc)
            return
        self.controller.load_state_dict(buffers)

    @staticmethod
    def _state_file(directory: Path, process_index: int) -> Path:
        return directory / STATE_FILENAME_TEMPLATE.format(rank=process_index)
