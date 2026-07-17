# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Trainer callback scheduling the sonicmoe merge-aware NVFP4 fake-quant."""

import json
from pathlib import Path

from transformers import TrainerCallback

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def write_merge_aware_metadata(adapter_dir, start_step=None) -> bool:
    """Record the quantizer identity in the adapter's adapter_config.json.

    ``merge-lora`` reads this to select the matching writer mode and to
    hard-error on a quantizer mismatch; without it the adapter merges as an
    unprepared one (reuse-grid) and the retention guarantee is silently void.
    PEFT ignores the extra key on load.
    """
    path = Path(adapter_dir) / "adapter_config.json"
    if not path.exists():
        return False
    try:
        import torchao

        encoder = f"torchao-{torchao.__version__}"
    except ImportError:
        encoder = None
    cfg = json.loads(path.read_text())
    cfg["nvfp4_merge_aware"] = {
        "scale_mode": "fresh",
        "pts_policy": "base_fused_max",
        "encoder": encoder,
        "start_step": start_step,
    }
    path.write_text(json.dumps(cfg, indent=2))
    return True


class MergeAwareScheduleCallback(TrainerCallback):
    """Turn on the merge-aware fake-quant forward at ``start_step``.

    ``start_step``: int = absolute optimizer step; float in (0, 1) = fraction
    of ``state.max_steps``; None/0 = on from the first step. Once on it stays
    on (including final eval/save), so the saved adapter's last forward is the
    merged model.
    """

    def __init__(self, start_step: int | float | None = None):
        self.start_step = start_step or 0
        self._enabled = False

    def _threshold(self, state) -> int:
        if isinstance(self.start_step, float) and 0 < self.start_step < 1:
            return int(self.start_step * state.max_steps)
        return int(self.start_step)

    def _maybe_enable(self, state) -> None:
        if self._enabled:
            return
        if state.global_step >= self._threshold(state):
            from axolotl.integrations.kernels.libs.sonicmoe import (
                set_merge_aware_enabled,
            )

            set_merge_aware_enabled(True)
            self._enabled = True
            LOG.info(
                "merge-aware NVFP4 fake-quant enabled at step %d", state.global_step
            )

    def on_train_begin(self, args, state, control, **kwargs):
        self._maybe_enable(state)

    def on_step_begin(self, args, state, control, **kwargs):
        self._maybe_enable(state)

    def on_save(self, args, state, control, **kwargs):
        # stamp only adapters that actually trained through the fake-quant;
        # a pre-warm-up checkpoint is an unprepared adapter and must merge as one
        if self._enabled and state.is_world_process_zero:
            ckpt = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            write_merge_aware_metadata(ckpt, start_step=self.start_step)
