# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Trainer callback scheduling the sonicmoe merge-aware NVFP4 fake-quant."""

import json
from pathlib import Path

import torch
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


def clear_merge_aware_metadata(adapter_dir):
    """Remove a stale guarantee when a training path fell back to ordinary LoRA."""
    path = Path(adapter_dir) / "adapter_config.json"
    if path.exists():
        cfg = json.loads(path.read_text())
        if cfg.pop("nvfp4_merge_aware", None) is not None:
            path.write_text(json.dumps(cfg, indent=2))


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
        self.merge_aware_valid = True

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

    def _check_fallback(self, model):
        if model is None:
            return
        unsupported = not self.merge_aware_valid or any(
            getattr(module, "_axolotl_merge_aware_unsupported", False)
            for module in model.modules()
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            device = (
                torch.device("cuda", torch.cuda.current_device())
                if torch.distributed.get_backend() == "nccl"
                else torch.device("cpu")
            )
            flag = torch.tensor(int(unsupported), device=device)
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
            unsupported = bool(flag.item())
        if unsupported and self.merge_aware_valid:
            LOG.warning(
                "NVFP4 MERGE WARNING: at least one training path used ordinary LoRA. "
                "Continuing training without merge-aware metadata; NVFP4 merging "
                "may round away the learned adapter update."
            )
        self.merge_aware_valid = not unsupported

    def on_train_end(self, args, state, control, **kwargs):
        self._check_fallback(kwargs.get("model"))

    def on_save(self, args, state, control, **kwargs):
        self._check_fallback(kwargs.get("model"))
        if state.is_world_process_zero:
            ckpt = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            if self._enabled and self.merge_aware_valid:
                write_merge_aware_metadata(ckpt, start_step=self.start_step)
            else:
                clear_merge_aware_metadata(ckpt)
