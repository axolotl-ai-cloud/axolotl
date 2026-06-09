"""Test-only plugin that captures param dtypes after the first optimizer step
and dumps them as JSON to ``$FP32_NORMS_DTYPE_DUMP_PATH``.

Loaded via ``plugins: [tests.e2e.multigpu._fp32_norms_dtype_capture.DtypeCapturePlugin]``
in the test yaml config; the dump path is the contract between the subprocess
and the outer pytest function. Rank 0 only — dtype is identical across ranks.
"""

from __future__ import annotations

import json
import os

import torch
from transformers.trainer_callback import TrainerCallback

from axolotl.integrations.base import BasePlugin


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


class _DtypeCaptureCallback(TrainerCallback):
    """Capture norm vs non-norm param dtypes after step 1, dump to JSON, exit."""

    def on_step_end(self, args, state, control, model=None, **kwargs):  # type: ignore[override]
        if state.global_step != 1 or model is None:
            return
        # Rank 0 only — every rank sees the same dtype info under FSDP2.
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return
        dump_path = os.environ.get("FP32_NORMS_DTYPE_DUMP_PATH")
        if not dump_path:
            return

        norm_dtypes: dict[str, str] = {}
        non_norm_dtypes: dict[str, str] = {}
        for name, param in model.named_parameters():
            entry = (name, _dtype_name(param.dtype))
            if "norm" in name.lower():
                norm_dtypes[entry[0]] = entry[1]
            else:
                non_norm_dtypes[entry[0]] = entry[1]

        with open(dump_path, "w", encoding="utf-8") as fout:
            json.dump(
                {"norms": norm_dtypes, "non_norms": non_norm_dtypes},
                fout,
                indent=2,
            )


class DtypeCapturePlugin(BasePlugin):
    """Plugin that registers :class:`_DtypeCaptureCallback` with the trainer."""

    def add_callbacks_pre_trainer(self, cfg, model):  # type: ignore[override]
        return [_DtypeCaptureCallback()]
