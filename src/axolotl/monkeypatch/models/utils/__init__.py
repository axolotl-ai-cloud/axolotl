"""Utilities shared across grouped MoE monkeypatches."""

from __future__ import annotations

import torch

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def clone_expert_parameter(param: torch.Tensor) -> torch.Tensor:
    """
    Clone a potentially ZeRO-partitioned expert parameter.

    Under ZeRO-3 the original tensor may reside on a meta device with ``numel == 0``
    on non-owner ranks. When this happens we gather the full parameter before
    cloning so downstream ``torch.stack`` calls see concrete storage. If gathering
    fails for any reason we fall back to cloning the local view; later code is
    expected to fill in zeros when needed.
    """
    ds_shape = getattr(param, "ds_shape", None)

    if param.numel() == 0 and ds_shape is not None and isinstance(param, torch.nn.Parameter):
        try:
            from deepspeed import zero  # Lazy import to avoid hard dependency

            with zero.GatheredParameters([param], modifier_rank=None):
                return param.data.detach().clone()
        except Exception:  # pragma: no cover - defensive path
            summary = None
            if hasattr(param, "ds_summary"):
                try:
                    summary = param.ds_summary()
                except Exception:
                    summary = None
            LOG.debug("Failed to gather ZeRO-partitioned parameter %s", summary or "<unnamed>", exc_info=True)
            return param.detach().clone()

    return param.detach().clone()

