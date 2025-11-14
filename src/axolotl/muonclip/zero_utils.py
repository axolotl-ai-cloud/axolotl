"""Gather helpers for DeepSpeed ZeRO and FSDP."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch

try:  # pragma: no cover - optional dependency
    from deepspeed.runtime.zero.partition_parameters import GatheredParameters
except ImportError:  # pragma: no cover
    GatheredParameters = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
except ImportError:  # pragma: no cover
    FSDP = None  # type: ignore


@contextmanager
def gather_full_param(param: torch.nn.Parameter) -> Iterator[torch.nn.Parameter]:
    """
    Context manager that temporarily gathers a full parameter tensor for ZeRO/FSDP.
    """

    module = param._replicated_tensor_module if hasattr(param, "_replicated_tensor_module") else None

    if FSDP is not None and module is not None and isinstance(module, FSDP):
        with module.summon_full_params([param], writeback=True):
            yield param
        return

    if GatheredParameters is not None and getattr(param, "ds_status", None) is not None:
        try:
            with GatheredParameters([param], modifier_rank=None, enabled=True):
                yield param
            return
        except Exception:  # pragma: no cover - fallback to default
            pass

    yield param
