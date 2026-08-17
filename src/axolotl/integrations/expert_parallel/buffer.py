# Copyright 2026 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""DeepEP `Buffer` singleton, lazily constructed on first call.

A single Buffer is reused across all MoE layers in a model, since DeepEP's
intranode kernels are sized by `num_nvl_bytes` which we set conservatively at
plugin init. Per-layer Buffer construction would burn memory.
"""

from __future__ import annotations

from typing import Optional

import torch.distributed as dist

_BUFFER = None
_EP_GROUP: Optional[dist.ProcessGroup] = None
_NUM_NVL_BYTES = 256 << 20
_DISPATCH_CONFIG = None
_COMBINE_CONFIG = None
_NUM_RDMA_BYTES = 0


def configure_buffer(
    ep_group: Optional[dist.ProcessGroup],
    num_nvl_bytes: int = 256 << 20,
    num_rdma_bytes: int = 0,
) -> None:
    """Stash params for lazy Buffer construction. Call from `post_model_build`."""
    global _EP_GROUP, _NUM_NVL_BYTES, _NUM_RDMA_BYTES, _BUFFER
    _EP_GROUP = ep_group
    _NUM_NVL_BYTES = num_nvl_bytes
    _NUM_RDMA_BYTES = num_rdma_bytes
    _BUFFER = None  # invalidate any prior buffer


def get_buffer():
    """Return the (lazily constructed) DeepEP Buffer."""
    global _BUFFER
    if _BUFFER is not None:
        return _BUFFER

    import deep_ep

    group = _EP_GROUP if _EP_GROUP is not None else dist.group.WORLD
    _BUFFER = deep_ep.Buffer(
        group=group,
        num_nvl_bytes=_NUM_NVL_BYTES,
        num_rdma_bytes=_NUM_RDMA_BYTES,
        low_latency_mode=False,
    )
    return _BUFFER


def _ep_world() -> int:
    grp = _EP_GROUP if _EP_GROUP is not None else dist.group.WORLD
    return dist.get_world_size(grp)


def get_dispatch_config():
    """Recommended DeepEP dispatch Config for the EP group. DeepEP's own tests ALWAYS pass an
    explicit config to dispatch/combine; the no-config default deadlocks / launch-fails the combine
    on the singleton buffer's reuse across MoE layers (cudaErrorLaunchFailure)."""
    global _DISPATCH_CONFIG
    if _DISPATCH_CONFIG is None:
        import deep_ep

        _DISPATCH_CONFIG = deep_ep.Buffer.get_dispatch_config(_ep_world())
    return _DISPATCH_CONFIG


def get_combine_config():
    """Recommended DeepEP combine Config for the EP group (see get_dispatch_config)."""
    global _COMBINE_CONFIG
    if _COMBINE_CONFIG is None:
        import deep_ep

        _COMBINE_CONFIG = deep_ep.Buffer.get_combine_config(_ep_world())
    return _COMBINE_CONFIG


def barrier_ep() -> None:
    """Barrier on the EP group. Placed before each DeepEP ``combine`` so the fast ranks wait (on
    NCCL, no short timeout) for any rank still AUTOTUNING the local expert kernel — otherwise the
    combine collective hits DeepEP's short internal timeout (``value=0``) and aborts. Negligible cost
    once kernels are cached (all ranks arrive together). Disable with AXOLOTL_EP_NO_BARRIER=1."""
    import os

    if os.environ.get("AXOLOTL_EP_NO_BARRIER"):
        return
    if not dist.is_initialized():
        return
    import torch

    torch.cuda.synchronize()
    dist.barrier(_EP_GROUP if _EP_GROUP is not None else dist.group.WORLD)


def reset_buffer() -> None:
    """Drop the cached Buffer + configs. Used in tests."""
    global _BUFFER, _DISPATCH_CONFIG, _COMBINE_CONFIG
    _BUFFER = None
    _DISPATCH_CONFIG = None
    _COMBINE_CONFIG = None
