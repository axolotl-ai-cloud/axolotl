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


def reset_buffer() -> None:
    """Drop the cached Buffer. Used in tests."""
    global _BUFFER
    _BUFFER = None
