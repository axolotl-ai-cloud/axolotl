# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0
"""
NVFP4 + FSDP2 multi-GPU smoke.

A frozen NVFP4 base sharded under FSDP2 must (a) run forward/backward with the
``fsdp_pre_all_gather`` snapshot hook and (b) round-trip through a
``FULL_STATE_DICT`` save. The FSDP-hooked NVFP4 tensor subclass is built lazily
(so this package never imports torchao at top level); it must therefore be
exposed at module scope / picklable, or the checkpoint save crashes with
``Can't get local object '_fsdp_nvfp4_class.<locals>.FSDPNVFP4Tensor'``.

Run with::

    torchrun --nproc-per-node=2 -m pytest tests/e2e/multigpu/test_nvfp4_fsdp.py

Skips on a 1-GPU / non-Blackwell executor.
"""

import io
import os

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

_LOCAL_RANK = os.environ.get("LOCAL_RANK")
_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))


def _is_blackwell() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 10


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Need >=2 GPUs for FSDP2"),
    pytest.mark.skipif(
        not _is_blackwell(), reason="NVFP4 requires Blackwell FP4 tensor cores"
    ),
    pytest.mark.skipif(
        _LOCAL_RANK is None or _WORLD_SIZE < 2,
        reason=(
            "Multi-rank test must be launched via "
            "`torchrun --nproc-per-node=2 -m pytest <file>`"
        ),
    ),
]


def test_nvfp4_fsdp2_full_state_dict_save():
    """Frozen NVFP4 base under FSDP2: forward/backward, then a FULL_STATE_DICT
    save (which pickles the frozen NVFP4 params) must both succeed."""
    pytest.importorskip("torchao.prototype.mx_formats.nvfp4_tensor")
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
    )
    from torch.distributed.fsdp import FSDPModule, fully_shard

    from axolotl.integrations.nvfp4.nvfp4_training import convert_to_nvfp4_training

    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    try:
        torch.manual_seed(0)
        dim = 256  # contraction dim must be a multiple of 32 for NVFP4
        block = (
            nn.Sequential(
                nn.Linear(dim, dim, bias=False),
                nn.GELU(),
                nn.Linear(dim, dim, bias=False),
            )
            .cuda()
            .bfloat16()
        )
        for p in block.parameters():
            p.requires_grad_(False)  # frozen base, LoRA-style

        n_swapped = convert_to_nvfp4_training(block)
        assert n_swapped > 0, "no Linear was converted to NVFP4"

        fully_shard(block)
        assert isinstance(block, FSDPModule)

        x = torch.randn(4, dim, device="cuda", dtype=torch.bfloat16)
        out = block(x)  # exercises fsdp_pre_all_gather / post_all_gather
        assert torch.isfinite(out).all()

        # the exact path that crashed before FSDPNVFP4Tensor was made picklable:
        sd = get_model_state_dict(
            block,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
        buf = io.BytesIO()
        torch.save(sd, buf)
        buf.seek(0)
        reloaded = torch.load(buf, weights_only=False)
        assert set(reloaded) == set(sd)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
