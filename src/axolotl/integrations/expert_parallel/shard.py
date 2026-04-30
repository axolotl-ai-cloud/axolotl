# Copyright 2026 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Generic expert-weight sharding for `@use_experts_implementation` modules.

After this runs (in `post_model_build`, before FSDP wraps), each rank's Experts
modules hold only their local slice of the experts dim. The registered
`deep_ep_*` forward function then handles dispatch -> local compute -> combine.
"""

from __future__ import annotations

import gc

import torch
import torch.distributed as dist

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def _replace_with_slice(module, attr_name: str, start: int, end: int) -> None:
    """Replace `module.{attr_name}` with a fresh-allocation slice [start:end].

    `tensor[start:end]` on a contiguous storage returns a VIEW that shares the
    same underlying allocation. `.contiguous()` is a no-op on an already-
    contiguous view, so wrapping the view in a new Parameter does NOT release
    the original full-size storage — refcount on the storage stays >= 1 from
    the new view.

    `.clone()` forces a fresh allocation. After we drop the old Parameter,
    the original storage's refcount drops to zero and PyTorch's caching
    allocator reclaims the memory on the next `empty_cache()`.
    """
    old = getattr(module, attr_name)
    new_data = old.data[start:end].detach().clone().contiguous()
    requires_grad = old.requires_grad
    # Drop the old Parameter from the module's _parameters registry first so
    # nothing is holding a reference when we assign the new one.
    if attr_name in module._parameters:
        del module._parameters[attr_name]
    setattr(
        module,
        attr_name,
        torch.nn.Parameter(new_data, requires_grad=requires_grad),
    )
    # `old` goes out of scope on return.


def _detect_experts_modules(model):
    """Yield (name, module) pairs for every module that looks like an Experts class.

    Detection: 3D `gate_up_proj` and `down_proj` parameters with experts on dim 0.
    This is the canonical layout enforced by `@use_experts_implementation`.
    Mixtral's `ModuleList[MixtralBlockSparseTop2MLP]` does NOT match — out of scope
    for v1.
    """
    for name, module in model.named_modules():
        gp = getattr(module, "gate_up_proj", None)
        dp = getattr(module, "down_proj", None)
        if gp is None or dp is None:
            continue
        if not (
            isinstance(gp, torch.nn.Parameter) and isinstance(dp, torch.nn.Parameter)
        ):
            continue
        if gp.dim() != 3 or dp.dim() != 3:
            continue
        yield name, module


def shard_expert_weights(model, ep_group) -> int:
    """Slice expert weights along dim 0 per the EP rank.

    Args:
        model: A built (but not yet FSDP-wrapped) HuggingFace model.
        ep_group: `torch.distributed.ProcessGroup` for EP, or `None` for
            single-rank (no-op).

    Returns:
        Number of Experts modules sharded (0 if EP disabled or none found).

    Raises:
        ValueError: if any Experts module's `num_experts` is not divisible by
            the EP world size.

    DDP composition: the sharded params hold DIFFERENT content per rank, so we
    add their fully-qualified names to `model._ddp_params_and_buffers_to_ignore`
    to prevent the startup broadcast from copying rank 0's slice everywhere.
    FSDP composition is handled in `ExpertParallelPlugin.fully_shard_experts`.
    """
    if ep_group is None:
        return 0

    ep_size = dist.get_world_size(ep_group)
    if ep_size <= 1:
        return 0

    ep_rank = dist.get_rank(ep_group)
    sharded = 0
    ignore_names: list[str] = []

    for name, module in _detect_experts_modules(model):
        gp = module.gate_up_proj
        E = gp.shape[0]
        if E % ep_size != 0:
            raise ValueError(
                f"Expert module {name!r}: num_experts={E} not divisible by "
                f"ep_size={ep_size}. Adjust the model config or ep_size."
            )
        E_local = E // ep_size
        start = ep_rank * E_local
        end = start + E_local

        with torch.no_grad():
            _replace_with_slice(module, "gate_up_proj", start, end)
            _replace_with_slice(module, "down_proj", start, end)
            for bias_name in ("gate_up_proj_bias", "down_proj_bias"):
                bias = getattr(module, bias_name, None)
                if (
                    isinstance(bias, torch.nn.Parameter)
                    and bias.dim() >= 1
                    and bias.shape[0] == E
                ):
                    _replace_with_slice(module, bias_name, start, end)

        # Stash metadata the registered fn needs.
        module.local_expert_offset = start
        module.num_local_experts = E_local
        module.num_experts_global = E
        # Override the kernel's view of num_experts for grouped_mm/scattermoe bucketing.
        module.num_experts = E_local

        # Mark sharded params as DDP-ignored — they hold rank-specific content
        # and must NOT be broadcast from rank 0 at DDP construction.
        ignore_names.append(f"{name}.gate_up_proj")
        ignore_names.append(f"{name}.down_proj")
        for bias_name in ("gate_up_proj_bias", "down_proj_bias"):
            if isinstance(getattr(module, bias_name, None), torch.nn.Parameter):
                ignore_names.append(f"{name}.{bias_name}")

        sharded += 1

    if sharded:
        # Drop refs to any leftover full-size storage and return cached memory
        # to the allocator. Without this, the original `from_pretrained`
        # allocations stay reserved and `memory_allocated()` doesn't drop.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Append (don't overwrite) — other systems may have set this too.
        existing = list(getattr(model, "_ddp_params_and_buffers_to_ignore", []))
        model._ddp_params_and_buffers_to_ignore = existing + ignore_names
        LOG.info(
            f"Sharded {sharded} Experts module(s) along the experts dim "
            f"(ep_rank={ep_rank}, ep_size={ep_size}). "
            f"Marked {len(ignore_names)} param(s) as DDP-ignored."
        )
    return sharded
