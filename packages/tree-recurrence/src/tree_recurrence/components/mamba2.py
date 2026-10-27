# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""The Mamba2 (SSD) component: Mamba2, Nemotron-H, Bamba, Granite-MoE-Hybrid.

mamba-ssm's kernels reset their state at ``seq_idx`` boundaries but take initial
states per batch row only, so the trie is not forked: every root-to-leaf path is
scanned in one call and each segment keeps its outputs from a path through it. The
fused ``mamba_split_conv1d_scan_combined`` runs that way whole (conv, scan, gated norm
and output projection), on the input projection's rows gathered per path; the
gather's backward sums the gradients of shared prefixes.

Forking would need per-depth batches with ``initial_states`` and padding that leaves
the state untouched (``dt`` masked to zero after its softplus), which no kernel
offers yet.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from tree_recurrence.driver import ScanFn, check_row, padded_scan, restore_tail
from tree_recurrence.plan import TreePlan, segment_ids


def ssd_scan(
    mamba_chunk_scan: Callable[..., Any],
    A: torch.Tensor,
    *,
    varlen: bool = True,
    **kwargs: Any,
) -> ScanFn:
    """A scan over ``mamba_chunk_scan_combined`` (inputs x, dt, B, C).

    ``kwargs`` are the kernel's other arguments (``chunk_size``, ``D``, ``dt_bias``,
    ...). ``varlen``: the kernel resets at ``seq_idx``; otherwise it gets a padded
    batch.
    """
    if varlen:

        def scan(inputs, cu_seqlens, initial_state):
            x, dt, B, C = inputs
            seq_idx = segment_ids(cu_seqlens, x.shape[1])
            return mamba_chunk_scan(x, dt, A, B, C, seq_idx=seq_idx, **kwargs), None

        return scan

    def batched(inputs, initial_state):
        x, dt, B, C = inputs
        return mamba_chunk_scan(x, dt, A, B, C, **kwargs)

    return padded_scan(batched)


def split_conv1d_scan_on_tree(
    mamba_split_conv1d_scan_combined: Callable[..., Any],
    zxbcdt: torch.Tensor,
    plan: TreePlan,
    **kwargs: Any,
) -> torch.Tensor:
    """The fused Mamba2 mixer over ``zxbcdt`` (``[1, T, P]``, row order) on a tree.

    ``kwargs`` are the kernel's other arguments; it must reset at ``seq_idx``. A kernel
    that declines (returns ``None``, as a guard for quantized weights may) declines
    here too, and the mixer runs its unfused path.
    """
    check_row(zxbcdt, plan)
    total, device = zxbcdt.shape[1], zxbcdt.device
    rows = zxbcdt[:, : plan.num_tokens]
    keep = None
    if plan.num_levels == 1:
        cu_seqlens = plan.segment_cu_seqlens(device)
    else:
        gather, cu_seqlens, keep = plan.path_layout(device)
        rows = rows[:, gather]
    out = mamba_split_conv1d_scan_combined(
        rows, seq_idx=segment_ids(cu_seqlens, rows.shape[1]), **kwargs
    )
    if out is None:
        return None
    if keep is not None:
        out = out[:, keep]
    return restore_tail(out, total)
