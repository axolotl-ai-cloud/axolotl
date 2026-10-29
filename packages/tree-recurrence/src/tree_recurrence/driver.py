# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Run a short causal conv or a recurrence over a tree-packed row.

``tree_conv`` runs every segment with its path history prepended, in one call.
``tree_scan`` either forks, running the trie level by level and starting each
segment from its parent's final state (the handoff stays in autograd), or runs every
root-to-leaf path (each sample once) in one call and keeps each segment's outputs
from a path through it. A kernel that cannot
take per-sequence initial states with varlen boundaries always takes paths; the
others fork when the tokens it saves outweigh one more kernel call per level
(``Policy``).

Inputs are ``[1, T, ...]`` in row order. ``T`` may exceed the plan's tokens when a
collator padded the row; the padding's outputs are zeros.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from tree_recurrence.plan import TreePlan

# A varlen scan over the sequences of one call: (inputs [1, T, ...] each,
# cu_seqlens int32 [N + 1], initial_state [N, ...] or None) -> (output
# [1, T, ...], final_state [N, ...] or None).
ScanFn = Callable[
    [Sequence[torch.Tensor], torch.Tensor, torch.Tensor | None],
    tuple[torch.Tensor, torch.Tensor | None],
]
# A short causal conv: (x [1, T, D], cu_seqlens int32 [N + 1]) -> [1, T, D].
ConvFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
# A scan over a batch of right-padded sequences: (inputs [N, L, ...] each,
# initial_state [N, ...] or None) -> output [N, L, ...].
BatchedScanFn = Callable[[Sequence[torch.Tensor], torch.Tensor | None], torch.Tensor]


@dataclass(frozen=True)
class Policy:
    """When ``tree_scan`` forks rather than rescanning shared paths.

    Forking runs one kernel call per trie level instead of one, and saves the
    rescanned prefix tokens. One more varlen call (forward and backward) of FLA's
    chunked kernels costs ~10 ms of host time per layer, rescanning a shared token
    ~0.2 us of GPU time per layer; on a launch-bound host a 24k-token saving still
    lost to rescanning.
    """

    # prefix tokens a fork must save per extra trie level
    fork_min_saved_tokens: int = 65536

    def forks(self, plan: TreePlan) -> bool:
        levels = plan.num_levels
        saved = plan.path_tokens - plan.num_tokens
        return levels > 1 and saved >= self.fork_min_saved_tokens * (levels - 1)


DEFAULT_POLICY = Policy()


def restore_tail(out: torch.Tensor, total: int) -> torch.Tensor:
    """Zeros for the collation padding past a plan's tokens."""
    missing = total - out.shape[1]
    if not missing:
        return out
    return torch.cat([out, out.new_zeros(out.shape[0], missing, *out.shape[2:])], 1)


def check_row(x: torch.Tensor, plan: TreePlan) -> None:
    if x.shape[0] != 1 or x.shape[1] < plan.num_tokens:
        raise ValueError(
            f"a tree-packed row is [1, >= {plan.num_tokens}, ...], got {list(x.shape)}"
        )


def tree_conv(
    conv: ConvFn, x: torch.Tensor, plan: TreePlan, width: int
) -> torch.Tensor:
    """A short causal conv of ``width`` over ``x`` (``[1, T, D]``) on a tree."""
    check_row(x, plan)
    total, tokens = x.shape[1], plan.num_tokens
    gather, cu_seqlens, keep = plan.conv_layout(width, x.device)
    padded = torch.nn.functional.pad(x[:, :tokens], (0, 0, 0, 1))
    return restore_tail(conv(padded[:, gather], cu_seqlens)[:, keep], total)


def tree_scan(
    scan: ScanFn,
    inputs: Sequence[torch.Tensor],
    plan: TreePlan,
    *,
    forkable: bool = True,
    policy: Policy | None = None,
) -> torch.Tensor:
    """Run a recurrence over ``inputs`` (``[1, T, ...]`` each) on a tree.

    ``forkable``: ``scan`` takes per-sequence initial states and returns final ones.
    """
    policy = policy or DEFAULT_POLICY
    for x in inputs:
        check_row(x, plan)
    device, total, tokens = inputs[0].device, inputs[0].shape[1], plan.num_tokens
    inputs = [x[:, :tokens] for x in inputs]
    if plan.num_levels == 1:  # nothing shared: the row is already in level order
        out = scan(inputs, plan.segment_cu_seqlens(device), None)[0]
    elif not (forkable and policy.forks(plan)):
        gather, cu_seqlens, keep = plan.path_layout(device)
        out = scan([x[:, gather] for x in inputs], cu_seqlens, None)[0][:, keep]
    else:
        # One gather into level order; each level is then a contiguous slice.
        order, inverse, levels = plan.level_layout(device)
        ordered = [x[:, order] for x in inputs]
        outs, final = [], None
        for level in levels:
            initial = None
            if level.parents is not None:
                assert final is not None, "a forkable scan returns final states"
                initial = final[level.parents]
            out, final = scan(
                [x[:, level.start : level.end] for x in ordered],
                level.cu_seqlens,
                initial,
            )
            outs.append(out)
        out = torch.cat(outs, dim=1)[:, inverse]
    return restore_tail(out, total)


def padded_scan(batched: BatchedScanFn) -> ScanFn:
    """A ``ScanFn`` from a scan without varlen support, over right-padded sequences.

    Each sequence becomes a row of one batch, so the scan needs no boundaries; a
    causal scan's outputs do not depend on the padding after them. Final states would
    include the padding, so the result cannot fork (``forkable=False``).
    """

    def scan(inputs, cu_seqlens, initial_state):
        bounds = cu_seqlens.tolist()
        lengths = torch.as_tensor(
            [end - start for start, end in zip(bounds[:-1], bounds[1:], strict=True)],
            device=cu_seqlens.device,
        )
        width = int(lengths.max())
        steps = torch.arange(width, device=cu_seqlens.device)
        valid = steps[None] < lengths[:, None]
        index = torch.where(valid, cu_seqlens[:-1, None].long() + steps, bounds[-1])
        batch = [
            torch.nn.functional.pad(x[0], (0, 0) * (x.dim() - 2) + (0, 1))[index]
            for x in inputs
        ]
        return batched(batch, initial_state)[valid][None], None

    return scan
