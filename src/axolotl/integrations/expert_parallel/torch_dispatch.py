# Copyright 2026 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Token-dispatch expert parallelism over plain ``all_to_all_single`` (NCCL or gloo).

The layout matches DeepEP's contract so every local experts kernel runs unchanged: each
token is sent ONCE to every rank that owns at least one of its top-k experts, carrying a
K-wide ``recv_topk_idx`` (local expert ids in ``[0, E_local)`` for that rank, ``-1`` for
slots owned elsewhere) and K-wide fp32 ``recv_topk_weights`` (``0`` on ``-1`` slots). The
local kernel weights and sums over K, and ``combine`` sums the per-rank partial outputs
back onto the source tokens.

The collectives are registered as ``torch.ops.axolotl.ep_all_to_all_single`` and
``torch.ops.axolotl.ep_all_to_all_single_equal`` so selective activation checkpointing
can match (and save) them by name.

Hang safety: the split sizes are derived from the routing, so every rank must see the
same routing for the same step. Recomputation under activation checkpointing must reuse
the forward's ``topk`` result and CPU split copies rather than re-running them.
"""

from dataclasses import dataclass
from typing import Callable

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol

from axolotl.kernels.op_registry import _UnregisteredOp, register_kernel_op
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

_EP_GROUP: dist.ProcessGroup | None = None


def set_ep_group(group: dist.ProcessGroup | None) -> None:
    global _EP_GROUP
    _EP_GROUP = group


def get_ep_group() -> dist.ProcessGroup | None:
    return _EP_GROUP


def _resolve_group(group_name: str) -> dist.ProcessGroup:
    return dist.distributed_c10d._resolve_process_group(group_name)


def _a2a_impl(
    x: torch.Tensor,
    output_splits: torch.Tensor,
    input_splits: torch.Tensor,
    group_name: str,
) -> torch.Tensor:
    out_list = output_splits.tolist()
    in_list = input_splits.tolist()
    out = x.new_empty((sum(out_list), *x.shape[1:]))
    dist.all_to_all_single(
        out, x.contiguous(), out_list, in_list, group=_resolve_group(group_name)
    )
    return out


def _a2a_equal_impl(x: torch.Tensor, group_name: str) -> torch.Tensor:
    out = x.new_empty(x.shape)
    dist.all_to_all_single(out, x.contiguous(), group=_resolve_group(group_name))
    return out


_a2a_op = register_kernel_op("ep_all_to_all_single")(_a2a_impl)
_a2a_equal_op = register_kernel_op("ep_all_to_all_single_equal")(_a2a_equal_impl)

OPS_REGISTERED = not (
    isinstance(_a2a_op, _UnregisteredOp) or isinstance(_a2a_equal_op, _UnregisteredOp)
)


if OPS_REGISTERED:

    @_a2a_op.register_fake
    def _a2a_fake(x, output_splits, input_splits, group_name):
        return x.new_empty((torch.library.get_ctx().new_dynamic_size(), *x.shape[1:]))

    @_a2a_equal_op.register_fake
    def _a2a_equal_fake(x, group_name):
        return x.new_empty(x.shape)

    def _a2a_setup_context(ctx, inputs, output):
        _, output_splits, input_splits, group_name = inputs
        ctx.save_for_backward(output_splits, input_splits)
        ctx.group_name = group_name

    def _a2a_backward(ctx, grad):
        output_splits, input_splits = ctx.saved_tensors
        return (
            _a2a_op(grad, input_splits, output_splits, ctx.group_name),
            None,
            None,
            None,
        )

    def _a2a_equal_setup_context(ctx, inputs, output):
        ctx.group_name = inputs[1]

    def _a2a_equal_backward(ctx, grad):
        return _a2a_equal_op(grad, ctx.group_name), None

    _a2a_op.register_autograd(_a2a_backward, setup_context=_a2a_setup_context)
    _a2a_equal_op.register_autograd(
        _a2a_equal_backward, setup_context=_a2a_equal_setup_context
    )
else:
    LOG.warning(
        "expert_parallel torch backend: all-to-all custom ops are unregistered; running "
        "them eagerly. Selective checkpointing cannot see (or save) the EP collectives."
    )


class _A2AFallback(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, output_splits, input_splits, group_name):
        ctx.save_for_backward(output_splits, input_splits)
        ctx.group_name = group_name
        return _a2a_impl(x, output_splits, input_splits, group_name)

    @staticmethod
    def backward(ctx, grad):
        output_splits, input_splits = ctx.saved_tensors
        return (
            _A2AFallback.apply(grad, input_splits, output_splits, ctx.group_name),
            None,
            None,
            None,
        )


class _A2AEqualFallback(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group_name):
        ctx.group_name = group_name
        return _a2a_equal_impl(x, group_name)

    @staticmethod
    def backward(ctx, grad):
        return _A2AEqualFallback.apply(grad, ctx.group_name), None


def all_to_all_single(
    x: torch.Tensor,
    output_splits: torch.Tensor,
    input_splits: torch.Tensor,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Differentiable uneven all-to-all along dim 0. ``output_splits`` / ``input_splits`` are
    CPU int64 tensors of length ``group.size()``; backward is the same op with them swapped."""
    if OPS_REGISTERED:
        return _a2a_op(x, output_splits, input_splits, group.group_name)
    return _A2AFallback.apply(x, output_splits, input_splits, group.group_name)


def all_to_all_single_equal(x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Differentiable equal-split all-to-all along dim 0."""
    if OPS_REGISTERED:
        return _a2a_equal_op(x, group.group_name)
    return _A2AEqualFallback.apply(x, group.group_name)


def all_to_all_single_async(
    x: torch.Tensor,
    output_splits: list[int],
    input_splits: list[int],
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Differentiable uneven all-to-all whose wait is deferred to the output's first use.

    Returns an ``AsyncCollectiveTensor`` so compute enqueued before that use overlaps the
    collective. Dispatches as ``_c10d_functional::all_to_all_single`` / ``wait_tensor``.
    """
    return funcol.all_to_all_single_autograd(
        x.contiguous(), output_splits, input_splits, group
    )


def compute_send_counts(
    topk_idx: torch.Tensor, num_ranks: int, num_local_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(dest [T,K], is_in_rank [T,P], send_counts [P])`` for global ``topk_idx``
    (``-1`` = unrouted). ``dest`` is the owning rank per slot, ``-1`` where unrouted."""
    dest = torch.where(
        topk_idx >= 0,
        torch.div(topk_idx, num_local_experts, rounding_mode="floor"),
        torch.full_like(topk_idx, -1),
    )
    ranks = torch.arange(num_ranks, device=topk_idx.device, dtype=dest.dtype)
    is_in_rank = (dest.unsqueeze(-1) == ranks).any(dim=1)
    return dest, is_in_rank, is_in_rank.sum(dim=0)


def build_send_layout(
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    dest: torch.Tensor,
    is_in_rank: torch.Tensor,
    num_send: int,
    num_local_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rank-major, token-ascending send order with per-destination local expert ids.

    Returns ``(send_token_idx [N], send_rank [N], send_idx [N,K], send_w [N,K])``. ``num_send``
    must equal ``is_in_rank.sum()``; passing it from the host avoids a device sync.
    """
    T = is_in_rank.shape[0]
    flat = is_in_rank.t().reshape(-1)
    # a stable sort of the (rank, token)-ordered membership bits puts the selected pairs first
    order = torch.argsort((~flat).to(torch.int8), stable=True)[:num_send]
    send_rank = torch.div(order, T, rounding_mode="floor")
    send_token_idx = order - send_rank * T
    rank_col = send_rank.unsqueeze(-1).to(topk_idx.dtype)
    send_idx = torch.where(
        dest[send_token_idx] == rank_col,
        topk_idx[send_token_idx] - rank_col * num_local_experts,
        torch.full_like(rank_col, -1),
    )
    w = topk_weights[send_token_idx]
    send_w = torch.where(send_idx >= 0, w, torch.zeros_like(w))
    return send_token_idx, send_rank, send_idx, send_w


@dataclass
class TorchEPHandle:
    """State ``combine`` needs to return expert outputs to their source tokens."""

    num_tokens: int
    send_token_idx: torch.Tensor | None = None
    send_splits: torch.Tensor | None = None
    recv_splits: torch.Tensor | None = None
    group: dist.ProcessGroup | None = None
    recv_x: torch.Tensor | None = None
    recv_w: torch.Tensor | None = None


def dispatch(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    num_local_experts: int,
    group: dist.ProcessGroup | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, TorchEPHandle]:
    """Send each token once to every rank owning one of its experts.

    ``x`` is ``[T,H]``, ``topk_idx`` ``[T,K]`` int64 global expert ids (``-1`` = unrouted),
    ``topk_weights`` ``[T,K]`` fp32. Returns ``(recv_x, recv_idx, recv_w, handle)`` in the
    DeepEP layout. Tokens with no routed expert are not sent; their combined output is zero.
    ``group=None`` or a size-1 group is the collective-free local path.
    """
    num_ranks = 1 if group is None else dist.get_world_size(group)
    T = x.shape[0]
    if num_ranks == 1:
        w = torch.where(topk_idx >= 0, topk_weights, torch.zeros_like(topk_weights))
        return x, topk_idx, w, TorchEPHandle(num_tokens=T)

    with torch.no_grad():
        dest, is_in_rank, send_counts = compute_send_counts(
            topk_idx, num_ranks, num_local_experts
        )
        recv_counts = all_to_all_single_equal(send_counts, group)
        splits = torch.stack((send_counts, recv_counts)).cpu()
    send_splits, recv_splits = splits[0], splits[1]

    send_token_idx, _send_rank, send_idx, send_w = build_send_layout(
        topk_idx,
        topk_weights,
        dest,
        is_in_rank,
        int(send_splits.sum()),
        num_local_experts,
    )
    recv_x = all_to_all_single(x[send_token_idx], recv_splits, send_splits, group)
    recv_w = all_to_all_single(send_w, recv_splits, send_splits, group)
    with torch.no_grad():
        recv_idx = all_to_all_single(send_idx, recv_splits, send_splits, group)
    handle = TorchEPHandle(
        num_tokens=T,
        send_token_idx=send_token_idx,
        send_splits=send_splits,
        recv_splits=recv_splits,
        group=group,
        recv_x=recv_x,
        recv_w=recv_w,
    )
    return recv_x, recv_idx, recv_w, handle


def _anchor_backward(local_out: torch.Tensor, handle: TorchEPHandle) -> torch.Tensor:
    if torch.is_grad_enabled() and (
        local_out.shape[0] == 0 or not local_out.requires_grad
    ):
        # A rank that received no rows must still join every backward all-to-all of this
        # layer (combine, recv_x, recv_w), else the peers' backward collectives hang.
        anchors = [t.sum() for t in (handle.recv_x, handle.recv_w) if t.requires_grad]
        if anchors:
            local_out = local_out + (sum(anchors) * 0).to(local_out.dtype)
        elif not local_out.requires_grad:
            # grad-ness can differ per rank (trainable experts, frozen inputs, zero
            # received rows); the combine backward must be issued on all or none
            local_out = local_out.detach().requires_grad_()
    return local_out


def combine(
    local_out: torch.Tensor, handle: TorchEPHandle, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Return ``[N_recv,H]`` expert outputs to their source ranks and sum them per token."""
    if handle.send_token_idx is None:
        return local_out if dtype is None else local_out.to(dtype)
    local_out = _anchor_backward(local_out, handle)
    out_send = all_to_all_single(
        local_out, handle.send_splits, handle.recv_splits, handle.group
    )
    if dtype is not None:
        out_send = out_send.to(dtype)
    y = out_send.new_zeros((handle.num_tokens, out_send.shape[-1]))
    return y.index_add_(0, handle.send_token_idx, out_send)


def chunk_bounds(num_tokens: int, chunks: int) -> list[tuple[int, int]]:
    """``chunks`` contiguous ``[start, end)`` token ranges; the last takes the remainder,
    so some are empty when ``num_tokens < chunks``."""
    size = num_tokens // chunks
    return [
        (i * size, num_tokens if i == chunks - 1 else (i + 1) * size)
        for i in range(chunks)
    ]


def dispatch_chunked_forward(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    local_kernel: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    num_local_experts: int,
    group: dist.ProcessGroup,
    chunks: int,
) -> torch.Tensor:
    """``dispatch -> local_kernel -> combine`` over ``chunks`` token ranges, pipelined.

    Chunk ``i+1``'s dispatch all-to-alls are issued before chunk ``i``'s local kernel, and
    chunk ``i``'s combine is consumed only after chunk ``i+1``'s kernel is enqueued, so the
    collectives overlap expert compute in forward. The split counts of every chunk come
    from ONE count exchange and ONE device->host copy. Every rank issues every chunk's
    collectives in the same order, including empty ones.
    """
    num_ranks = dist.get_world_size(group)
    bounds = chunk_bounds(x.shape[0], chunks)

    with torch.no_grad():
        dest, is_in_rank, _ = compute_send_counts(
            topk_idx, num_ranks, num_local_experts
        )
        send_counts = torch.stack([is_in_rank[s:e].sum(dim=0) for s, e in bounds])
        # the equal all-to-all splits dim 0 by destination rank, so exchange [P, C]
        recv_counts = all_to_all_single_equal(send_counts.t().contiguous(), group).t()
        splits = torch.stack((send_counts, recv_counts)).cpu()

    def issue_dispatch(i):
        s, e = bounds[i]
        send_splits, recv_splits = splits[0, i], splits[1, i]
        send_list, recv_list = send_splits.tolist(), recv_splits.tolist()
        send_token_idx, _send_rank, send_idx, send_w = build_send_layout(
            topk_idx[s:e],
            topk_weights[s:e],
            dest[s:e],
            is_in_rank[s:e],
            sum(send_list),
            num_local_experts,
        )
        recv_x = all_to_all_single_async(
            x[s:e][send_token_idx], recv_list, send_list, group
        )
        recv_w = all_to_all_single_async(send_w, recv_list, send_list, group)
        with torch.no_grad():
            recv_idx = all_to_all_single_async(send_idx, recv_list, send_list, group)
        handle = TorchEPHandle(
            num_tokens=e - s,
            send_token_idx=send_token_idx,
            send_splits=send_splits,
            recv_splits=recv_splits,
            group=group,
            recv_x=recv_x,
            recv_w=recv_w,
        )
        return recv_idx, handle

    def issue_combine(local_out, handle):
        local_out = _anchor_backward(local_out, handle)
        return all_to_all_single_async(
            local_out,
            handle.send_splits.tolist(),
            handle.recv_splits.tolist(),
            group,
        )

    def finalize(out_send, handle):
        y = out_send.new_zeros((handle.num_tokens, out_send.shape[-1]))
        return y.index_add_(0, handle.send_token_idx, out_send)

    parts: list[torch.Tensor | None] = [None] * chunks
    in_flight: list[tuple[torch.Tensor, TorchEPHandle] | None] = [None] * chunks
    pending = issue_dispatch(0)
    for i in range(chunks):
        recv_idx, handle = pending
        if i + 1 < chunks:
            pending = issue_dispatch(i + 1)
        # wait before the kernel: an unwaited collective output has data_ptr() 0, so a
        # Triton/CuTe kernel that is its first reader bypasses the deferred wait
        handle.recv_x = funcol.wait_tensor(handle.recv_x)
        handle.recv_w = funcol.wait_tensor(handle.recv_w)
        recv_idx = funcol.wait_tensor(recv_idx)
        local_out = local_kernel(handle.recv_x, recv_idx, handle.recv_w)
        in_flight[i] = (issue_combine(local_out, handle), handle)
        if i >= 1:
            parts[i - 1] = finalize(*in_flight[i - 1])
            in_flight[i - 1] = None
    parts[-1] = finalize(*in_flight[-1])
    return torch.cat(parts, dim=0)
