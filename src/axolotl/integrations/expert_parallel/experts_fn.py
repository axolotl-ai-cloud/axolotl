# Copyright 2026 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""DeepEP-backed registered functions for `ALL_EXPERTS_FUNCTIONS`.

Three names registered (eager / grouped_mm / scattermoe) sharing one
`_deep_ep_forward` body. Templates ported from `bench_deep_ep.py` Stage 1 modes
3 and 4 — see `BENCHMARK.md` for validating numbers.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .buffer import get_buffer


def _eager_local(experts, recv_x, recv_topk_idx, recv_topk_weights):
    """Eager Python loop over local experts. Reference for numerics."""
    out = torch.zeros_like(recv_x)
    num_local = getattr(experts, "num_local_experts", experts.num_experts)
    for e in range(num_local):
        rows, ks = (recv_topk_idx == e).nonzero(as_tuple=True)
        if rows.numel() == 0:
            continue
        x_e = recv_x[rows]
        gate_up = F.linear(x_e, experts.gate_up_proj[e])
        gate, up = gate_up.chunk(2, dim=-1)
        h = experts.act_fn(gate) * up
        y = F.linear(h, experts.down_proj[e])
        weighted = (y * recv_topk_weights[rows, ks].unsqueeze(-1)).to(out.dtype)
        out.index_add_(0, rows, weighted)
    return out


def _maybe_install_decorator_attrs(experts):
    """`@use_experts_implementation` injects has_gate/has_bias/is_transposed.
    For models loaded outside the decorator path (or in tests), patch them in.
    """
    if not hasattr(experts, "has_gate"):
        experts.has_gate = True
    if not hasattr(experts, "has_bias"):
        experts.has_bias = hasattr(experts, "gate_up_proj_bias")
    if not hasattr(experts, "is_transposed"):
        experts.is_transposed = False


def _grouped_mm_local(experts, recv_x, recv_topk_idx, recv_topk_weights):
    from transformers.integrations.moe import grouped_mm_experts_forward

    _maybe_install_decorator_attrs(experts)
    return grouped_mm_experts_forward(experts, recv_x, recv_topk_idx, recv_topk_weights)


def _scattermoe_local(experts, recv_x, recv_topk_idx, recv_topk_weights):
    from axolotl.integrations.kernels.libs.scattermoe_lora.gemma4_experts import (
        scattermoe_experts_forward,
    )

    return scattermoe_experts_forward(experts, recv_x, recv_topk_idx, recv_topk_weights)


_LOCAL_KERNELS = {
    "eager": _eager_local,
    "grouped_mm": _grouped_mm_local,
    "scattermoe": _scattermoe_local,
}


class _DeepEPDispatch(torch.autograd.Function):
    """Autograd wrapper for `Buffer.dispatch`.

    Forward: routes `x` to ranks owning its top-k experts.
    Backward of dispatch is combine — gradients on `recv_x` are reduced back
    across ranks using the same handle. Only `x`'s grad is returned (idx/weights
    are non-differentiable bookkeeping).
    """

    @staticmethod
    def forward(
        ctx, x, topk_idx, topk_weights, num_per_rank, num_per_expert, is_in_rank
    ):
        buffer = get_buffer()
        recv_x, recv_topk_idx, recv_topk_weights, _, handle, _ = buffer.dispatch(
            x,
            num_tokens_per_rank=num_per_rank,
            is_token_in_rank=is_in_rank,
            num_tokens_per_expert=num_per_expert,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
        )
        ctx.handle = handle
        return recv_x, recv_topk_idx, recv_topk_weights, _DeepEPHandleHolder(handle)

    @staticmethod
    def backward(ctx, grad_recv_x, _grad_idx, grad_recv_w, _grad_handle):
        buffer = get_buffer()
        # Backward-of-dispatch is combine: reduce grad on recv_x (and recv_topk_weights)
        # back to the source rank via the same handle.
        topk_w_grad = (
            grad_recv_w.contiguous()
            if (grad_recv_w is not None and grad_recv_w.numel() > 0)
            else None
        )
        grad_x, grad_topk_w, _ = buffer.combine(
            grad_recv_x.contiguous(),
            ctx.handle,
            topk_weights=topk_w_grad,
        )
        return grad_x, None, grad_topk_w, None, None, None


class _DeepEPCombine(torch.autograd.Function):
    """Autograd wrapper for `Buffer.combine`.

    Forward: sums per-rank partial outputs back into the source token.
    Backward of combine is dispatch — reuses the handle from the dispatch call.
    """

    @staticmethod
    def forward(ctx, x, handle_holder):
        buffer = get_buffer()
        ctx.handle = handle_holder.handle
        combined, _, _ = buffer.combine(x.contiguous(), handle_holder.handle)
        return combined

    @staticmethod
    def backward(ctx, grad_combined):
        buffer = get_buffer()
        # backward-of-combine is dispatch: reuse the cached handle to send
        # gradients to the ranks that produced partial outputs.
        recv_grad, _, _, _, _, _ = buffer.dispatch(
            grad_combined.contiguous(), handle=ctx.handle
        )
        return recv_grad, None


class _DeepEPHandleHolder:
    """Carries the dispatch handle through to combine without breaking autograd
    (autograd.Function output must be Tensors or non-Tensor objects; tuples
    of opaque handles work as a passthrough since combine doesn't differentiate
    against them)."""

    def __init__(self, handle):
        self.handle = handle


def _deep_ep_forward(self, hidden_states, top_k_index, top_k_weights, *, kernel_name):
    """Shared dispatch -> local-experts -> combine pipeline.

    Inputs come in with **global** routing indices (we do not run
    `transformers.RouterParallel`; see DEEP_EP.md §2.4 for why). Sentinels are
    `-1` for slots routed to remote experts; we mask them to a valid local id
    with weight=0 so the local kernel can index safely.
    """
    if hidden_states.dtype != torch.bfloat16:
        original_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.bfloat16)
    else:
        original_dtype = None

    buffer = get_buffer()
    E_global = getattr(self, "num_experts_global", self.num_experts)

    topk_idx_i64 = top_k_index.to(torch.int64)
    topk_w_f32 = top_k_weights.to(torch.float32)

    # Layout is non-differentiable (bookkeeping only).
    with torch.no_grad():
        num_per_rank, _, num_per_expert, is_in_rank, _ = buffer.get_dispatch_layout(
            topk_idx_i64, E_global
        )

    recv_x, recv_topk_idx, recv_topk_weights, handle_holder = _DeepEPDispatch.apply(
        hidden_states,
        topk_idx_i64,
        topk_w_f32,
        num_per_rank,
        num_per_expert,
        is_in_rank,
    )

    # Mask -1 sentinels for kernels that don't filter.
    safe_idx = torch.where(
        recv_topk_idx >= 0, recv_topk_idx, torch.zeros_like(recv_topk_idx)
    )
    valid_mask = (recv_topk_idx >= 0).to(recv_topk_weights.dtype)
    safe_w = recv_topk_weights * valid_mask

    local_out = _LOCAL_KERNELS[kernel_name](self, recv_x, safe_idx, safe_w)

    combined = _DeepEPCombine.apply(local_out, handle_holder)

    if original_dtype is not None:
        combined = combined.to(original_dtype)
    return combined


def deep_ep_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    return _deep_ep_forward(
        self, hidden_states, top_k_index, top_k_weights, kernel_name="eager"
    )


def deep_ep_grouped_mm_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    return _deep_ep_forward(
        self, hidden_states, top_k_index, top_k_weights, kernel_name="grouped_mm"
    )


def deep_ep_scattermoe_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    return _deep_ep_forward(
        self, hidden_states, top_k_index, top_k_weights, kernel_name="scattermoe"
    )


REGISTRY = {
    "deep_ep": deep_ep_experts_forward,
    "deep_ep_grouped_mm": deep_ep_grouped_mm_experts_forward,
    "deep_ep_scattermoe": deep_ep_scattermoe_experts_forward,
}


def register_all() -> None:
    """Register the three names in `ALL_EXPERTS_FUNCTIONS` and whitelist them."""
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS
    from transformers.modeling_utils import PreTrainedModel

    for name, fn in REGISTRY.items():
        ALL_EXPERTS_FUNCTIONS.register(name, fn)

    if not getattr(
        PreTrainedModel.get_correct_experts_implementation, "_deep_ep_patched", False
    ):
        original = PreTrainedModel.get_correct_experts_implementation

        def patched(self, requested_experts):
            if requested_experts in REGISTRY:
                return requested_experts
            return original(self, requested_experts)

        patched._deep_ep_patched = True  # type: ignore[attr-defined]
        PreTrainedModel.get_correct_experts_implementation = patched  # type: ignore[assignment]


def kernel_to_registered_name(kernel: str) -> str:
    """Map `expert_parallel_local_kernel` -> registered name."""
    return {
        "eager": "deep_ep",
        "grouped_mm": "deep_ep_grouped_mm",
        "scattermoe": "deep_ep_scattermoe",
    }[kernel]
