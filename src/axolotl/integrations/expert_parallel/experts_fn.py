# Copyright 2026 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""The ``expert_parallel`` experts implementation for `ALL_EXPERTS_FUNCTIONS`.

Dispatches tokens to the ranks owning their experts (DeepEP or torch all-to-all), runs
whichever experts implementation the user configured on the received rows, and combines
the results back. Slots owned by another rank reach the local implementation as expert id
``num_experts`` with weight 0.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

from . import torch_dispatch
from .buffer import barrier_ep, get_buffer, get_combine_config, get_dispatch_config

# Per-forward [B*S] bool mask of real (non-padding) tokens, set by the EP plugin's model
# pre-hook from the batch `attention_mask` and consumed by `_ep_forward` to exclude
# padding from the dispatch. ``None`` when there is no padding (packed / no attention_mask).
_VALID_TOKEN_MASK: torch.Tensor | None = None

# Max tokens routed to any single expert per forward (``expert_parallel_token_capacity``), set by the
# EP plugin. ``None`` disables the cap. Overloaded experts deadlock DeepEP's intranode combine, so
# GLM-style routers (concentration grows with depth) need this.
_TOKEN_CAPACITY: int | None = None


def set_token_capacity(cap: int | None) -> None:
    global _TOKEN_CAPACITY
    _TOKEN_CAPACITY = cap


# ``expert_parallel_dispatch_chunks``: token chunks the torch backend pipelines per MoE forward.
_DISPATCH_CHUNKS: int = 1


def set_dispatch_chunks(chunks: int) -> None:
    global _DISPATCH_CHUNKS
    _DISPATCH_CHUNKS = chunks


def _apply_expert_capacity(topk_idx, topk_w, cap):
    """Cap tokens-per-expert to ``cap`` by sentinelling (-1) the lowest-weight excess (token,expert)
    assignments, then rescale each token's surviving weights back to its pre-drop gate sum. DeepEP's
    intranode combine hangs once one expert receives too many tokens (the GLM router concentrates more
    with depth, crossing the hang threshold ~layer 31); standard MoE capacity-dropping keeps every
    expert under the limit. The GLM router normalizes the top-k weights to sum to 1, so dropping an
    expert without rescaling would silently attenuate that token's combined expert output. Already-(-1)
    slots are left untouched. Returns ``(capped_idx, rescaled_w)``."""
    import torch

    ntok, K = topk_idx.shape
    flat_e = topk_idx.reshape(-1)
    flat_w = topk_w.reshape(-1)
    # sort assignments by (expert asc, weight desc) via two stable passes
    o1 = torch.argsort(flat_w, descending=True, stable=True)
    o2 = torch.argsort(flat_e[o1], stable=True)
    order = o1[o2]
    se = flat_e[order]
    pos = torch.arange(se.numel(), device=se.device)
    is_new = torch.ones_like(se, dtype=torch.bool)
    is_new[1:] = se[1:] != se[:-1]
    grp_start = torch.cummax(torch.where(is_new, pos, torch.zeros_like(pos)), 0).values
    within = pos - grp_start
    drop_sorted = (within >= cap) & (se >= 0)
    drop = torch.zeros_like(flat_e, dtype=torch.bool)
    drop[order] = drop_sorted
    capped_idx = flat_e.masked_fill(drop, -1).reshape(ntok, K)

    orig_sum = (topk_w * (topk_idx >= 0)).sum(dim=-1, keepdim=True)
    kept = (capped_idx >= 0).to(topk_w.dtype)
    kept_sum = (topk_w * kept).sum(dim=-1, keepdim=True)
    # double-where: guard the divisor so a fully-dropped token (kept_sum==0) can't backprop 0*inf=NaN
    safe_kept_sum = torch.where(kept_sum > 0, kept_sum, torch.ones_like(kept_sum))
    rescale = torch.where(
        kept_sum > 0, orig_sum / safe_kept_sum, torch.ones_like(kept_sum)
    )
    rescaled_w = topk_w * kept * rescale
    return capped_idx, rescaled_w


def set_valid_token_mask(mask: torch.Tensor | None) -> None:
    global _VALID_TOKEN_MASK
    _VALID_TOKEN_MASK = mask


def _get_valid_token_mask() -> torch.Tensor | None:
    return _VALID_TOKEN_MASK


EXPERT_PARALLEL = "expert_parallel"

# Dispatch backend (``deep_ep`` | ``torch``) and the wrapped local experts implementation,
# both set by the EP plugin before the model is built.
_BACKEND: str | None = None
_LOCAL_IMPLEMENTATION: str = "grouped_mm"


def set_backend(backend: str | None) -> None:
    global _BACKEND
    if backend is not None and backend not in _BACKENDS:
        raise ValueError(f"unknown expert_parallel backend {backend!r}")
    _BACKEND = backend


def get_backend() -> str | None:
    return _BACKEND


def set_local_implementation(name: str) -> None:
    global _LOCAL_IMPLEMENTATION
    if name == EXPERT_PARALLEL:
        raise ValueError(
            "expert_parallel cannot wrap itself as the local implementation"
        )
    _LOCAL_IMPLEMENTATION = name


def get_local_implementation() -> str:
    return _LOCAL_IMPLEMENTATION


def _eager_local(experts, recv_x, recv_topk_idx, recv_topk_weights):
    """Eager Python loop over local experts. Reference for numerics; ids outside
    ``[0, num_local_experts)`` are sentinels and skipped."""
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


def resolve_local_implementation(name: str | None = None):
    """The experts forward registered under ``name`` (default: the configured one)."""
    name = name or _LOCAL_IMPLEMENTATION
    if name == "eager":
        # transformers' "eager" is the class's own forward, unreachable from inside the interface
        return _eager_local
    if name == EXPERT_PARALLEL:
        raise ValueError(
            "expert_parallel cannot wrap itself as the local implementation"
        )
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS

    return ALL_EXPERTS_FUNCTIONS.get_interface(name, _eager_local)


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


def _normalize_sentinels(recv_topk_idx, recv_topk_weights, num_local_experts):
    """Map the dispatch's ``-1`` remote slots to ``num_local_experts`` with weight 0.

    transformers' grouped_mm / sonicmoe / deepgemm drop ids ``>= num_experts`` natively;
    batched_mm clamps them and relies on the zero weight. Both need the module's
    ``_is_expert_parallel`` set first, which ``_ep_forward`` does.
    """
    remote = recv_topk_idx < 0
    return (
        recv_topk_idx.masked_fill(remote, num_local_experts),
        recv_topk_weights.masked_fill(remote, 0),
    )


def _pad_expert_rows(recv_x, idx, weights, num_local_experts):
    """Append one zero row routed to each local expert (unit weight, sentinels elsewhere).

    Every expert then sees at least one row on every rank, so every expert weight gets a
    grad (exactly zero for an otherwise idle expert) and FSDP's reduce-scatter payload is
    the same on every dp_shard peer; the pad rows are sliced off before combine.
    """
    e = num_local_experts
    k = idx.shape[1]
    pad_idx = idx.new_full((e, k), e)
    pad_idx[:, 0] = torch.arange(e, device=idx.device)
    pad_w = weights.new_zeros((e, k))
    pad_w[:, 0] = 1
    return (
        torch.cat((recv_x, recv_x.new_zeros((e, recv_x.shape[1])))),
        torch.cat((idx, pad_idx)),
        torch.cat((weights, pad_w)),
    )


def _run_local(experts, local, recv_x, recv_topk_idx, recv_topk_weights):
    e_local = getattr(experts, "num_local_experts", experts.num_experts)
    idx, weights = _normalize_sentinels(recv_topk_idx, recv_topk_weights, e_local)
    n = recv_x.shape[0]
    recv_x, idx, weights = _pad_expert_rows(recv_x, idx, weights, e_local)
    return local(experts, recv_x, idx, weights)[:n]


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
            config=get_dispatch_config(),
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
        barrier_ep()
        grad_x, grad_topk_w, _ = buffer.combine(
            grad_recv_x.contiguous(),
            ctx.handle,
            topk_weights=topk_w_grad,
            config=get_combine_config(),
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
        combined, _, _ = buffer.combine(
            x.contiguous(), handle_holder.handle, config=get_combine_config()
        )
        return combined

    @staticmethod
    def backward(ctx, grad_combined):
        buffer = get_buffer()
        # backward-of-combine is dispatch: reuse the cached handle to send
        # gradients to the ranks that produced partial outputs.
        recv_grad, _, _, _, _, _ = buffer.dispatch(
            grad_combined.contiguous(), handle=ctx.handle, config=get_dispatch_config()
        )
        return recv_grad, None


class _DeepEPHandleHolder:
    """Carries the dispatch handle through to combine without breaking autograd
    (autograd.Function output must be Tensors or non-Tensor objects; tuples
    of opaque handles work as a passthrough since combine doesn't differentiate
    against them)."""

    def __init__(self, handle):
        self.handle = handle


class _DeepEPBackend:
    cast_bf16 = True

    @staticmethod
    def dispatch(
        hidden_states, topk_idx, topk_w, *, num_experts_global, num_local_experts
    ):
        buffer = get_buffer()
        # Layout is non-differentiable (bookkeeping only).
        with torch.no_grad():
            num_per_rank, _, num_per_expert, is_in_rank, _ = buffer.get_dispatch_layout(
                topk_idx, num_experts_global
            )
        return _DeepEPDispatch.apply(
            hidden_states, topk_idx, topk_w, num_per_rank, num_per_expert, is_in_rank
        )

    @staticmethod
    def combine(local_out, handle_holder):
        barrier_ep()  # wait out the local-kernel autotune skew before the combine collective
        return _DeepEPCombine.apply(local_out, handle_holder)


class _TorchBackend:
    cast_bf16 = False

    @staticmethod
    def group(num_experts_global, num_local_experts):
        if num_experts_global == num_local_experts:
            return None
        group = torch_dispatch.get_ep_group()
        num_ranks = num_experts_global // num_local_experts
        if group is None or dist.get_world_size(group) != num_ranks:
            raise RuntimeError(
                f"expert_parallel torch backend: experts are sharded {num_ranks}-way "
                f"({num_experts_global} global / {num_local_experts} local) but the EP "
                f"group is {'unset' if group is None else dist.get_world_size(group)}."
            )
        return group

    @staticmethod
    def dispatch(
        hidden_states, topk_idx, topk_w, *, num_experts_global, num_local_experts
    ):
        return torch_dispatch.dispatch(
            hidden_states,
            topk_idx,
            topk_w,
            num_local_experts=num_local_experts,
            group=_TorchBackend.group(num_experts_global, num_local_experts),
        )

    @staticmethod
    def combine(local_out, handle):
        return torch_dispatch.combine(local_out, handle)


_BACKENDS: dict[str, type[_DeepEPBackend] | type[_TorchBackend]] = {
    "deep_ep": _DeepEPBackend,
    "torch": _TorchBackend,
}


def _ep_forward(
    self,
    hidden_states,
    top_k_index,
    top_k_weights,
    *,
    local: str | None = None,
    backend: str | None = None,
):
    """Dispatch -> local experts implementation -> combine.

    Inputs come in with **global** routing indices (we do not run
    `transformers.RouterParallel`; see DEEP_EP.md §2.4 for why). Dispatch returns
    local expert ids in `[0, E_local)` with `-1` for slots routed to remote experts;
    `_run_local` rewrites those to `E_local` / weight 0 before the local implementation
    (``local``, default: the configured one) sees them. ``backend`` defaults to the one
    the plugin set.
    """
    backend = backend or _BACKEND
    if backend is None:
        raise RuntimeError(
            "expert_parallel: no dispatch backend set; the ExpertParallelPlugin sets it "
            "in pre_model_load (or call experts_fn.set_backend)."
        )
    ep_backend = _BACKENDS[backend]
    local_fn = resolve_local_implementation(local)
    _maybe_install_decorator_attrs(self)
    self._is_expert_parallel = True
    original_dtype = None
    if ep_backend.cast_bf16 and hidden_states.dtype != torch.bfloat16:
        original_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.bfloat16)

    E_global = getattr(self, "num_experts_global", self.num_experts)
    E_local = getattr(self, "num_local_experts", self.num_experts)

    topk_idx_i64 = top_k_index.to(torch.int64)
    topk_w_f32 = top_k_weights.to(torch.float32)

    # Drop padding tokens from the dispatch. Under non-packed training with
    # `pad_to_sequence_len`, padding rows carry identical embeddings, so the router
    # sends them all to the same one or two experts — a routing imbalance DeepEP's
    # intranode dispatch can't survive ('unspecified launch failure'). Sentinelling
    # their routing to -1 makes the dispatch layout skip them entirely (they get a
    # zero expert output, which is correct: their loss is masked anyway). Real long
    # sequences (packed, or a single long sample) have no padding and are unaffected.
    valid = _get_valid_token_mask()
    if valid is not None and valid.numel() == topk_idx_i64.shape[0]:
        import os as _os

        if _os.environ.get("AXOLOTL_EP_DEBUG"):
            from axolotl.utils.logging import get_logger

            get_logger(__name__).info(
                f"EP padding sentinel: {int(valid.sum())}/{valid.numel()} real tokens dispatched"
            )
        topk_idx_i64 = topk_idx_i64.masked_fill(~valid.view(-1, 1), -1)

    # Cap tokens-per-expert before building the dispatch layout — an overloaded expert deadlocks
    # DeepEP's intranode combine (the GLM router concentrates with depth). Runs after the padding
    # sentinel so identical pad rows can't occupy an expert's capacity and evict real tokens.
    if _TOKEN_CAPACITY is not None:
        topk_idx_i64, topk_w_f32 = _apply_expert_capacity(
            topk_idx_i64, topk_w_f32, _TOKEN_CAPACITY
        )

    group = (
        _TorchBackend.group(E_global, E_local)
        if backend == "torch" and _DISPATCH_CHUNKS > 1
        else None
    )
    if group is not None and dist.get_world_size(group) > 1:
        combined = torch_dispatch.dispatch_chunked_forward(
            hidden_states,
            topk_idx_i64,
            topk_w_f32,
            lambda rx, ri, rw: _run_local(self, local_fn, rx, ri, rw),
            num_local_experts=E_local,
            group=group,
            chunks=_DISPATCH_CHUNKS,
        )
    else:
        recv_x, recv_topk_idx, recv_topk_weights, handle = ep_backend.dispatch(
            hidden_states,
            topk_idx_i64,
            topk_w_f32,
            num_experts_global=E_global,
            num_local_experts=E_local,
        )
        local_out = _run_local(self, local_fn, recv_x, recv_topk_idx, recv_topk_weights)
        combined = ep_backend.combine(local_out, handle)

    if original_dtype is not None:
        combined = combined.to(original_dtype)
    return combined


def expert_parallel_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    return _ep_forward(self, hidden_states, top_k_index, top_k_weights)


REGISTRY = {EXPERT_PARALLEL: expert_parallel_experts_forward}


def _decorator_experts_interface():
    """The interface `@use_experts_implementation` binds as its default, which transformers'
    lazy import can make a different object than `ALL_EXPERTS_FUNCTIONS`."""
    import inspect

    try:
        from transformers.integrations import use_experts_implementation

        return (
            inspect.signature(use_experts_implementation)
            .parameters["experts_interface"]
            .default
        )
    except (ImportError, KeyError, AttributeError):
        return None


def register_all() -> None:
    """Register `expert_parallel` in `ALL_EXPERTS_FUNCTIONS` and whitelist it."""
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS
    from transformers.modeling_utils import PreTrainedModel

    interfaces = [ALL_EXPERTS_FUNCTIONS]
    canon = _decorator_experts_interface()
    if canon is not None and canon is not ALL_EXPERTS_FUNCTIONS:
        interfaces.append(canon)
    for interface in interfaces:
        for name, fn in REGISTRY.items():
            interface.register(name, fn)

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
