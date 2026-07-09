# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Frozen-base grouped LoRA forward/backward for the sonicmoe NVFP4 backend.

The base expert weights are frozen (NVFP4 in production; the sibling ``nvfp4``
module either dequantizes them or keeps them packed for the fp4_cute kernel).
Only the LoRA A/B tensors are trainable. Where the whole-weight path (``lora.MoELoRAMaterialize``) builds
``W_eff = W + scaling * (B @ A)`` per expert and hands it to an opaque CUTLASS
call, this module keeps the low-rank factors separate and fuses them at the
grouped-token level so it composes with a real grouped GEMM and never forms a
base-weight gradient.

The forward per expert group ``e`` is exactly ``x_e @ W_eff_e^T`` with
``W_eff_e = W_e + scaling * (B_e @ A_e)``, so the hand-written backward mirrors
``MoELoRAMaterialize.backward``: it forms the full ``dW_eff_e`` from the grouped
tokens, then maps it to ``dA_e / dB_e`` with the same rank-major reshape/permute.
Because ``W_e`` is frozen we only route grads to ``x``, ``lora_A`` and
``lora_B``; ``W`` is treated as a constant tensor for the ``dx`` term.

PEFT rank-major layout (matching ``lora.MoELoRAMaterialize``): for a weight of
logical shape ``[E, dim1, dim2]``, ``lora_A`` is ``[r*E, dim2]`` (E-outer,
r-inner rows) and ``lora_B`` is ``[dim1, r*E]`` (r-outer, E-inner cols).
"""

from __future__ import annotations

from typing import Optional

import torch

from .nvfp4 import (
    dequantize_expert_slice,
    dequantize_expert_weight,
    gated_activation,
    grouped_down_gemm,
    grouped_up_gemm,
    is_nvfp4_param,
)


def _use_grouped_mm(x: torch.Tensor) -> bool:
    """``torch._grouped_mm`` replaces the per-expert Python loops on sm90+.

    The loops cost one host sync per expert plus E small kernel launches; the
    grouped GEMM is a single launch with ragged (unaligned, possibly empty)
    segments, which matches expert token counts exactly.
    """
    return (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and hasattr(torch, "_grouped_mm")
        and torch.cuda.get_device_capability(x.device)[0] >= 9
    )


def _grouped_offs(expert_offsets: torch.Tensor, device) -> torch.Tensor:
    """``[E]`` int32 cumulative segment ends from ``[E+1]`` offsets."""
    return expert_offsets[1:].to(device=device, dtype=torch.int32)


def _b3d_contiguous(lora_B: torch.Tensor, E: int, dim1: int) -> torch.Tensor:
    """``[E, dim1, r]`` contiguous view of ``lora_B`` ``[dim1, r*E]``.

    The permute leaves no unit-stride dim; grouped GEMM needs one.
    """
    r = lora_B.shape[1] // E
    return lora_B.reshape(dim1, r, E).permute(2, 0, 1).contiguous()


def _lora_delta_per_group(
    x_grouped: torch.Tensor,
    expert_offsets: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    scaling: float,
    E: int,
    dim1: int,
    dim2: int,
    B_c: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Low-rank contribution ``scaling * ((x_e @ A_e^T) @ B_e^T)`` per group.

    ``lora_A`` is ``[r*E, dim2]``, ``lora_B`` is ``[dim1, r*E]``. Returns a
    ``[T, dim1]`` tensor aligned row-for-row with ``x_grouped`` ``[T, dim2]``.
    ``B_c`` is an optional precomputed ``_b3d_contiguous(lora_B, E, dim1)``.
    """
    r = lora_A.shape[0] // E
    A_3d = lora_A.reshape(E, r, dim2)  # [E, r, dim2]
    B_3d = lora_B.reshape(dim1, r, E).permute(2, 0, 1)  # [E, dim1, r]

    if _use_grouped_mm(x_grouped) and lora_A.dtype == x_grouped.dtype:
        offs = _grouped_offs(expert_offsets, x_grouped.device)
        z = torch._grouped_mm(x_grouped, A_3d.transpose(-2, -1), offs=offs)
        if B_c is None:
            B_c = B_3d.contiguous()
        # scaling applied on the small [T, r] intermediate, not the [T, dim1] product
        return torch._grouped_mm(z * scaling, B_c.transpose(-2, -1), offs=offs)

    out = x_grouped.new_zeros((x_grouped.shape[0], dim1))
    for e in range(E):
        start = int(expert_offsets[e])
        end = int(expert_offsets[e + 1])
        if end <= start:
            continue
        x_e = x_grouped[start:end]  # [T_e, dim2]
        z_e = x_e @ A_3d[e].transpose(0, 1)  # [T_e, r]
        out[start:end] = scaling * (z_e @ B_3d[e].transpose(0, 1))  # [T_e, dim1]
    return out


def _lora_backward_per_group(
    grad_h: torch.Tensor,
    x_grouped: torch.Tensor,
    expert_offsets: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    base_weight: torch.Tensor,
    scaling: float,
    B_c: Optional[torch.Tensor] = None,
) -> tuple:
    """Grads for a frozen-base grouped-LoRA linear ``h_e = x_e @ W_eff_e^T``.

    ``base_weight`` is ``[E, dim1, dim2]`` (frozen; dense or packed NVFP4,
    dequantized one expert slice at a time), used only for the ``dx`` term.
    ``B_c`` is an optional precomputed ``_b3d_contiguous(lora_B, E, dim1)``.
    Returns ``(dx, d_lora_A, d_lora_B)``.
    """
    E, dim1, dim2 = base_weight.shape
    r = lora_A.shape[0] // E
    A_3d = lora_A.reshape(E, r, dim2)  # [E, r, dim2]
    B_3d = lora_B.reshape(dim1, r, E).permute(2, 0, 1)  # [E, dim1, r]

    if _use_grouped_mm(grad_h) and lora_A.dtype == grad_h.dtype:
        offs = _grouped_offs(expert_offsets, grad_h.device)
        if B_c is None:
            # B_3d's permute leaves no unit-stride dim; grouped GEMM needs one.
            B_c = B_3d.contiguous()
        # dz_e = g_e @ B_e; z_e = x_e @ A_e^T (forward intermediate, recomputed:
        # cheaper than saving a [T, r] per GEMM through gradient checkpointing).
        # scaling lands once on each [T, r] intermediate, not the [T, dim] products.
        dz = torch._grouped_mm(grad_h, B_c, offs=offs) * scaling  # [T, r]
        z = torch._grouped_mm(x_grouped, A_3d.transpose(-2, -1), offs=offs) * scaling
        # dx_e = g_e @ W_e + dz_e @ A_e. Base term: fp8 weight cache on DeepGEMM
        # when available, else whole-weight dequant + one grouped GEMM. Either
        # way the weights match the forward operands (folded SFB + alpha on
        # fp4_cute; the fp8 cache is built from that same folded dequant).
        from .fp8_bwd import fp8_dx_supported, grouped_fp8_dx

        if fp8_dx_supported(grad_h, base_weight):
            dx = grouped_fp8_dx(grad_h, base_weight, expert_offsets)
        else:
            from .fp4_cute_ops import dequantize_engine_weight

            w_dense = dequantize_engine_weight(base_weight).to(grad_h.dtype)
            dx = torch._grouped_mm(grad_h, w_dense, offs=offs)
        dx += torch._grouped_mm(dz, A_3d, offs=offs)
        # dA_e = dz_e^T @ x_e; dB_e = g_e^T @ z_e
        d_A_3d = torch._grouped_mm(dz.transpose(0, 1), x_grouped, offs=offs)
        d_B_3d = torch._grouped_mm(grad_h.transpose(0, 1), z, offs=offs)
        d_lora_A = d_A_3d.reshape(E * r, dim2)
        d_lora_B = d_B_3d.permute(1, 2, 0).reshape(dim1, E * r)
        return dx, d_lora_A, d_lora_B

    dx = torch.zeros_like(x_grouped)
    d_A_3d = grad_h.new_zeros((E, r, dim2))
    d_B_3d = grad_h.new_zeros((E, dim1, r))

    for e in range(E):
        start = int(expert_offsets[e])
        end = int(expert_offsets[e + 1])
        if end <= start:
            continue
        x_e = x_grouped[start:end]  # [T_e, dim2]
        g_e = grad_h[start:end]  # [T_e, dim1]

        # dx_e = g_e @ W_eff_e with W_eff_e = W_e + scaling * (B_e @ A_e),
        # split so W_eff is never materialized and the NVFP4 base dequantizes
        # one expert slice at a time.
        w_e = dequantize_expert_slice(base_weight, e)  # [dim1, dim2]
        dx[start:end] = g_e @ w_e.to(g_e.dtype) + scaling * ((g_e @ B_3d[e]) @ A_3d[e])

        # dW_eff_e = grad_h_e^T @ x_e  ([dim1, dim2], the [E, dim1, dim2] convention)
        dW_e = g_e.transpose(0, 1) @ x_e  # [dim1, dim2]

        # Same map as MoELoRAMaterialize.backward:
        #   dA_e = scaling * B_e^T @ dW_e     ([r, dim1] @ [dim1, dim2] = [r, dim2])
        #   dB_e = scaling * dW_e @ A_e^T     ([dim1, dim2] @ [dim2, r] = [dim1, r])
        d_A_3d[e] = scaling * (B_3d[e].transpose(0, 1) @ dW_e)
        d_B_3d[e] = scaling * (dW_e @ A_3d[e].transpose(0, 1))

    d_lora_A = d_A_3d.reshape(E * r, dim2)
    d_lora_B = d_B_3d.permute(1, 2, 0).reshape(dim1, E * r)
    return dx, d_lora_A, d_lora_B


class GroupedUpProjLoRA(torch.autograd.Function):
    """Grouped up-projection with frozen base + trainable LoRA.

    ``h = base_up(x) + scaling * ((x @ A1^T) @ B1^T)`` per expert group, where
    ``base_up`` is the grouped GEMM ``x_e @ w1[e]^T``. ``w1`` is ``[E, 2I, H]``
    (dim1=2I, dim2=H); ``lora_A1`` is ``[r*E, H]``, ``lora_B1`` is ``[2I, r*E]``.
    Grads route to ``x_grouped``, ``lora_A1``, ``lora_B1`` only (``w1`` frozen).
    """

    @staticmethod
    def forward(
        ctx,
        x_grouped: torch.Tensor,
        w1: torch.Tensor,
        expert_offsets: torch.Tensor,
        lora_A1: torch.Tensor,
        lora_B1: torch.Tensor,
        scaling: float,
        backend: str,
        concat: bool,
    ) -> torch.Tensor:
        E, dim1, dim2 = w1.shape
        B_c = (
            _b3d_contiguous(lora_B1, E, dim1)
            if _use_grouped_mm(x_grouped) and lora_A1.dtype == x_grouped.dtype
            else None
        )
        delta = _lora_delta_per_group(
            x_grouped, expert_offsets, lora_A1, lora_B1, scaling, E, dim1, dim2, B_c=B_c
        )
        if backend == "fp4_cute":
            from .fp4_cute_ops import grouped_nvfp4_linear_add_delta

            h = grouped_nvfp4_linear_add_delta(x_grouped, w1, expert_offsets[1:], delta)
        else:
            base = grouped_up_gemm(
                x_grouped, w1, expert_offsets, backend=backend, concat=concat
            )
            h = base + delta

        ctx.save_for_backward(x_grouped, w1, expert_offsets, lora_A1, lora_B1, B_c)
        ctx.scaling = scaling
        return h

    @staticmethod
    def backward(ctx, grad_h: torch.Tensor):
        x_grouped, w1, expert_offsets, lora_A1, lora_B1, B_c = ctx.saved_tensors
        dx, dA, dB = _lora_backward_per_group(
            grad_h.contiguous(),
            x_grouped,
            expert_offsets,
            lora_A1,
            lora_B1,
            w1,
            ctx.scaling,
            B_c=B_c,
        )
        return dx, None, None, dA, dB, None, None, None


def _fused_up_act_enabled() -> bool:
    """Fuse up-GEMM + gated activation + LoRA-delta add into one fp4_cute
    gated-engine call. Default on; ``AXOLOTL_SONICMOE_NVFP4_FUSED_UP=0`` is the
    kill switch back to the unfused up-GEMM + separate activation."""
    import os

    return os.environ.get("AXOLOTL_SONICMOE_NVFP4_FUSED_UP", "1") != "0"


class GroupedUpProjActLoRA(torch.autograd.Function):
    """Fused grouped up-projection + gated activation with frozen NVFP4 base.

    One fp4_cute gated-engine call computes
    ``a = act(pts * base_up(x) + delta)`` per expert group: the LoRA delta
    rides the epilogue as a preact-space aux add, the per-expert pts as an
    exact per-row colvec multiply, and the activation runs on the fp32
    accumulator. The INTERLEAVED preact D is stashed for backward; grads route
    to ``x_grouped``, ``lora_A1``, ``lora_B1`` only (``w1`` frozen).
    """

    @staticmethod
    def forward(
        ctx,
        x_grouped: torch.Tensor,
        w1: torch.Tensor,
        expert_offsets: torch.Tensor,
        lora_A1: torch.Tensor,
        lora_B1: torch.Tensor,
        scaling: float,
    ) -> torch.Tensor:
        from .fp4_cute_ops import gated_nvfp4_forward
        from .sf_layout import gate_up_interleave_perm

        E, dim1, dim2 = w1.shape
        B_c = _b3d_contiguous(lora_B1, E, dim1)
        # The delta rides the kernel's INTERLEAVED preact space: permute the
        # small [2I, r*E] B factor's rows once so the [T, 2I] delta itself
        # never needs a gather. B_c stays concat for the shared backward.
        perm = gate_up_interleave_perm(dim1, device=lora_B1.device)
        B_il = lora_B1.index_select(0, perm)
        delta = _lora_delta_per_group(
            x_grouped, expert_offsets, lora_A1, B_il, scaling, E, dim1, dim2
        )
        cu = expert_offsets.to(torch.int32)
        postact, preact = gated_nvfp4_forward(
            x_grouped, w1, cu, delta.to(torch.bfloat16)
        )
        ctx.save_for_backward(
            x_grouped, w1, expert_offsets, lora_A1, lora_B1, B_c, preact
        )
        ctx.scaling = scaling
        return postact.to(x_grouped.dtype)

    @staticmethod
    def backward(ctx, grad_a: torch.Tensor):
        x_grouped, w1, expert_offsets, lora_A1, lora_B1, B_c, preact = ctx.saved_tensors
        # swiglu backward from the INTERLEAVED preact: even lanes gate, odd up.
        pre = preact.view(preact.shape[0], -1, 2)
        gate = pre[..., 0].float()
        up = pre[..., 1].float()
        sig = torch.sigmoid(gate)
        ga = grad_a.float()
        grad_gate = ga * up * (sig * (1.0 + gate * (1.0 - sig)))
        grad_up = ga * (gate * sig)
        # grad wrt the CONCAT-layout virtual preact h = [gate | up], matching
        # the concat layouts of w1 / lora_B1 in the shared backward.
        grad_h = torch.cat([grad_gate, grad_up], dim=1).to(grad_a.dtype)
        dx, dA, dB = _lora_backward_per_group(
            grad_h,
            x_grouped,
            expert_offsets,
            lora_A1,
            lora_B1,
            w1,
            ctx.scaling,
            B_c=B_c,
        )
        return dx, None, None, dA, dB, None


class GroupedDownProjLoRA(torch.autograd.Function):
    """Grouped down-projection with frozen base + trainable LoRA.

    ``y = base_down(a) + scaling * ((a @ A2^T) @ B2^T)`` per expert group, where
    ``base_down`` is the grouped GEMM ``a_e @ w2[e]^T``. ``w2`` is ``[E, H, I]``
    (dim1=H, dim2=I); ``lora_A2`` is ``[r*E, I]``, ``lora_B2`` is ``[H, r*E]``.
    Grads route to ``a_grouped``, ``lora_A2``, ``lora_B2`` only (``w2`` frozen).
    """

    @staticmethod
    def forward(
        ctx,
        a_grouped: torch.Tensor,
        w2: torch.Tensor,
        expert_offsets: torch.Tensor,
        lora_A2: torch.Tensor,
        lora_B2: torch.Tensor,
        scaling: float,
        backend: str,
    ) -> torch.Tensor:
        E, dim1, dim2 = w2.shape
        B_c = (
            _b3d_contiguous(lora_B2, E, dim1)
            if _use_grouped_mm(a_grouped) and lora_A2.dtype == a_grouped.dtype
            else None
        )
        delta = _lora_delta_per_group(
            a_grouped, expert_offsets, lora_A2, lora_B2, scaling, E, dim1, dim2, B_c=B_c
        )
        if backend == "fp4_cute":
            from .fp4_cute_ops import grouped_nvfp4_linear_add_delta

            y = grouped_nvfp4_linear_add_delta(a_grouped, w2, expert_offsets[1:], delta)
        else:
            base = grouped_down_gemm(a_grouped, w2, expert_offsets, backend=backend)
            y = base + delta

        ctx.save_for_backward(a_grouped, w2, expert_offsets, lora_A2, lora_B2, B_c)
        ctx.scaling = scaling
        return y

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        a_grouped, w2, expert_offsets, lora_A2, lora_B2, B_c = ctx.saved_tensors
        da, dA, dB = _lora_backward_per_group(
            grad_y.contiguous(),
            a_grouped,
            expert_offsets,
            lora_A2,
            lora_B2,
            w2,
            ctx.scaling,
            B_c=B_c,
        )
        return da, None, None, dA, dB, None, None


def _add_expert_bias(
    out: torch.Tensor,
    expert_offsets: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Add a per-expert bias ``[E, dim]`` to grouped rows ``[T, dim]``."""
    E = bias.shape[0]
    result = out
    for e in range(E):
        start = int(expert_offsets[e])
        end = int(expert_offsets[e + 1])
        if end <= start:
            continue
        result = result.index_add(
            0,
            torch.arange(start, end, device=out.device),
            bias[e].unsqueeze(0).expand(end - start, -1),
        )
    return result


def grouped_expert_mlp_lora(
    x_grouped: torch.Tensor,
    expert_offsets: torch.Tensor,
    w1: torch.Tensor,
    b1: Optional[torch.Tensor],
    w2: torch.Tensor,
    b2: Optional[torch.Tensor],
    lora1: Optional[tuple],
    lora2: Optional[tuple],
    *,
    act: str,
    backend: str,
    concat: bool,
    scaling1: float,
    scaling2: float,
    limit: Optional[float] = None,
) -> torch.Tensor:
    """Chain up-LoRA -> gated activation -> down-LoRA over grouped tokens.

    ``lora1`` / ``lora2`` are ``(lora_A, lora_B)`` tuples or ``None`` (``None``
    means plain base grouped GEMM, no low-rank path). ``b1`` / ``b2`` are
    optional per-expert biases ``[E, dim]``. ``limit`` is the clamped-SwiGLU
    bound (e.g. DeepSeek-V4). Returns ``y_grouped`` ``[T, H]``.
    """
    if (
        lora1 is not None
        and backend == "fp4_cute"
        and b1 is None
        and limit is None
        and concat
        and act in ("silu", "swiglu")
        and _fused_up_act_enabled()
    ):
        A1, B1 = lora1
        a = GroupedUpProjActLoRA.apply(x_grouped, w1, expert_offsets, A1, B1, scaling1)
    else:
        if lora1 is not None:
            A1, B1 = lora1
            h = GroupedUpProjLoRA.apply(
                x_grouped, w1, expert_offsets, A1, B1, scaling1, backend, concat
            )
        else:
            h = grouped_up_gemm(
                x_grouped, w1, expert_offsets, backend=backend, concat=concat
            )
        if b1 is not None:
            h = _add_expert_bias(h, expert_offsets, b1)

        a = gated_activation(h, act, concat=concat, limit=limit)

    if lora2 is not None:
        A2, B2 = lora2
        y = GroupedDownProjLoRA.apply(a, w2, expert_offsets, A2, B2, scaling2, backend)
    else:
        y = grouped_down_gemm(a, w2, expert_offsets, backend=backend)
    if b2 is not None:
        y = _add_expert_bias(y, expert_offsets, b2)

    return y


def route_and_group(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    num_experts: int,
) -> tuple:
    """Gather tokens into expert-sorted order for a grouped MoE forward.

    ``top_k_index`` / ``top_k_weights`` are ``[T, K]`` (per-token routed experts
    and combine weights). Returns ``(x_grouped [T*K, H], expert_offsets [E+1],
    gather_token_idx [T*K], weights_grouped [T*K])`` where ``x_grouped`` rows are
    sorted by expert. ``index_select`` keeps this differentiable wrt
    ``hidden_states``; the combine weights stay differentiable wrt the router.
    """
    T, K = top_k_index.shape
    device = hidden_states.device

    flat_expert = top_k_index.reshape(-1)
    # CUDA skips this guard: it costs a device-to-host sync per MoE layer, EP is
    # already rejected upstream, and non-EP routed ids come from a width-E topk.
    if (
        not flat_expert.is_cuda
        and flat_expert.numel()
        and int(flat_expert.max()) >= num_experts
    ):
        raise NotImplementedError(
            "sonicmoe NVFP4 path received routed expert id >= num_experts "
            f"({int(flat_expert.max())} >= {num_experts}); expert parallelism is "
            "not supported yet"
        )

    flat_weight = top_k_weights.reshape(-1).to(hidden_states.dtype)
    token_ids = torch.arange(T, device=device).repeat_interleave(K)

    sorted_experts, order = torch.sort(flat_expert, stable=True)
    gather_token_idx = token_ids[order]
    weights_grouped = flat_weight[order]
    if hidden_states.is_cuda:
        # autograd's backward for index_select is an atomic index_add; this
        # gathers in both directions instead (see _CombineByGather).
        x_grouped = _GatherRows.apply(hidden_states, gather_token_idx)
    else:
        x_grouped = hidden_states.index_select(0, gather_token_idx)

    # searchsorted on the sorted ids instead of bincount + cumsum: bincount's
    # output size is data-dependent, so it host-syncs every call.
    expert_offsets = torch.searchsorted(
        sorted_experts,
        torch.arange(num_experts + 1, device=device, dtype=sorted_experts.dtype),
    )

    return x_grouped, expert_offsets, gather_token_idx, weights_grouped


class _GatherRows(torch.autograd.Function):
    """``hidden[idx]`` whose backward gathers instead of scatter-adding.

    Every token id appears exactly K times in ``idx``, so ``dh[t]`` is the sum
    of the K grad rows a stable sort of ``idx`` places contiguously.
    """

    @staticmethod
    def forward(ctx, hidden, gather_token_idx):
        ctx.save_for_backward(gather_token_idx)
        ctx.num_tokens = hidden.shape[0]
        return hidden.index_select(0, gather_token_idx)

    @staticmethod
    def backward(ctx, grad):
        (gidx,) = ctx.saved_tensors
        t = ctx.num_tokens
        k = gidx.shape[0] // t
        by_token = torch.sort(gidx, stable=True).indices
        dh = (
            grad.index_select(0, by_token)
            .view(t, k, grad.shape[-1])
            .sum(1, dtype=torch.float32)
            .to(grad.dtype)
        )
        return dh, None


class _CombineByGather(torch.autograd.Function):
    """Router-weighted combine with gathers in BOTH directions (no atomics).

    Each token receives exactly ``K = rows / num_tokens`` contributions, so a
    stable sort of the token ids groups each token's rows contiguously and the
    combine is a gather + fixed-order sum; the backward w.r.t. the grouped rows
    is a plain gather of ``dout`` by token id. ``index_add`` (the scatter the
    naive version needs, and what autograd emits for ``index_select``) is
    atomic-bound and dominated the layer profile.
    """

    @staticmethod
    def forward(ctx, y_grouped, gather_token_idx, weights_grouped, num_tokens):
        k = y_grouped.shape[0] // num_tokens
        by_token = torch.sort(gather_token_idx, stable=True).indices
        scaled = y_grouped * weights_grouped.unsqueeze(-1)
        out = (
            scaled.index_select(0, by_token)
            .view(num_tokens, k, y_grouped.shape[-1])
            .sum(1, dtype=torch.float32)
            .to(y_grouped.dtype)
        )
        ctx.save_for_backward(y_grouped, gather_token_idx, weights_grouped)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        y_grouped, gather_token_idx, weights_grouped = ctx.saved_tensors
        g = grad_out.index_select(0, gather_token_idx)
        dy = g * weights_grouped.unsqueeze(-1)
        dw = (g.float() * y_grouped.float()).sum(-1).to(weights_grouped.dtype)
        return dy, None, dw, None


def combine_expert_outputs(
    y_grouped: torch.Tensor,
    gather_token_idx: torch.Tensor,
    weights_grouped: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    """Scale expert-sorted rows by router weights and combine to ``[T, H]``."""
    if y_grouped.is_cuda and y_grouped.shape[0] % max(num_tokens, 1) == 0:
        return _CombineByGather.apply(
            y_grouped, gather_token_idx, weights_grouped, num_tokens
        )
    out = y_grouped.new_zeros((num_tokens, y_grouped.shape[-1]))
    scaled = y_grouped * weights_grouped.unsqueeze(-1)
    return out.index_add(0, gather_token_idx, scaled)


def grouped_moe_reference_forward(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    w1: torch.Tensor,
    b1: Optional[torch.Tensor],
    w2: torch.Tensor,
    b2: Optional[torch.Tensor],
    lora1: Optional[tuple],
    lora2: Optional[tuple],
    num_experts: int,
    *,
    act: str,
    backend: str,
    concat: bool,
    scaling1: float,
    scaling2: float,
    limit: Optional[float] = None,
) -> torch.Tensor:
    """End-to-end NVFP4 MoE forward: route -> grouped gated MLP -> combine.

    The frozen base ``w1`` / ``w2`` may be packed NVFP4. ``backend="dequant"``
    dequantizes the whole base to dense once up front so both the forward and
    the hand-written backward operate on dense tensors; ``backend="fp4_cute"``
    keeps the weights packed and runs the in-kernel SM100 W4A4 grouped GEMM
    (quantized activations, chunked-dequant dX in backward). ``limit`` is the
    clamped-SwiGLU bound (e.g. DeepSeek-V4). Runs on CPU (dense base) so the
    path is validated without a GPU.
    """
    if backend == "dequant":
        w1 = dequantize_expert_weight(w1)
        w2 = dequantize_expert_weight(w2)
        backend = "torch"
    elif backend == "fp4_cute":
        from .fp4_cute import fp4_cute_available

        if not fp4_cute_available():
            raise RuntimeError(
                "backend='fp4_cute' requires an SM100/SM110 GPU with quack and "
                "the cutlass DSL installed"
            )
        if not (is_nvfp4_param(w1) and is_nvfp4_param(w2)):
            raise ValueError(
                "backend='fp4_cute' requires packed NVFP4 base weights; use "
                "backend='torch' for a dense base"
            )

    x_grouped, expert_offsets, gather_token_idx, weights_grouped = route_and_group(
        hidden_states, top_k_index, top_k_weights, num_experts
    )
    y_grouped = grouped_expert_mlp_lora(
        x_grouped,
        expert_offsets,
        w1,
        b1,
        w2,
        b2,
        lora1,
        lora2,
        act=act,
        backend=backend,
        concat=concat,
        scaling1=scaling1,
        scaling2=scaling2,
        limit=limit,
    )
    return combine_expert_outputs(
        y_grouped, gather_token_idx, weights_grouped, hidden_states.shape[0]
    )
