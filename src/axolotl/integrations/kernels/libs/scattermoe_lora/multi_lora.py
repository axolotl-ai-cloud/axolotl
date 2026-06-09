# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""ScatterMoE + multi-adapter LoRA.

Extends the single-adapter ScatterMoE+LoRA path to co-train many LoRA adapters
("tenants") over one frozen expert stack. An MoE token carries two routings: the
router's expert assignment and a per-row tenant id. The base expert GEMM keys on
the expert; the LoRA keys on the combined ``(expert, tenant)`` group, so LoRA is
stacked over ``E*T`` groups (``A:[E*T*R, K]``, ``B:[N, E*T*R]``).

The key trick: ``combined = expert * T + tenant`` is also sorted by expert, so a
single sort produces both the expert grouping (base) and the combined grouping
(LoRA). The base GEMM and both LoRA projections (forward and the dX/dA/dB
backward) are grouped GEMMs over the combined groups, expressed with the existing
``scatter2scatter`` op plus one small grouped-Gram kernel (``_grouped_gram_kernel``)
for the weight gradients.

The backward precomputes ``XA = X @ A^T`` (saved from the forward) and
``YB = dY @ B`` (computed once, shared by dX and dA) so dA/dB become plain grouped
Gram products with no per-output-block recompute -- the cost that made the shared
single-adapter split kernel scale poorly as tenant count (E*T groups) grows.

Single-adapter (T == 1) reduces to the original behaviour exactly.
"""

from itertools import product

import torch
import triton
import triton.language as tl

from .kernels import lora_ops, ops as base_ops
from .kernels.lora_ops import _block_r_for_rank, _bucket_m
from .parallel_experts import compileable_bincount


def _grouped_gram_configs():
    configs = []
    for block_wide, block_m, warps, stages in product(
        [32, 64, 128, 256], [32, 64, 128], [4, 8], [3, 4]
    ):
        configs.append(
            triton.Config(
                {"BLOCK_WIDE": block_wide, "BLOCK_M": block_m},
                num_warps=warps,
                num_stages=stages,
            )
        )
    return configs


@triton.autotune(configs=_grouped_gram_configs(), key=["M_BUCKET", "WIDE", "RANK_IS_I"])
@triton.jit
def _grouped_gram_kernel(
    P_ptr,
    stride_pm,
    stride_pd,
    Q_ptr,
    stride_qm,
    stride_qd,
    OUT_ptr,
    stride_og,
    stride_oi,
    stride_oj,
    offsets_ptr,
    M_BUCKET,
    WIDE: tl.constexpr,
    RANK: tl.constexpr,
    scaling,
    RANK_IS_I: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_WIDE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    allow_tf32: tl.constexpr,
):
    """Grouped Gram product: out[g] = scaling * (P_g^T @ Q_g), summed over the
    tokens of combined-group g (a contiguous [start, end) slice).

    Both LoRA weight gradients are this op once XA/YB are precomputed:
      dA[g] = scaling * YB_g^T @ X_g   (P=YB[M,R], Q=X[M,K]; rank is the I/row dim)
      dB[g] = scaling * DY_g^T @ XA_g  (P=DY[M,N], Q=XA[M,R]; rank is the J/col dim)

    The wide dim (K for dA, N for dB) is tiled; the rank dim fits one BLOCK_R.
    Every (group, wide-block) writes its full tile, so empty groups self-zero --
    no output pre-init needed. Output strides place the [I, J] tile into either
    lora_A-grad ([E*T*R, K]) or lora_B-grad ([N, E*T*R]) layout directly.
    """
    g = tl.program_id(0)
    w_blk = tl.program_id(1)

    start = tl.where(g == 0, 0, tl.load(offsets_ptr + g - 1, mask=g > 0, other=0))
    end = tl.load(offsets_ptr + g)
    num = end - start

    w_idx = w_blk * BLOCK_WIDE + tl.arange(0, BLOCK_WIDE)
    w_mask = w_idx < WIDE
    r_idx = tl.arange(0, BLOCK_R)
    r_mask = r_idx < RANK
    input_dtype = P_ptr.dtype.element_ty

    if RANK_IS_I:
        acc = tl.zeros((BLOCK_R, BLOCK_WIDE), dtype=tl.float32)
    else:
        acc = tl.zeros((BLOCK_WIDE, BLOCK_R), dtype=tl.float32)

    for i in range(tl.cdiv(num, BLOCK_M)):
        m_idx = start + i * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = m_idx < end
        if RANK_IS_I:
            p = tl.load(
                P_ptr + m_idx[:, None] * stride_pm + r_idx[None, :] * stride_pd,
                mask=m_mask[:, None] & r_mask[None, :],
                other=0.0,
            ).to(input_dtype)
            q = tl.load(
                Q_ptr + m_idx[:, None] * stride_qm + w_idx[None, :] * stride_qd,
                mask=m_mask[:, None] & w_mask[None, :],
                other=0.0,
            ).to(input_dtype)
            acc += tl.dot(tl.trans(p), q, allow_tf32=allow_tf32)
        else:
            p = tl.load(
                P_ptr + m_idx[:, None] * stride_pm + w_idx[None, :] * stride_pd,
                mask=m_mask[:, None] & w_mask[None, :],
                other=0.0,
            ).to(input_dtype)
            q = tl.load(
                Q_ptr + m_idx[:, None] * stride_qm + r_idx[None, :] * stride_qd,
                mask=m_mask[:, None] & r_mask[None, :],
                other=0.0,
            ).to(input_dtype)
            acc += tl.dot(tl.trans(p), q, allow_tf32=allow_tf32)

    acc = acc * scaling
    if RANK_IS_I:
        out_ptrs = (
            OUT_ptr
            + g * stride_og
            + r_idx[:, None] * stride_oi
            + w_idx[None, :] * stride_oj
        )
        out_mask = r_mask[:, None] & w_mask[None, :]
    else:
        out_ptrs = (
            OUT_ptr
            + g * stride_og
            + w_idx[:, None] * stride_oi
            + r_idx[None, :] * stride_oj
        )
        out_mask = w_mask[:, None] & r_mask[None, :]
    tl.store(out_ptrs, acc.to(OUT_ptr.dtype.element_ty), mask=out_mask)


def grouped_lora_weight_grads(
    grouped_grad_out: torch.Tensor,  # DY  [M*k, N], grouped
    grouped_x: torch.Tensor,  # X   [M*k, K], grouped
    yb: torch.Tensor,  # DY@B [M*k, R], grouped (reused from dX path)
    xa: torch.Tensor,  # X@A^T [M*k, R], grouped (saved from forward)
    lora_A: torch.Tensor,  # [E*T*R, K]
    lora_B: torch.Tensor,  # [N, E*T*R]
    combined_offsets: torch.Tensor,
    e_total: int,
    scaling: float,
):
    """dA/dB via grouped Gram GEMMs over precomputed XA/YB.

    Avoids the shared split kernel's per-output-block recompute of XA/YB (an
    inner reduction over K or N, repeated cdiv(wide_dim, BLOCK) times per group);
    that recompute is what makes the split path scale poorly as tenants grow.
    """
    rank = lora_A.size(0) // e_total
    k_dim = lora_A.size(1)
    n_dim = lora_B.size(0)
    block_r = _block_r_for_rank(rank)
    m_bucket = _bucket_m(max(1, grouped_x.size(0) // e_total))

    dA = torch.empty_like(lora_A)
    dB = torch.empty_like(lora_B)

    grid_a = lambda meta: (e_total, triton.cdiv(k_dim, meta["BLOCK_WIDE"]))  # noqa: E731
    _grouped_gram_kernel[grid_a](
        yb,
        yb.stride(0),
        yb.stride(1),
        grouped_x,
        grouped_x.stride(0),
        grouped_x.stride(1),
        dA,
        rank * dA.stride(0),
        dA.stride(0),
        dA.stride(1),
        combined_offsets,
        m_bucket,
        WIDE=k_dim,
        RANK=rank,
        scaling=scaling,
        RANK_IS_I=True,
        BLOCK_R=block_r,
        allow_tf32=lora_ops.ALLOW_TF32,
    )

    grid_b = lambda meta: (e_total, triton.cdiv(n_dim, meta["BLOCK_WIDE"]))  # noqa: E731
    _grouped_gram_kernel[grid_b](
        grouped_grad_out,
        grouped_grad_out.stride(0),
        grouped_grad_out.stride(1),
        xa,
        xa.stride(0),
        xa.stride(1),
        dB,
        rank * dB.stride(1),
        dB.stride(0),
        dB.stride(1),
        combined_offsets,
        m_bucket,
        WIDE=n_dim,
        RANK=rank,
        scaling=scaling,
        RANK_IS_I=False,
        BLOCK_R=block_r,
        allow_tf32=lora_ops.ALLOW_TF32,
    )
    return dA, dB


def build_multilora_routing(
    flat_expert_idxs: torch.Tensor,  # [M*k] expert per (token, top-k slot)
    flat_tenant_idxs: torch.Tensor,  # [M*k] tenant per (token, top-k slot)
    num_experts: int,
    num_tenants: int,
):
    """One sort by ``expert*T + tenant`` yields both groupings.

    Returns ``(sorted_expert_idxs, sorted_combined_idxs, sorted_scattered_idxs,
    expert_offsets, combined_offsets)``.
    """
    combined = flat_expert_idxs * num_tenants + flat_tenant_idxs
    sorted_combined, sorted_scattered = torch.sort(combined)
    combined_offsets = compileable_bincount(
        sorted_combined, num_experts * num_tenants
    ).cumsum(-1)
    sorted_expert = torch.div(sorted_combined, num_tenants, rounding_mode="floor")
    expert_offsets = compileable_bincount(sorted_expert, num_experts).cumsum(-1)
    return (
        sorted_expert.contiguous(),
        sorted_combined.contiguous(),
        sorted_scattered.contiguous(),
        expert_offsets,
        combined_offsets,
    )


class ScatterMoEMultiLoRA(torch.autograd.Function):
    """Y = X @ W[expert] + scaling * (X @ A[expert,tenant]^T) @ B[expert,tenant]^T,
    frozen W; gradients flow to the per-(expert,tenant) LoRA A/B and to X."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        expert_weights: torch.Tensor,  # [E, K, N], frozen
        k: int,
        sorted_expert_idxs: torch.Tensor,
        sorted_combined_idxs: torch.Tensor,
        sorted_scattered_idxs: torch.Tensor,
        expert_offsets: torch.Tensor,
        combined_offsets: torch.Tensor,
        lora_A: torch.Tensor,  # [E*T*R, K]
        lora_B: torch.Tensor,  # [N, E*T*R]
        scaling: float,
    ):
        if expert_weights.dtype != x.dtype:
            expert_weights = expert_weights.to(x.dtype)
        e_total = combined_offsets.size(0)  # E*T groups
        rank = lora_A.size(0) // e_total
        n_dim = expert_weights.size(-1)
        k_dim = expert_weights.size(1)

        with torch.device(x.device):
            # 1. base: Y = X @ W  (keyed by expert)
            output = base_ops.scatter2scatter(
                X=x,
                W=expert_weights,
                k=k,
                sorted_expert_idxs=sorted_expert_idxs,
                sorted_scattered_idxs=sorted_scattered_idxs,
            )
            # 2. XA = X @ A^T  (keyed by combined; grouped output [M*k, R])
            w_a = lora_A.reshape(e_total, rank, k_dim).permute(0, 2, 1).contiguous()
            xa = base_ops.scatter2scatter(
                X=x,
                W=w_a,
                k=k,
                sorted_expert_idxs=sorted_combined_idxs,
                sorted_scattered_idxs=sorted_scattered_idxs,
                y_grouped=True,
            )
            # 3. Y_lora = XA @ B^T  (keyed by combined)
            w_b = lora_B.t().reshape(e_total, rank, n_dim).contiguous()
            y_lora = base_ops.scatter2scatter(
                X=xa,
                W=w_b,
                k=1,
                sorted_expert_idxs=sorted_combined_idxs,
                sorted_scattered_idxs=sorted_scattered_idxs,
                x_grouped=True,
            )
            output.add_(y_lora, alpha=scaling)

            ctx.save_for_backward(
                x,
                lora_A,
                lora_B,
                sorted_expert_idxs,
                sorted_combined_idxs,
                sorted_scattered_idxs,
                expert_offsets,
                combined_offsets,
                xa,
            )
            ctx.expert_weights = expert_weights
            ctx.k = k
            ctx.scaling = scaling
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (
            x,
            lora_A,
            lora_B,
            sorted_expert_idxs,
            sorted_combined_idxs,
            sorted_scattered_idxs,
            expert_offsets,
            combined_offsets,
            xa,
        ) = ctx.saved_tensors
        w = ctx.expert_weights
        k = ctx.k
        scaling = ctx.scaling
        e_total = combined_offsets.size(0)
        rank = lora_A.size(0) // e_total
        k_dim = lora_A.size(1)
        n_dim = lora_B.size(0)

        with torch.device(grad_out.device):
            # one grouping (by scattered idx) serves both -- combined-sorted is
            # also expert-sorted
            grouped_grad_out = base_ops.group(
                grad_out, sorted_scattered_idxs, fan_out=1
            )
            grouped_x = base_ops.group(x, sorted_scattered_idxs, fan_out=k)

            # YB = dY @ B (keyed by combined). Computed once and reused for both
            # the LoRA weight grads (dA) and the LoRA input grad (dX_lora).
            w_dyb = lora_B.reshape(n_dim, e_total, rank).permute(1, 0, 2).contiguous()
            dyb = base_ops.scatter2scatter(
                X=grouped_grad_out,
                W=w_dyb,
                k=1,
                sorted_expert_idxs=sorted_combined_idxs,
                sorted_scattered_idxs=sorted_scattered_idxs,
                x_grouped=True,
                y_grouped=True,
            )

            # dA/dB as grouped Gram GEMMs over precomputed XA (saved fwd) and YB.
            # No per-output-block recompute of XA/YB -- the cost that made the
            # shared split kernel scale poorly as tenants (E*T groups) grow.
            d_lora_A, d_lora_B = grouped_lora_weight_grads(
                grouped_grad_out,
                grouped_x,
                dyb,
                xa,
                lora_A,
                lora_B,
                combined_offsets,
                e_total,
                scaling,
            )

            # dX = base (expert) + LoRA (combined)
            d_grouped = base_ops.scatter2scatter(
                X=grouped_grad_out,
                x_grouped=True,
                W=w.permute(0, 2, 1),
                sorted_expert_idxs=sorted_expert_idxs,
                sorted_scattered_idxs=sorted_scattered_idxs,
                k=1,
            )
            if scaling != 0.0:
                # dX_lora = scaling * YB @ A, keyed by combined groups (one more
                # grouped GEMM; mirrors the forward's XA @ B step).
                w_dxa = lora_A.reshape(e_total, rank, k_dim).contiguous()
                d_lora = base_ops.scatter2scatter(
                    X=dyb,
                    W=w_dxa,
                    k=1,
                    sorted_expert_idxs=sorted_combined_idxs,
                    sorted_scattered_idxs=sorted_scattered_idxs,
                    x_grouped=True,
                    y_grouped=True,
                )
                d_grouped[sorted_scattered_idxs] += scaling * d_lora

            if k == 1:
                d_input = d_grouped
            else:
                d_input = d_grouped.view(x.size(0), k, d_grouped.size(-1)).sum(-2)

        return (
            d_input,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            d_lora_A,
            d_lora_B,
            None,
        )


def scatter2scatter_multilora(
    x,
    expert_weights,
    k,
    sorted_expert_idxs,
    sorted_combined_idxs,
    sorted_scattered_idxs,
    expert_offsets,
    combined_offsets,
    lora_A,
    lora_B,
    scaling: float = 1.0,
):
    """Drop-in multi-adapter forward (autograd-backed)."""
    return ScatterMoEMultiLoRA.apply(
        x,
        expert_weights,
        k,
        sorted_expert_idxs,
        sorted_combined_idxs,
        sorted_scattered_idxs,
        expert_offsets,
        combined_offsets,
        lora_A,
        lora_B,
        scaling,
    )
