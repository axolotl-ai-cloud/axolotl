# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Grouped-Gram LoRA weight gradients (dA/dB) without per-output-block recompute.

The split kernel in ``lora_ops`` recomputes XA = X@A^T (resp. YB = dY@B) inside the
kernel, once per output dim-block, to avoid materializing it. For LoRA's small rank
those intermediates are tiny (``[M*k, R]``), so materializing them once and reducing
the per-output-block recompute is a large net win at high expert counts. dA/dB then
reduce to plain grouped Gram products: dA[g] = scaling * YB_g^T @ X_g,
dB[g] = scaling * DY_g^T @ XA_g (reduction over the group's tokens).
"""

import torch
import triton
import triton.language as tl

from . import lora_ops
from .lora_ops import _block_r_for_rank, _bucket_m


def _device_shared_mem_optin() -> int:
    """Opt-in max dynamic shared memory per block for the current CUDA device (bytes). Returns a
    large sentinel when CUDA is unavailable so the big-SMEM config is chosen by default."""
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).shared_memory_per_block_optin
    except Exception:  # pylint: disable=broad-except
        pass
    return 1 << 30


# The large tile (BLOCK_M=128, BLOCK_WIDE=128, num_stages=3) needs ~144 KiB of shared memory per
# block: it fits sm_80 (A100, ~163 KiB) and sm_90/sm_100 (H100/B200, ~228 KiB) but NOT sm_89 (L40S)
# or consumer sm_120 (RTX 6000 / 5090, ~99 KiB), where Triton raises OutOfResources.
_GRAM_LARGE_CONFIG_SMEM = 147456  # bytes


def _grouped_gram_configs():
    # ONE config sized to the device's shared memory. A single config keeps the autotune "sweep"
    # instant — sweeping across configs re-times per rank at scattered layers and trips DeepEP's
    # combine barrier (why this isn't a full per-shape search). Big-SMEM GPUs (H100/B200/A100) run
    # the larger tile; small-SMEM GPUs (L40S / RTX 6000 / 5090) run a halved BLOCK_WIDE + double
    # buffering that fits ~99 KiB (<= ~64 KiB even at BLOCK_R=128) instead of OOM-ing.
    if _device_shared_mem_optin() >= _GRAM_LARGE_CONFIG_SMEM:
        cfg = triton.Config(
            {"BLOCK_WIDE": 128, "BLOCK_M": 128},
            num_warps=4,
            num_stages=3,
            num_ctas=1,
        )
    else:
        cfg = triton.Config(
            {"BLOCK_WIDE": 64, "BLOCK_M": 128},
            num_warps=4,
            num_stages=2,
            num_ctas=1,
        )
    return [cfg]


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
