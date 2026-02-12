# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""
Fused ScatterMoE + LoRA Triton Kernels
=======================================

Provides fused forward and backward kernels for ScatterMoE with LoRA adapters.

Forward: Y = X @ W + scaling * (X @ A^T) @ B^T
Backward (LoRA training, W frozen):
  - dX = dY @ W^T + scaling * (dY @ B) @ A    (input gradient)
  - dA = scaling * (dY @ B)^T @ X              (LoRA A gradient)
  - dB = scaling * dY^T @ (X @ A^T)            (LoRA B gradient)

LoRA weight layout (from PEFT ParamWrapper):
  - A: [r*E, K]  -- for expert e, rows [e*r : (e+1)*r] give A_e of shape [r, K]
  - B: [N, r*E]  -- for expert e, cols [e*r : (e+1)*r] give B_e of shape [N, r]

Key design decisions:
  - The forward kernel fuses X@W and X@A^T in the same K-loop for data reuse on X,
    then computes (X@A^T) @ B^T in the epilogue.
  - The backward dA/dB kernel operates on grouped (expert-contiguous) data and
    iterates over tokens per expert, accumulating gradients in registers.
  - R (LoRA rank) is a tl.constexpr, allowing tl.arange(0, R). We pad R to a
    power-of-2 for Triton tile compatibility; typical ranks (4, 8, 16, 32, 64)
    already satisfy this.
"""

from itertools import product
from typing import Optional

import torch
import triton
import triton.language as tl

# =============================================================================
# Configuration
# =============================================================================

BLOCK_M = 128
ALLOW_TF32 = True


def _next_power_of_2(n: int) -> int:
    """Round up to next power of 2."""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


# Triton tl.dot requires minimum tile dimensions of 16 on modern GPUs.
MIN_TRITON_DOT_SIZE = 16


def _block_r_for_rank(r: int) -> int:
    """Compute BLOCK_R: next power-of-2 >= max(r, MIN_TRITON_DOT_SIZE)."""
    return _next_power_of_2(max(r, MIN_TRITON_DOT_SIZE))


# =============================================================================
# Token Rounding: pad expert counts to BLOCK_M multiples
# =============================================================================


def round_expert_counts(
    sorted_expert_idxs: torch.Tensor,
    sorted_scattered_idxs: torch.Tensor,
    expert_offsets: torch.Tensor,
    E: int,
    block_m: int = BLOCK_M,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pad each expert's token count to a multiple of block_m to eliminate
    partial-tile waste in the backward kernel.

    Padding is done by duplicating the last valid token index for each expert.
    The kernel's M_mask = M_idx < real_end_idx masks these padding entries, so
    correctness is preserved (they contribute 0 to the accumulation via other=0.0).

    This only helps the backward dA/dB kernel where per-expert iteration is
    explicit. The forward scatter2scatter kernel handles partial tiles via masking.

    Args:
        sorted_expert_idxs: Expert assignments sorted [M*k]
        sorted_scattered_idxs: Original indices sorted [M*k]
        expert_offsets: Cumulative token counts per expert [E]
        E: Number of experts
        block_m: Block size for token dimension (default: BLOCK_M)

    Returns:
        padded_expert_idxs: [M_padded] expert assignments with padding
        padded_scattered_idxs: [M_padded] original indices with padding
        padded_offsets: [E] cumulative padded counts (for kernel iteration range)
        real_offsets: [E] original cumulative counts (for M_mask in kernel)
    """
    device = sorted_expert_idxs.device

    # Compute per-expert counts
    counts = torch.zeros(E, dtype=torch.int64, device=device)
    prev = 0
    for e in range(E):
        curr = expert_offsets[e].item()
        counts[e] = curr - prev
        prev = curr

    # Round up each count to multiple of block_m
    padded_counts = ((counts + block_m - 1) // block_m) * block_m
    # Experts with 0 tokens stay at 0
    padded_counts = torch.where(
        counts > 0, padded_counts, torch.zeros_like(padded_counts)
    )
    total_padded = padded_counts.sum().item()

    padded_expert_idxs = torch.empty(
        total_padded, dtype=sorted_expert_idxs.dtype, device=device
    )
    padded_scattered_idxs = torch.empty(
        total_padded, dtype=sorted_scattered_idxs.dtype, device=device
    )

    src_offset = 0
    dst_offset = 0
    for e in range(E):
        count = counts[e].item()
        padded_count = padded_counts[e].item()

        if count > 0:
            # Copy original tokens
            padded_expert_idxs[dst_offset : dst_offset + count] = sorted_expert_idxs[
                src_offset : src_offset + count
            ]
            padded_scattered_idxs[dst_offset : dst_offset + count] = (
                sorted_scattered_idxs[src_offset : src_offset + count]
            )

            # Pad with last valid token (masked out by kernel via M_mask)
            if padded_count > count:
                padded_expert_idxs[dst_offset + count : dst_offset + padded_count] = (
                    sorted_expert_idxs[src_offset + count - 1]
                )
                padded_scattered_idxs[
                    dst_offset + count : dst_offset + padded_count
                ] = sorted_scattered_idxs[src_offset + count - 1]

        src_offset += count
        dst_offset += padded_count

    # Padded offsets: cumulative padded counts (for iteration range in kernel)
    padded_offsets = padded_counts.cumsum(-1).to(expert_offsets.dtype)
    # Real offsets: original cumulative counts (for M_mask in kernel)
    real_offsets = expert_offsets.clone()

    return padded_expert_idxs, padded_scattered_idxs, padded_offsets, real_offsets


# =============================================================================
# Autotuning: SMEM estimation and config pruning
# =============================================================================

_SMEM_CAPACITY: int | None = None


def _get_smem_capacity() -> int:
    """Get device shared memory capacity (bytes). Cached after first call."""
    global _SMEM_CAPACITY
    if _SMEM_CAPACITY is None:
        props = triton.runtime.driver.active.utils.get_device_properties(
            torch.cuda.current_device()
        )
        _SMEM_CAPACITY = props["max_shared_mem"]
    return _SMEM_CAPACITY


def _estimate_smem_usage(
    num_stages: int, BLOCK_M: int, BLOCK_N: int, BLOCK_K: int, dtype_bytes: int = 2
) -> int:
    """Estimate shared memory in bytes for a GEMM-style tile.

    Formula: stages * BLOCK_K * (BLOCK_M + BLOCK_N) + BLOCK_M * BLOCK_N
    Multiply by dtype_bytes (2 for fp16/bf16).
    """
    return (
        num_stages * BLOCK_K * (BLOCK_M + BLOCK_N) + BLOCK_M * BLOCK_N
    ) * dtype_bytes


# Conservative margin (bytes) subtracted from SMEM capacity to account for
# estimation inaccuracies and kernel overhead (registers spilled to SMEM, etc.)
_SMEM_SLACK = 10_000


# =============================================================================
# Forward Kernel: scatter2scatter with fused LoRA
# =============================================================================


@triton.jit
def _compute_expert_block_lora(
    E_idx,
    E_mask,
    M_in_idx,
    N_block,
    N_mask,
    # Base weight
    X_ptr,
    stride_xm,
    stride_xk,
    W_ptr,
    stride_we,
    stride_wk,
    stride_wn,
    # LoRA weights
    A_ptr,
    stride_ar,
    stride_ak,  # A: [r*E, K], stride_ar = stride for r*E dim, stride_ak = stride for K dim
    B_ptr,
    stride_bn,
    stride_br,  # B: [N, r*E], stride_bn = stride for N dim, stride_br = stride for r*E dim
    # Dimensions
    K,
    ACTUAL_R: tl.constexpr,  # True LoRA rank (for indexing into weight arrays)
    acc,
    no_k_mask,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,  # Padded tile size >= max(ACTUAL_R, 16)
    scaling,
    allow_tf32: tl.constexpr,
):
    """
    Compute Y_block = X_block @ W_e + scaling * (X_block @ A_e^T) @ B_e^T

    for tokens in this M-block assigned to expert E_idx.

    ACTUAL_R is the true LoRA rank used for indexing into A[e*r:(e+1)*r, :].
    BLOCK_R >= ACTUAL_R is the padded tile dimension (must be >= 16 for tl.dot).
    When BLOCK_R > ACTUAL_R, loads are masked on the R dimension.
    """
    K_block = tl.arange(0, BLOCK_K)
    R_block = tl.arange(0, BLOCK_R)
    R_mask = R_block < ACTUAL_R  # Mask for padding when BLOCK_R > ACTUAL_R

    # Base weight pointers: W[E_idx, :, :] is [K, N], load [BLOCK_K, BLOCK_N]
    X_blk_ptrs = X_ptr + M_in_idx[:, None] * stride_xm + K_block[None, :] * stride_xk
    W_blk_ptrs = (
        W_ptr
        + E_idx * stride_we
        + K_block[:, None] * stride_wk
        + N_block[None, :] * stride_wn
    )

    # LoRA A pointers: A[e*ACTUAL_R:(e+1)*ACTUAL_R, :] for expert e, shape [r, K]
    A_expert_offset = E_idx * ACTUAL_R
    A_blk_ptrs = (
        A_ptr
        + (A_expert_offset + R_block)[:, None] * stride_ar
        + K_block[None, :] * stride_ak
    )

    iters = tl.cdiv(K, BLOCK_K)

    # Accumulator for X @ A^T: [BLOCK_M, BLOCK_R]
    xa_acc = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)

    # Determine the input element type for consistent casting.
    # Masked tl.load with other=0.0 can upcast bf16->fp32 in some Triton versions,
    # causing dtype mismatches in tl.dot.  We cast all tiles to the same type.
    INPUT_DTYPE = X_ptr.dtype.element_ty

    for i in range(iters):
        if no_k_mask:
            x = tl.load(X_blk_ptrs, mask=E_mask[:, None], other=0.0).to(INPUT_DTYPE)
            w = tl.load(W_blk_ptrs, mask=N_mask[None, :], other=0.0).to(INPUT_DTYPE)
            a = tl.load(A_blk_ptrs, mask=R_mask[:, None], other=0.0).to(INPUT_DTYPE)
        else:
            K_mask = (i * BLOCK_K + K_block) < K
            x = tl.load(
                X_blk_ptrs, mask=E_mask[:, None] & K_mask[None, :], other=0.0
            ).to(INPUT_DTYPE)
            w = tl.load(
                W_blk_ptrs, mask=K_mask[:, None] & N_mask[None, :], other=0.0
            ).to(INPUT_DTYPE)
            a = tl.load(
                A_blk_ptrs, mask=R_mask[:, None] & K_mask[None, :], other=0.0
            ).to(INPUT_DTYPE)

        # Base: acc += X @ W  ([M, K] @ [K, N] -> [M, N])
        acc += tl.dot(x, w, allow_tf32=allow_tf32).to(tl.float32)

        # LoRA: xa_acc += X @ A^T  ([M, K] @ [K, R] -> [M, R])
        xa_acc += tl.dot(x, tl.trans(a), allow_tf32=allow_tf32).to(tl.float32)

        X_blk_ptrs += BLOCK_K * stride_xk
        W_blk_ptrs += BLOCK_K * stride_wk
        A_blk_ptrs += BLOCK_K * stride_ak

    # Epilogue: load B[e] and compute (X @ A^T) @ B^T
    # B[e] is B[:, e*ACTUAL_R:(e+1)*ACTUAL_R], shape [N, r]. Load [BLOCK_N, BLOCK_R].
    B_expert_offset = E_idx * ACTUAL_R
    B_blk_ptrs = (
        B_ptr
        + N_block[:, None] * stride_bn
        + (B_expert_offset + R_block)[None, :] * stride_br
    )
    b = tl.load(
        B_blk_ptrs, mask=N_mask[:, None] & R_mask[None, :], other=0.0
    )  # [BLOCK_N, BLOCK_R]

    # Cast xa_acc and b to same dtype for tl.dot (required when input is bf16/fp16)
    # Both operands must match; cast to float32 (accumulator type) for precision.
    b_f32 = b.to(tl.float32)

    # (X @ A^T) @ B^T: [M, R] @ [R, N] -> [M, N]
    lora_out = tl.dot(xa_acc, tl.trans(b_f32), allow_tf32=allow_tf32)

    acc += scaling * lora_out
    return acc


def _scatter2scatter_lora_configs():
    """Generate forward kernel autotune configs.

    Search space includes smaller tile sizes and fewer pipeline stages to
    support GPUs with limited shared memory (e.g. ~99KB on some GPUs).

    Search space:
      BLOCK_N:    {32, 64, 128, 256}
      BLOCK_K:    {32, 64, 128}
      num_warps:  {4, 8}
      num_stages: {3, 4, 5}

    BLOCK_M is fixed at 128 (module-level constant, not autotuned in the
    scatter2scatter pattern).
    """
    configs = []
    for block_n, block_k, warps, stages in product(
        [32, 64, 128, 256],  # BLOCK_N
        [32, 64, 128],  # BLOCK_K
        [4, 8],  # num_warps
        [3, 4, 5],  # num_stages
    ):
        configs.append(
            triton.Config(
                {"BLOCK_N": block_n, "BLOCK_K": block_k},
                num_stages=stages,
                num_warps=warps,
            )
        )
    return configs


def _prune_fwd_configs(configs, named_args, **kwargs):
    """Prune forward configs based on SMEM capacity.

    The forward kernel inner loop loads three tiles per pipeline stage:
      X[BLOCK_M, BLOCK_K], W[BLOCK_K, BLOCK_N], A[BLOCK_R, BLOCK_K].
    The base estimate only accounts for X and W. We add:
      - A tile [BLOCK_R, BLOCK_K] per pipeline stage (loaded in the inner loop)
      - B tile [BLOCK_N, BLOCK_R] loaded once in the epilogue
      - Extra headroom for compiler overhead (register spills, metadata)
    """
    smem_cap = _get_smem_capacity()

    # Get BLOCK_R from named_args if available, else assume worst case
    block_r = named_args.get("BLOCK_R", 64)

    scored = []
    for config in configs:
        block_n = config.kwargs["BLOCK_N"]
        block_k = config.kwargs["BLOCK_K"]
        # Base: stages * BLOCK_K * (BLOCK_M + BLOCK_N) + BLOCK_M * BLOCK_N
        smem_base = _estimate_smem_usage(config.num_stages, BLOCK_M, block_n, block_k)
        # A tile [BLOCK_R, BLOCK_K] loaded per stage in the inner loop
        smem_lora_loop = config.num_stages * block_r * block_k * 2
        # B tile [BLOCK_N, BLOCK_R] loaded once in epilogue
        smem_lora_epilogue = block_n * block_r * 2
        smem = smem_base + smem_lora_loop + smem_lora_epilogue
        scored.append((smem, config))

    pruned = [c for s, c in scored if s <= smem_cap - _SMEM_SLACK]
    if pruned:
        return pruned
    # All configs exceed SMEM — return the one with smallest estimated usage
    scored.sort(key=lambda x: x[0])
    return [scored[0][1]]


@triton.autotune(
    configs=_scatter2scatter_lora_configs(),
    key=["M", "N", "K"],
    prune_configs_by={"early_config_prune": _prune_fwd_configs},
)
@triton.heuristics(
    {
        "NO_K_MASK": lambda args: (args["K"] % args["BLOCK_K"]) == 0,
        "NO_N_MASK": lambda args: (args["N"] % args["BLOCK_N"]) == 0,
    }
)
@triton.jit
def _scatter2scatter_lora(
    # Input/Output
    X_ptr,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    W_ptr,
    stride_we,
    stride_wk: tl.constexpr,
    stride_wn: tl.constexpr,
    Y_ptr,
    stride_ym: tl.constexpr,
    stride_yn: tl.constexpr,
    # Bias
    Bias_ptr,
    stride_bias_e: tl.constexpr,
    stride_bias_n: tl.constexpr,
    # LoRA weights
    LA_ptr,
    stride_la_r,
    stride_la_k,  # A: [r*E, K]
    LB_ptr,
    stride_lb_n,
    stride_lb_r,  # B: [N, r*E]
    # Routing
    grouped_idx_ptr,
    expert_idxs_ptr,
    # Dimensions
    FAN_OUT: tl.constexpr,
    M,
    K: tl.constexpr,
    N: tl.constexpr,
    E: tl.constexpr,
    ACTUAL_R: tl.constexpr,  # True LoRA rank (for weight indexing)
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_R: tl.constexpr,  # Padded tile size >= max(ACTUAL_R, 16)
    # Config
    ACC_TYPE: tl.constexpr,
    scaling,
    allow_tf32: tl.constexpr,
    x_grouped: tl.constexpr,
    y_grouped: tl.constexpr,
    NO_K_MASK: tl.constexpr,
    NO_N_MASK: tl.constexpr,
):
    """
    Fused scatter2scatter with LoRA: Y = X @ W + scaling * (X @ A^T) @ B^T + bias
    """
    pid = tl.program_id(axis=0)

    N_BLOCK_COUNT = tl.cdiv(N, BLOCK_N)
    M_block_id = pid // N_BLOCK_COUNT
    N_block_id = pid % N_BLOCK_COUNT

    M_block = M_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    N_block = N_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
    N_mask = N_block < N
    M_boundary_mask = M_block < (FAN_OUT * M)

    E_idxs = tl.load(expert_idxs_ptr + M_block, mask=M_boundary_mask, other=E)

    no_k_mask = NO_K_MASK

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    E_first_idx = tl.min(E_idxs)
    E_last_idx = tl.minimum(tl.max(E_idxs), E - 1)
    M_idx = tl.load(grouped_idx_ptr + M_block, mask=M_boundary_mask).to(tl.int32)

    for E_idx in range(E_first_idx, E_last_idx + 1):
        E_mask = E_idxs == E_idx
        if x_grouped:
            M_in_idx = M_block
        else:
            M_in_idx = M_idx // FAN_OUT

        acc = _compute_expert_block_lora(
            E_idx,
            E_mask,
            M_in_idx,
            N_block,
            N_mask,
            X_ptr,
            stride_xm,
            stride_xk,
            W_ptr,
            stride_we,
            stride_wk,
            stride_wn,
            LA_ptr,
            stride_la_r,
            stride_la_k,
            LB_ptr,
            stride_lb_n,
            stride_lb_r,
            K,
            ACTUAL_R,
            acc,
            no_k_mask,
            BLOCK_M,
            BLOCK_K,
            BLOCK_N,
            BLOCK_R,
            scaling,
            allow_tf32=allow_tf32,
        )

    # Add bias if present
    if Bias_ptr is not None:
        B_blk_ptrs = (
            Bias_ptr
            + E_idxs[:, None] * stride_bias_e
            + N_block[None, :] * stride_bias_n
        )
        acc += tl.load(B_blk_ptrs, mask=M_boundary_mask[:, None] & N_mask[None, :])

    # Store output
    if y_grouped:
        M_out_idx = M_block
    else:
        M_out_idx = M_idx
    Y_blk_ptrs = Y_ptr + (M_out_idx[:, None] * stride_ym + N_block[None, :] * stride_yn)
    tl.store(Y_blk_ptrs, acc, mask=M_boundary_mask[:, None] & N_mask[None, :])


def scatter2scatter_lora(
    X: torch.Tensor,
    W: torch.Tensor,
    sorted_expert_idxs: torch.Tensor,
    sorted_scattered_idxs: torch.Tensor,
    k: int,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    scaling: float,
    b: Optional[torch.Tensor] = None,
    x_grouped: bool = False,
    y_grouped: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Fused scatter2scatter with LoRA: Y[i] = X[i] @ W[e] + scaling * (X[i] @ A[e]^T) @ B[e]^T + b[e]

    Args:
        X: Input [M, K] or [M*k, K] if x_grouped
        W: Expert weights [E, K, N]
        sorted_expert_idxs: Expert assignments sorted [M*k]
        sorted_scattered_idxs: Original indices sorted [M*k]
        k: Fan-out (top-k)
        lora_A: LoRA A weights [r*E, K]
        lora_B: LoRA B weights [N, r*E]
        scaling: LoRA scaling factor (alpha/r)
        b: Optional bias [E, N]
        x_grouped: Input pre-grouped by expert
        y_grouped: Keep output grouped
        out: Optional pre-allocated output buffer

    Returns:
        Y: Output [M*k, N]
    """
    assert sorted_scattered_idxs.size(0) == sorted_expert_idxs.size(0)
    assert sorted_scattered_idxs.size(0) == X.size(0) * k

    E = W.size(0)
    K = W.size(1)
    N = W.size(2)
    R = lora_A.size(0) // E

    # Pad R to power of 2 for Triton tile size
    BLOCK_R = _block_r_for_rank(R)

    L_scattered = sorted_expert_idxs.size(0)

    if out is None:
        output = torch.empty((L_scattered, N), device=X.device, dtype=X.dtype)
    else:
        assert out.size(0) == L_scattered and out.size(1) == N
        output = out

    def grid(META):
        return (
            triton.cdiv(L_scattered, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        )

    if b is None:
        stride_be = stride_bn = 0
        b_ptr = None
    else:
        stride_be, stride_bn = b.stride()
        b_ptr = b

    _scatter2scatter_lora[grid](
        X,
        X.stride(0),
        X.stride(1),
        W,
        W.stride(0),
        W.stride(1),
        W.stride(2),
        output,
        output.stride(0),
        output.stride(1),
        b_ptr,
        stride_be,
        stride_bn,
        # A: [r*E, K] -> stride(0) is r*E dim stride, stride(1) is K dim stride
        lora_A,
        lora_A.stride(0),
        lora_A.stride(1),
        # B: [N, r*E] -> stride(0) is N dim stride, stride(1) is r*E dim stride
        lora_B,
        lora_B.stride(0),
        lora_B.stride(1),
        sorted_scattered_idxs,
        sorted_expert_idxs,
        FAN_OUT=k,
        M=X.size(0),
        K=K,
        N=N,
        E=E,
        ACTUAL_R=R,  # True LoRA rank for weight indexing
        BLOCK_M=BLOCK_M,
        BLOCK_R=BLOCK_R,  # Padded tile size >= max(R, 16)
        ACC_TYPE=tl.float32,
        scaling=scaling,
        allow_tf32=ALLOW_TF32,
        x_grouped=x_grouped,
        y_grouped=y_grouped,
    )

    return output


# =============================================================================
# Backward Kernel: Fused dX = dY @ W^T + scaling * (dY @ B) @ A
# =============================================================================


@triton.jit
def _compute_expert_block_lora_dX(
    E_idx,
    E_mask,
    M_in_idx,
    K_block,
    K_mask,
    # Input: DY (gradient w.r.t. output)
    DY_ptr,
    stride_dym,
    stride_dyn,
    # Base weight W^T: we load W[e] as [K, N] and index as W^T[e] = [N, K]
    W_ptr,
    stride_we,
    stride_wk,
    stride_wn,
    # LoRA weights
    A_ptr,
    stride_ar,
    stride_ak,  # A: [r*E, K]
    B_ptr,
    stride_bn,
    stride_br,  # B: [N, r*E]
    # Dimensions
    N,
    ACTUAL_R: tl.constexpr,
    acc,
    no_n_mask,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_R: tl.constexpr,
    scaling,
    allow_tf32: tl.constexpr,
):
    """
    Compute dX_block = DY_block @ W_e^T + scaling * (DY_block @ B_e) @ A_e

    for tokens in this M-block assigned to expert E_idx.

    Inner loop over N dimension (reduction dim for dY @ W^T and dY @ B).
    Output dimension is K.
    Epilogue computes (dY @ B) @ A.

    Transpose mapping from forward:
      Forward: X@W (K-loop), X@A^T (K-loop), (X@A^T)@B^T (epilogue)
      Backward: DY@W^T (N-loop), DY@B (N-loop), (DY@B)@A (epilogue)
    """
    N_block = tl.arange(0, BLOCK_N)
    R_block = tl.arange(0, BLOCK_R)
    R_mask = R_block < ACTUAL_R

    # DY pointers: DY is [M_total, N], load [BLOCK_M, BLOCK_N]
    DY_blk_ptrs = (
        DY_ptr + M_in_idx[:, None] * stride_dym + N_block[None, :] * stride_dyn
    )

    # W^T pointers: W[e] is [K, N], W^T[e] is [N, K]. We load W^T as [BLOCK_N, BLOCK_K].
    # W stored as [E, K, N], so W^T[e][n, k] = W[e][k, n] = W_ptr + e*stride_we + k*stride_wk + n*stride_wn
    # As [BLOCK_N, BLOCK_K] tile: row=n, col=k
    WT_blk_ptrs = (
        W_ptr
        + E_idx * stride_we
        + N_block[:, None] * stride_wn  # row = n dimension
        + K_block[None, :] * stride_wk
    )  # col = k dimension

    # B pointers: B[e] is B[:, e*R:(e+1)*R], shape [N, R]. Load [BLOCK_N, BLOCK_R].
    B_expert_offset = E_idx * ACTUAL_R
    B_blk_ptrs = (
        B_ptr
        + N_block[:, None] * stride_bn
        + (B_expert_offset + R_block)[None, :] * stride_br
    )

    iters = tl.cdiv(N, BLOCK_N)

    # Accumulator for DY @ B: [BLOCK_M, BLOCK_R]
    dy_b_acc = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)

    # Determine the input element type for consistent casting.
    INPUT_DTYPE = DY_ptr.dtype.element_ty

    for i in range(iters):
        if no_n_mask:
            dy = tl.load(DY_blk_ptrs, mask=E_mask[:, None], other=0.0).to(INPUT_DTYPE)
            wt = tl.load(WT_blk_ptrs, mask=K_mask[None, :], other=0.0).to(INPUT_DTYPE)
            b = tl.load(B_blk_ptrs, mask=R_mask[None, :], other=0.0).to(INPUT_DTYPE)
        else:
            N_mask_iter = (i * BLOCK_N + N_block) < N
            dy = tl.load(
                DY_blk_ptrs, mask=E_mask[:, None] & N_mask_iter[None, :], other=0.0
            ).to(INPUT_DTYPE)
            wt = tl.load(
                WT_blk_ptrs, mask=N_mask_iter[:, None] & K_mask[None, :], other=0.0
            ).to(INPUT_DTYPE)
            b = tl.load(
                B_blk_ptrs, mask=N_mask_iter[:, None] & R_mask[None, :], other=0.0
            ).to(INPUT_DTYPE)

        # Base: acc += DY @ W^T  ([M, N] @ [N, K] -> [M, K])
        acc += tl.dot(dy, wt, allow_tf32=allow_tf32).to(tl.float32)

        # LoRA: dy_b_acc += DY @ B  ([M, N] @ [N, R] -> [M, R])
        dy_b_acc += tl.dot(dy, b, allow_tf32=allow_tf32).to(tl.float32)

        DY_blk_ptrs += BLOCK_N * stride_dyn
        WT_blk_ptrs += BLOCK_N * stride_wn
        B_blk_ptrs += BLOCK_N * stride_bn

    # Epilogue: load A[e] and compute (DY @ B) @ A
    # A[e] is A[e*R:(e+1)*R, :], shape [R, K]. Load [BLOCK_R, BLOCK_K].
    A_expert_offset = E_idx * ACTUAL_R
    A_blk_ptrs = (
        A_ptr
        + (A_expert_offset + R_block)[:, None] * stride_ar
        + K_block[None, :] * stride_ak
    )
    a_e = tl.load(A_blk_ptrs, mask=R_mask[:, None] & K_mask[None, :], other=0.0)

    # Cast to float32 for precision
    a_f32 = a_e.to(tl.float32)

    # (DY @ B) @ A: [M, R] @ [R, K] -> [M, K]
    lora_dx = tl.dot(dy_b_acc, a_f32, allow_tf32=allow_tf32)

    acc += scaling * lora_dx
    return acc


def _scatter2scatter_lora_dX_configs():
    """Generate backward dX kernel autotune configs.

    The inner loop is over N (not K as in forward). The output dimension is K.
    So BLOCK_K tiles the output and BLOCK_N tiles the reduction.

    Search space includes smaller tile sizes and fewer pipeline stages to
    support GPUs with limited shared memory (e.g. ~99KB on some GPUs).

    Search space:
      BLOCK_K:    {32, 64, 128, 256}   (output tile)
      BLOCK_N:    {32, 64, 128, 256}   (reduction tile)
      num_warps:  {4, 8}
      num_stages: {3, 4, 5}
    """
    configs = []
    for block_k, block_n, warps, stages in product(
        [32, 64, 128, 256],  # BLOCK_K (output dimension)
        [32, 64, 128, 256],  # BLOCK_N (reduction dimension)
        [4, 8],  # num_warps
        [3, 4, 5],  # num_stages
    ):
        configs.append(
            triton.Config(
                {"BLOCK_K": block_k, "BLOCK_N": block_n},
                num_stages=stages,
                num_warps=warps,
            )
        )
    return configs


def _prune_dX_configs(configs, named_args, **kwargs):
    """Prune backward dX configs based on SMEM capacity.

    The dX kernel inner loop loads three tiles per pipeline stage:
      DY[BLOCK_M, BLOCK_N], W^T[BLOCK_N, BLOCK_K], B[BLOCK_N, BLOCK_R].
    The base estimate only accounts for DY and W^T. We add:
      - B tile [BLOCK_N, BLOCK_R] per pipeline stage (loaded in the inner loop)
      - A tile [BLOCK_R, BLOCK_K] loaded once in the epilogue
      - Extra headroom for compiler overhead (register spills, metadata)
    """
    smem_cap = _get_smem_capacity()

    # Get BLOCK_R from named_args if available, else assume worst case
    block_r = named_args.get("BLOCK_R", 64)

    scored = []
    for config in configs:
        block_k = config.kwargs["BLOCK_K"]
        block_n = config.kwargs["BLOCK_N"]
        # Base: stages * BLOCK_N * (BLOCK_M + BLOCK_K) + BLOCK_M * BLOCK_K
        smem_base = _estimate_smem_usage(config.num_stages, BLOCK_M, block_k, block_n)
        # B tile [BLOCK_N, BLOCK_R] loaded per stage in the inner loop
        smem_lora_loop = config.num_stages * block_n * block_r * 2
        # A tile [BLOCK_R, BLOCK_K] loaded once in epilogue
        smem_lora_epilogue = block_r * block_k * 2
        smem = smem_base + smem_lora_loop + smem_lora_epilogue
        scored.append((smem, config))

    pruned = [c for s, c in scored if s <= smem_cap - _SMEM_SLACK]
    if pruned:
        return pruned
    # All configs exceed SMEM — return the one with smallest estimated usage
    scored.sort(key=lambda x: x[0])
    return [scored[0][1]]


@triton.autotune(
    configs=_scatter2scatter_lora_dX_configs(),
    key=["M", "N", "K"],
    prune_configs_by={"early_config_prune": _prune_dX_configs},
)
@triton.heuristics(
    {
        "NO_K_MASK": lambda args: (args["K"] % args["BLOCK_K"]) == 0,
        "NO_N_MASK": lambda args: (args["N"] % args["BLOCK_N"]) == 0,
    }
)
@triton.jit
def _scatter2scatter_lora_dX(
    # Input: DY (gradient w.r.t. output, grouped)
    DY_ptr,
    stride_dym: tl.constexpr,
    stride_dyn: tl.constexpr,
    # Base weight: W [E, K, N] (we compute DY @ W^T)
    W_ptr,
    stride_we,
    stride_wk: tl.constexpr,
    stride_wn: tl.constexpr,
    # Output: dX
    DX_ptr,
    stride_dxm: tl.constexpr,
    stride_dxk: tl.constexpr,
    # LoRA weights
    LA_ptr,
    stride_la_r,
    stride_la_k,  # A: [r*E, K]
    LB_ptr,
    stride_lb_n,
    stride_lb_r,  # B: [N, r*E]
    # Routing
    grouped_idx_ptr,
    expert_idxs_ptr,
    # Dimensions
    FAN_OUT: tl.constexpr,
    M,
    K: tl.constexpr,
    N: tl.constexpr,
    E: tl.constexpr,
    ACTUAL_R: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_R: tl.constexpr,
    # Config
    ACC_TYPE: tl.constexpr,
    scaling,
    allow_tf32: tl.constexpr,
    dy_grouped: tl.constexpr,
    dx_grouped: tl.constexpr,
    NO_K_MASK: tl.constexpr,
    NO_N_MASK: tl.constexpr,
):
    """
    Fused backward dX = DY @ W^T + scaling * (DY @ B) @ A

    DY is in expert-grouped order (x_grouped=True).
    dX is output in ungrouped or grouped order based on dx_grouped.

    Grid: (cdiv(M_total, BLOCK_M) * cdiv(K, BLOCK_K),)
    """
    pid = tl.program_id(axis=0)

    K_BLOCK_COUNT = tl.cdiv(K, BLOCK_K)
    M_block_id = pid // K_BLOCK_COUNT
    K_block_id = pid % K_BLOCK_COUNT

    M_block = M_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    K_block = K_block_id * BLOCK_K + tl.arange(0, BLOCK_K)
    K_mask = K_block < K
    M_boundary_mask = M_block < (FAN_OUT * M)

    E_idxs = tl.load(expert_idxs_ptr + M_block, mask=M_boundary_mask, other=E)

    no_n_mask = NO_N_MASK

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=ACC_TYPE)

    E_first_idx = tl.min(E_idxs)
    E_last_idx = tl.minimum(tl.max(E_idxs), E - 1)
    M_idx = tl.load(grouped_idx_ptr + M_block, mask=M_boundary_mask).to(tl.int32)

    for E_idx in range(E_first_idx, E_last_idx + 1):
        E_mask = E_idxs == E_idx
        if dy_grouped:
            M_in_idx = M_block
        else:
            M_in_idx = M_idx // FAN_OUT

        acc = _compute_expert_block_lora_dX(
            E_idx,
            E_mask,
            M_in_idx,
            K_block,
            K_mask,
            DY_ptr,
            stride_dym,
            stride_dyn,
            W_ptr,
            stride_we,
            stride_wk,
            stride_wn,
            LA_ptr,
            stride_la_r,
            stride_la_k,
            LB_ptr,
            stride_lb_n,
            stride_lb_r,
            N,
            ACTUAL_R,
            acc,
            no_n_mask,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            BLOCK_R,
            scaling,
            allow_tf32=allow_tf32,
        )

    # Store output
    if dx_grouped:
        M_out_idx = M_block
    else:
        M_out_idx = M_idx
    DX_blk_ptrs = DX_ptr + (
        M_out_idx[:, None] * stride_dxm + K_block[None, :] * stride_dxk
    )
    tl.store(DX_blk_ptrs, acc, mask=M_boundary_mask[:, None] & K_mask[None, :])


def scatter2scatter_lora_dX(
    DY: torch.Tensor,
    W: torch.Tensor,
    sorted_expert_idxs: torch.Tensor,
    sorted_scattered_idxs: torch.Tensor,
    k: int,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    scaling: float,
    dy_grouped: bool = True,
    dx_grouped: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Fused backward dX = DY @ W^T + scaling * (DY @ B) @ A

    Replaces the separate:
      1. base_ops.scatter2scatter(DY, W^T, x_grouped=True, ...)
      2. _compute_lora_input_grad(DY, A, B, ...)

    Args:
        DY: Gradient w.r.t. output [M*k, N] (grouped by expert)
        W: Expert weights [E, K, N] (NOT transposed — kernel handles W^T internally)
        sorted_expert_idxs: Expert assignments sorted [M*k]
        sorted_scattered_idxs: Original indices sorted [M*k]
        k: Fan-out (top-k)
        lora_A: LoRA A weights [r*E, K]
        lora_B: LoRA B weights [N, r*E]
        scaling: LoRA scaling factor
        dy_grouped: Whether DY is in grouped (expert-sorted) order (default True)
        dx_grouped: Whether to output dX in grouped order (default False)
        out: Optional pre-allocated output buffer

    Returns:
        dX: Input gradient [M*k, K]
    """
    assert sorted_scattered_idxs.size(0) == sorted_expert_idxs.size(0)

    E = W.size(0)
    K = W.size(1)
    N = W.size(2)
    R = lora_A.size(0) // E

    BLOCK_R = _block_r_for_rank(R)

    L_scattered = sorted_expert_idxs.size(0)

    # M for the kernel is DY.size(0) when dy_grouped, else the original M
    if dy_grouped:
        M = DY.size(0)
        fan_out = 1  # DY is already expanded
    else:
        M = DY.size(0)
        fan_out = k

    if out is None:
        output = torch.empty((L_scattered, K), device=DY.device, dtype=DY.dtype)
    else:
        assert out.size(0) == L_scattered and out.size(1) == K
        output = out

    def grid(META):
        return (
            triton.cdiv(L_scattered, META["BLOCK_M"]) * triton.cdiv(K, META["BLOCK_K"]),
        )

    _scatter2scatter_lora_dX[grid](
        DY,
        DY.stride(0),
        DY.stride(1),
        W,
        W.stride(0),
        W.stride(1),
        W.stride(2),
        output,
        output.stride(0),
        output.stride(1),
        lora_A,
        lora_A.stride(0),
        lora_A.stride(1),
        lora_B,
        lora_B.stride(0),
        lora_B.stride(1),
        sorted_scattered_idxs,
        sorted_expert_idxs,
        FAN_OUT=fan_out,
        M=M,
        K=K,
        N=N,
        E=E,
        ACTUAL_R=R,
        BLOCK_M=BLOCK_M,
        BLOCK_R=BLOCK_R,
        ACC_TYPE=tl.float32,
        scaling=scaling,
        allow_tf32=ALLOW_TF32,
        dy_grouped=dy_grouped,
        dx_grouped=dx_grouped,
    )

    return output


# =============================================================================
# Backward Kernel: LoRA gradient computation (dA, dB)
# =============================================================================


def _group_bwd_lora_configs():
    """Generate backward (dA/dB) kernel autotune configs.

    Search space includes smaller tile sizes and fewer pipeline stages to
    support GPUs with limited shared memory (e.g. ~99KB on some GPUs).

    Search space:
      BLOCK_M:    {32, 64, 128, 256}   (token-loop tile)
      BLOCK_K:    {32, 64, 128, 256}
      BLOCK_N:    {32, 64, 128, 256}
      num_warps:  {4, 8}
      num_stages: {3, 4, 5}

    The backward kernel also uses BLOCK_R (from LoRA rank), but that is
    determined by the rank and not autotunable.
    """
    configs = []
    for block_m, block_k, block_n, warps, stages in product(
        [32, 64, 128, 256],  # BLOCK_M
        [32, 64, 128, 256],  # BLOCK_K
        [32, 64, 128, 256],  # BLOCK_N
        [4, 8],  # num_warps
        [3, 4, 5],  # num_stages
    ):
        configs.append(
            triton.Config(
                {"BLOCK_M": block_m, "BLOCK_K": block_k, "BLOCK_N": block_n},
                num_stages=stages,
                num_warps=warps,
            )
        )
    return configs


def _prune_bwd_lora_configs(configs, named_args, **kwargs):
    """Prune backward configs based on SMEM capacity.

    The backward kernel loads X[BLOCK_M, BLOCK_K] and DY[BLOCK_M, BLOCK_N]
    in the inner loop, plus holds A[BLOCK_R, BLOCK_K] and B[BLOCK_N, BLOCK_R]
    for the full expert. We estimate SMEM based on the dominant terms.
    """
    smem_cap = _get_smem_capacity()
    block_r = named_args.get("BLOCK_R", 64)

    scored = []
    for config in configs:
        block_m = config.kwargs["BLOCK_M"]
        block_k = config.kwargs["BLOCK_K"]
        block_n = config.kwargs["BLOCK_N"]
        # Inner loop loads X[M,K] and DY[M,N], pipeline over M iterations
        smem_base = _estimate_smem_usage(config.num_stages, block_m, block_n, block_k)
        # A[BLOCK_R, BLOCK_K] and B[BLOCK_N, BLOCK_R] held for the full expert
        smem_lora = (block_r * block_k + block_n * block_r) * 2
        smem = smem_base + smem_lora
        scored.append((smem, config))

    pruned = [c for s, c in scored if s <= smem_cap - _SMEM_SLACK]
    if pruned:
        return pruned
    # All configs exceed SMEM — return the one with smallest estimated usage
    scored.sort(key=lambda x: x[0])
    return [scored[0][1]]


@triton.autotune(
    configs=_group_bwd_lora_configs(),
    key=["M", "N", "K"],
    prune_configs_by={"early_config_prune": _prune_bwd_lora_configs},
    reset_to_zero=["DLA_ptr", "DLB_ptr"],
)
@triton.heuristics(
    {
        "NO_K_MASK": lambda args: (args["K"] % args["BLOCK_K"]) == 0,
        "NO_N_MASK": lambda args: (args["N"] % args["BLOCK_N"]) == 0,
    }
)
@triton.jit
def _group_bwd_lora(
    # Inputs
    DY_ptr,
    stride_dym,
    stride_dyn,
    X_ptr,
    stride_xm,
    stride_xk,
    # LoRA weights (needed for cross-terms)
    LA_ptr,
    stride_la_r,
    stride_la_k,  # A: [r*E, K]
    LB_ptr,
    stride_lb_n,
    stride_lb_r,  # B: [N, r*E]
    # Gradient outputs
    DLA_ptr,
    stride_dla_r,
    stride_dla_k,
    DLB_ptr,
    stride_dlb_n,
    stride_dlb_r,
    # Expert offsets
    expert_offsets_ptr,
    # Dimensions
    M,
    K: tl.constexpr,
    N: tl.constexpr,
    ACTUAL_R: tl.constexpr,  # True LoRA rank (for weight indexing)
    BLOCK_R: tl.constexpr,  # Padded tile size >= max(ACTUAL_R, 16)
    scaling,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ACC_TYPE: tl.constexpr,
    allow_tf32: tl.constexpr,
    NO_K_MASK: tl.constexpr,
    NO_N_MASK: tl.constexpr,
):
    """
    Compute LoRA gradients for each expert on grouped data.

    Grid: (E * cdiv(K, BLOCK_K), cdiv(N, BLOCK_N))

    For expert e:
      dA[e] = scaling * (dY @ B[e])^T @ X   -> [r, K], accumulate over M tokens
      dB[e] = scaling * dY^T @ (X @ A[e]^T)  -> [N, r], accumulate over M tokens

    ACTUAL_R is the true LoRA rank. BLOCK_R >= ACTUAL_R is padded for tl.dot min size.
    """
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)

    K_BLOCK_COUNT = tl.cdiv(K, BLOCK_K)
    E_idx = pid0 // K_BLOCK_COUNT
    K_block_id = pid0 % K_BLOCK_COUNT
    N_block_id = pid1

    # Get expert's token range from cumulative offsets
    if E_idx == 0:
        start_idx = 0
    else:
        start_idx = tl.load(expert_offsets_ptr + E_idx - 1).to(tl.int32)
    end_idx = tl.load(expert_offsets_ptr + E_idx).to(tl.int32)
    num_tokens = end_idx - start_idx

    if num_tokens > 0:
        M_block = tl.arange(0, BLOCK_M)
        K_block = K_block_id * BLOCK_K + tl.arange(0, BLOCK_K)
        K_mask = K_block < K
        N_block = N_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
        N_mask = N_block < N
        R_block = tl.arange(0, BLOCK_R)
        R_mask = R_block < ACTUAL_R  # Mask for padding

        lora_offset = E_idx * ACTUAL_R

        # Determine input element type for consistent casting.
        INPUT_DTYPE = X_ptr.dtype.element_ty

        # Load B[e]: [BLOCK_N, BLOCK_R] (masked on R and N, other=0 for padding)
        B_blk_ptrs = (
            LB_ptr
            + N_block[:, None] * stride_lb_n
            + (lora_offset + R_block)[None, :] * stride_lb_r
        )
        b_e = tl.load(B_blk_ptrs, mask=N_mask[:, None] & R_mask[None, :], other=0.0).to(
            INPUT_DTYPE
        )

        # Load A[e]: [BLOCK_R, BLOCK_K] (masked on R and K, other=0 for padding)
        A_blk_ptrs = (
            LA_ptr
            + (lora_offset + R_block)[:, None] * stride_la_r
            + K_block[None, :] * stride_la_k
        )
        a_e = tl.load(A_blk_ptrs, mask=R_mask[:, None] & K_mask[None, :], other=0.0).to(
            INPUT_DTYPE
        )

        # Accumulators
        dA_acc = tl.zeros((BLOCK_R, BLOCK_K), dtype=ACC_TYPE)
        dB_acc = tl.zeros((BLOCK_N, BLOCK_R), dtype=ACC_TYPE)

        iters = tl.cdiv(num_tokens, BLOCK_M)
        for i in range(iters):
            M_idx = start_idx + i * BLOCK_M + M_block
            M_mask = M_idx < end_idx

            # Load X: [BLOCK_M, BLOCK_K]
            X_blk_ptrs = (
                X_ptr + M_idx[:, None] * stride_xm + K_block[None, :] * stride_xk
            )
            x = tl.load(
                X_blk_ptrs, mask=M_mask[:, None] & K_mask[None, :], other=0.0
            ).to(INPUT_DTYPE)

            # Load dY: [BLOCK_M, BLOCK_N]
            DY_blk_ptrs = (
                DY_ptr + M_idx[:, None] * stride_dym + N_block[None, :] * stride_dyn
            )
            dy = tl.load(
                DY_blk_ptrs, mask=M_mask[:, None] & N_mask[None, :], other=0.0
            ).to(INPUT_DTYPE)

            # X @ A[e]^T: [M, K] @ [K, R] -> [M, R]
            xa = tl.dot(x, tl.trans(a_e), allow_tf32=allow_tf32)

            # dY @ B[e]: [M, N] @ [N, R] -> [M, R]
            dy_b = tl.dot(dy, b_e, allow_tf32=allow_tf32)

            # Cast intermediates to input dtype for subsequent tl.dot calls
            # (tl.dot requires both operands to have the same dtype)
            dy_b_cast = dy_b.to(INPUT_DTYPE)
            xa_cast = xa.to(INPUT_DTYPE)

            # dA += (dY @ B)^T @ X: [R, M] @ [M, K] -> [R, K]
            dA_acc += tl.dot(tl.trans(dy_b_cast), x, allow_tf32=allow_tf32)

            # dB += dY^T @ (X @ A^T): [N, M] @ [M, R] -> [N, R]
            dB_acc += tl.dot(tl.trans(dy), xa_cast, allow_tf32=allow_tf32)

        # Store dA with scaling (atomic add since multiple N_blocks contribute)
        # Only store the actual R rows, not the padded ones
        DLA_blk_ptrs = (
            DLA_ptr
            + (lora_offset + R_block)[:, None] * stride_dla_r
            + K_block[None, :] * stride_dla_k
        )
        tl.atomic_add(
            DLA_blk_ptrs,
            (dA_acc * scaling).to(DLA_ptr.dtype.element_ty),
            mask=R_mask[:, None] & K_mask[None, :],
        )

        # Store dB with scaling (atomic add since multiple K_blocks contribute)
        DLB_blk_ptrs = (
            DLB_ptr
            + N_block[:, None] * stride_dlb_n
            + (lora_offset + R_block)[None, :] * stride_dlb_r
        )
        tl.atomic_add(
            DLB_blk_ptrs,
            (dB_acc * scaling).to(DLB_ptr.dtype.element_ty),
            mask=N_mask[:, None] & R_mask[None, :],
        )


def group_bwd_lora(
    DY: torch.Tensor,
    X: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    expert_offsets: torch.Tensor,
    E: int,
    scaling: float,
    sorted_scattered_idxs: Optional[torch.Tensor] = None,
    k: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute LoRA gradients for A and B on expert-grouped data.

    Args:
        DY: Gradient w.r.t. output [M_total, N] (grouped by expert)
        X: Input [M_total, K] (grouped by expert)
        lora_A: LoRA A weights [r*E, K]
        lora_B: LoRA B weights [N, r*E]
        expert_offsets: Cumulative token counts per expert [E]
        E: Number of experts
        scaling: LoRA scaling factor

    Returns:
        dA: Gradient for A [r*E, K]
        dB: Gradient for B [N, r*E]
    """
    R = lora_A.size(0) // E
    K = X.size(1)
    N = DY.size(1)

    # Zero-init for atomic accumulation
    dA = torch.zeros_like(lora_A)
    dB = torch.zeros_like(lora_B)

    BLOCK_R = _block_r_for_rank(R)

    def grid(META):
        return (
            E * triton.cdiv(K, META["BLOCK_K"]),
            triton.cdiv(N, META["BLOCK_N"]),
        )

    _group_bwd_lora[grid](
        DY,
        DY.stride(0),
        DY.stride(1),
        X,
        X.stride(0),
        X.stride(1),
        lora_A,
        lora_A.stride(0),
        lora_A.stride(1),
        lora_B,
        lora_B.stride(0),
        lora_B.stride(1),
        dA,
        dA.stride(0),
        dA.stride(1),
        dB,
        dB.stride(0),
        dB.stride(1),
        expert_offsets,
        M=DY.size(0),
        K=K,
        N=N,
        ACTUAL_R=R,  # True LoRA rank
        BLOCK_R=BLOCK_R,  # Padded tile size
        scaling=scaling,
        ACC_TYPE=tl.float32,
        allow_tf32=ALLOW_TF32,
    )

    return dA, dB


# =============================================================================
# Backward Kernel: Fused gather + LoRA gradient (dA, dB) — eliminates group()
# =============================================================================


@triton.autotune(
    configs=_group_bwd_lora_configs(),
    key=["M", "N", "K"],
    prune_configs_by={"early_config_prune": _prune_bwd_lora_configs},
    reset_to_zero=["DLA_ptr", "DLB_ptr"],
)
@triton.heuristics(
    {
        "NO_K_MASK": lambda args: (args["K"] % args["BLOCK_K"]) == 0,
        "NO_N_MASK": lambda args: (args["N"] % args["BLOCK_N"]) == 0,
    }
)
@triton.jit
def _group_bwd_lora_fused(
    # Inputs (ungrouped or grouped)
    DY_ptr,
    stride_dym,
    stride_dyn,
    X_ptr,
    stride_xm,
    stride_xk,
    # Scatter indices for gather-on-load
    sorted_scattered_idxs_ptr,
    FAN_OUT: tl.constexpr,
    # LoRA weights (needed for cross-terms)
    LA_ptr,
    stride_la_r,
    stride_la_k,  # A: [r*E, K]
    LB_ptr,
    stride_lb_n,
    stride_lb_r,  # B: [N, r*E]
    # Gradient outputs
    DLA_ptr,
    stride_dla_r,
    stride_dla_k,
    DLB_ptr,
    stride_dlb_n,
    stride_dlb_r,
    # Expert offsets
    expert_offsets_ptr,
    # Real expert offsets (for M_mask when using token rounding, else same as expert_offsets_ptr)
    real_expert_offsets_ptr,
    # Dimensions
    M,
    K: tl.constexpr,
    N: tl.constexpr,
    ACTUAL_R: tl.constexpr,
    BLOCK_R: tl.constexpr,
    scaling,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ACC_TYPE: tl.constexpr,
    allow_tf32: tl.constexpr,
    NO_K_MASK: tl.constexpr,
    NO_N_MASK: tl.constexpr,
    # Whether DY is already in grouped (expert-sorted) order
    dy_grouped: tl.constexpr = False,
):
    """
    Fused gather + LoRA gradient computation. Same as _group_bwd_lora but
    reads X from ungrouped buffers using sorted_scattered_idxs for indirect
    indexing, eliminating the need for a separate group(X) call.

    When dy_grouped=False (default): both X and DY are read via indirect
    indexing through sorted_scattered_idxs.  This eliminates both group()
    calls entirely.

    When dy_grouped=True: DY is already in grouped order (e.g. gate_up_proj
    backward where grouped_out=True) and is read directly.  Only X uses
    indirect indexing.  This avoids the group(X) allocation while
    still supporting the grouped DY case.

    Grid: (E * cdiv(K, BLOCK_K), cdiv(N, BLOCK_N))

    For expert e:
      dA[e] = scaling * (dY @ B[e])^T @ X   -> [r, K]
      dB[e] = scaling * dY^T @ (X @ A[e]^T)  -> [N, r]

    Supports token rounding: expert_offsets_ptr gives the iteration range
    (padded to BLOCK_M multiples), real_expert_offsets_ptr gives the real
    token count for M_mask (to exclude padding tokens).
    """
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)

    K_BLOCK_COUNT = tl.cdiv(K, BLOCK_K)
    E_idx = pid0 // K_BLOCK_COUNT
    K_block_id = pid0 % K_BLOCK_COUNT
    N_block_id = pid1

    # Get expert's token range from cumulative offsets
    # start_idx/end_idx from expert_offsets_ptr: iteration range (possibly padded)
    # real_end_idx from real_expert_offsets_ptr: for M_mask (real token count)
    if E_idx == 0:
        start_idx = 0
        real_start_idx = 0
    else:
        start_idx = tl.load(expert_offsets_ptr + E_idx - 1).to(tl.int32)
        real_start_idx = tl.load(real_expert_offsets_ptr + E_idx - 1).to(tl.int32)
    end_idx = tl.load(expert_offsets_ptr + E_idx).to(tl.int32)
    real_end_idx = tl.load(real_expert_offsets_ptr + E_idx).to(tl.int32)
    num_tokens = end_idx - start_idx

    if num_tokens > 0:
        M_block = tl.arange(0, BLOCK_M)
        K_block = K_block_id * BLOCK_K + tl.arange(0, BLOCK_K)
        K_mask = K_block < K
        N_block = N_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
        N_mask = N_block < N
        R_block = tl.arange(0, BLOCK_R)
        R_mask = R_block < ACTUAL_R

        lora_offset = E_idx * ACTUAL_R

        # Determine input element type for consistent casting.
        INPUT_DTYPE = X_ptr.dtype.element_ty

        # Load B[e] and A[e] — same as non-fused kernel
        B_blk_ptrs = (
            LB_ptr
            + N_block[:, None] * stride_lb_n
            + (lora_offset + R_block)[None, :] * stride_lb_r
        )
        b_e = tl.load(B_blk_ptrs, mask=N_mask[:, None] & R_mask[None, :], other=0.0).to(
            INPUT_DTYPE
        )

        A_blk_ptrs = (
            LA_ptr
            + (lora_offset + R_block)[:, None] * stride_la_r
            + K_block[None, :] * stride_la_k
        )
        a_e = tl.load(A_blk_ptrs, mask=R_mask[:, None] & K_mask[None, :], other=0.0).to(
            INPUT_DTYPE
        )

        # Accumulators
        dA_acc = tl.zeros((BLOCK_R, BLOCK_K), dtype=ACC_TYPE)
        dB_acc = tl.zeros((BLOCK_N, BLOCK_R), dtype=ACC_TYPE)

        real_num_tokens = real_end_idx - real_start_idx
        iters = tl.cdiv(num_tokens, BLOCK_M)
        for i in range(iters):
            M_idx = start_idx + i * BLOCK_M + M_block
            # Use real token count for masking (excludes padding tokens)
            M_local = i * BLOCK_M + M_block
            M_mask = M_local < real_num_tokens

            # Fused gather: load scatter indices for indirect X access
            scatter_idx = tl.load(
                sorted_scattered_idxs_ptr + M_idx, mask=M_mask, other=0
            ).to(tl.int32)
            X_token_idx = scatter_idx // FAN_OUT  # X is [M, K], not expanded by k

            # Load X via indirect index: [BLOCK_M, BLOCK_K]
            X_blk_ptrs = (
                X_ptr + X_token_idx[:, None] * stride_xm + K_block[None, :] * stride_xk
            )
            x = tl.load(
                X_blk_ptrs, mask=M_mask[:, None] & K_mask[None, :], other=0.0
            ).to(INPUT_DTYPE)

            # Load DY: indirect via scatter_idx when ungrouped, direct via M_idx when grouped
            if dy_grouped:
                DY_blk_ptrs = (
                    DY_ptr + M_idx[:, None] * stride_dym + N_block[None, :] * stride_dyn
                )
            else:
                DY_blk_ptrs = (
                    DY_ptr
                    + scatter_idx[:, None] * stride_dym
                    + N_block[None, :] * stride_dyn
                )
            dy = tl.load(
                DY_blk_ptrs, mask=M_mask[:, None] & N_mask[None, :], other=0.0
            ).to(INPUT_DTYPE)

            # X @ A[e]^T: [M, K] @ [K, R] -> [M, R]
            xa = tl.dot(x, tl.trans(a_e), allow_tf32=allow_tf32)

            # dY @ B[e]: [M, N] @ [N, R] -> [M, R]
            dy_b = tl.dot(dy, b_e, allow_tf32=allow_tf32)

            dy_b_cast = dy_b.to(INPUT_DTYPE)
            xa_cast = xa.to(INPUT_DTYPE)

            # dA += (dY @ B)^T @ X: [R, M] @ [M, K] -> [R, K]
            dA_acc += tl.dot(tl.trans(dy_b_cast), x, allow_tf32=allow_tf32)

            # dB += dY^T @ (X @ A^T): [N, M] @ [M, R] -> [N, R]
            dB_acc += tl.dot(tl.trans(dy), xa_cast, allow_tf32=allow_tf32)

        # Store dA with scaling (atomic add since multiple N_blocks contribute)
        DLA_blk_ptrs = (
            DLA_ptr
            + (lora_offset + R_block)[:, None] * stride_dla_r
            + K_block[None, :] * stride_dla_k
        )
        tl.atomic_add(
            DLA_blk_ptrs,
            (dA_acc * scaling).to(DLA_ptr.dtype.element_ty),
            mask=R_mask[:, None] & K_mask[None, :],
        )

        # Store dB with scaling (atomic add since multiple K_blocks contribute)
        DLB_blk_ptrs = (
            DLB_ptr
            + N_block[:, None] * stride_dlb_n
            + (lora_offset + R_block)[None, :] * stride_dlb_r
        )
        tl.atomic_add(
            DLB_blk_ptrs,
            (dB_acc * scaling).to(DLB_ptr.dtype.element_ty),
            mask=N_mask[:, None] & R_mask[None, :],
        )


def group_bwd_lora_fused(
    DY: torch.Tensor,
    X: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    expert_offsets: torch.Tensor,
    sorted_scattered_idxs: torch.Tensor,
    E: int,
    k: int,
    scaling: float,
    real_expert_offsets: Optional[torch.Tensor] = None,
    dy_grouped: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused gather + LoRA gradient computation. Same result as
    group(X) + group(DY) + group_bwd_lora(DY, X, ...) but without
    the intermediate grouped buffers.

    Args:
        DY: Gradient w.r.t. output [M*k, N].
            If dy_grouped=False: ungrouped (original token order), read via
            indirect indexing through sorted_scattered_idxs.
            If dy_grouped=True: already in grouped (expert-sorted) order,
            read directly.
        X: Input [M, K] (ungrouped, original token order).  Always read via
            indirect indexing through sorted_scattered_idxs.
        lora_A: LoRA A weights [r*E, K]
        lora_B: LoRA B weights [N, r*E]
        expert_offsets: Cumulative token counts per expert [E]
            (or padded offsets if using token rounding)
        sorted_scattered_idxs: Maps grouped position -> original position [M*k]
            (or padded version if using token rounding)
        E: Number of experts
        k: Fan-out (top-k)
        scaling: LoRA scaling factor
        real_expert_offsets: Original cumulative counts for M_mask when using
            token rounding. If None, expert_offsets is used for both.
        dy_grouped: Whether DY is already in grouped order (default False).
            When True, avoids indirect indexing for DY, used for gate_up_proj
            backward where grouped_out=True.

    Returns:
        dA: Gradient for A [r*E, K]
        dB: Gradient for B [N, r*E]
    """
    R = lora_A.size(0) // E
    K = X.size(1)
    N = DY.size(1)

    # Zero-init for atomic accumulation
    dA = torch.zeros_like(lora_A)
    dB = torch.zeros_like(lora_B)

    BLOCK_R = _block_r_for_rank(R)

    if real_expert_offsets is None:
        real_expert_offsets = expert_offsets

    def grid(META):
        return (
            E * triton.cdiv(K, META["BLOCK_K"]),
            triton.cdiv(N, META["BLOCK_N"]),
        )

    _group_bwd_lora_fused[grid](
        DY,
        DY.stride(0),
        DY.stride(1),
        X,
        X.stride(0),
        X.stride(1),
        sorted_scattered_idxs,
        FAN_OUT=k,
        LA_ptr=lora_A,
        stride_la_r=lora_A.stride(0),
        stride_la_k=lora_A.stride(1),
        LB_ptr=lora_B,
        stride_lb_n=lora_B.stride(0),
        stride_lb_r=lora_B.stride(1),
        DLA_ptr=dA,
        stride_dla_r=dA.stride(0),
        stride_dla_k=dA.stride(1),
        DLB_ptr=dB,
        stride_dlb_n=dB.stride(0),
        stride_dlb_r=dB.stride(1),
        expert_offsets_ptr=expert_offsets,
        real_expert_offsets_ptr=real_expert_offsets,
        M=sorted_scattered_idxs.size(0),
        K=K,
        N=N,
        ACTUAL_R=R,
        BLOCK_R=BLOCK_R,
        scaling=scaling,
        ACC_TYPE=tl.float32,
        allow_tf32=ALLOW_TF32,
        dy_grouped=dy_grouped,
    )

    return dA, dB
