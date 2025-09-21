# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Vendored forward Triton contiguous grouped GEMM kernels."""

import torch
import triton
import triton.language as tl

from .tma_cuda_autotune import STANDARD_CONFIGS, early_config_prune

GROUP_SIZE_M = 128


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, super_group_m):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * super_group_m
    group_size_m = min(num_pid_m - first_pid_m, super_group_m)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.autotune(
    configs=STANDARD_CONFIGS,
    key=["M_TOTAL", "N", "K"],
    prune_configs_by={"early_config_prune": early_config_prune},
)
@triton.jit
def _kernel_cg_persistent_forward(
    a_ptr,
    b_ptr,
    c_ptr,
    indices_ptr,
    M_TOTAL: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr = GROUP_SIZE_M,
    SUPER_GROUP_M: tl.constexpr = 32,
):
    """
    Contiguous Grouped GEMM kernel forward (persistent variant).
    """

    c_type = c_ptr.dtype.element_ty

    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M_TOTAL, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n
    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = SUPER_GROUP_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS):
        tile_m_idx, tile_n_idx = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, SUPER_GROUP_M
        )

        m_start = tile_m_idx * BLOCK_SIZE_M
        n_start = tile_n_idx * BLOCK_SIZE_N

        if m_start < M_TOTAL:
            offs_m = m_start + tl.arange(0, BLOCK_SIZE_M)
            offs_n = n_start + tl.arange(0, BLOCK_SIZE_N)
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

            for ki in range(k_tiles):
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

                mask_m = offs_m < M_TOTAL
                mask_n = offs_n < N
                mask_k = offs_k < K

                mask_a = mask_m[:, None] & mask_k[None, :]
                mask_b = mask_n[:, None] & mask_k[None, :]

                group_idx = m_start // GROUP_SIZE_M
                expert_idx = tl.load(indices_ptr + group_idx * GROUP_SIZE_M)

                a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
                a = tl.load(a_ptrs, mask=mask_a, other=0.0)

                b_ptrs = (
                    b_ptr + expert_idx * N * K + offs_n[:, None] * K + offs_k[None, :]
                )
                b = tl.load(b_ptrs, mask=mask_b, other=0.0)

                accumulator += tl.dot(a, b.T)

            tile_id_c += NUM_SMS
            tile_m_idx, tile_n_idx = _compute_pid(
                tile_id_c, num_pid_in_group, num_pid_m, SUPER_GROUP_M
            )

            offs_m = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_n = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

            mask_m = offs_m < M_TOTAL
            mask_n = offs_n < N
            mask_c = mask_m[:, None] & mask_n[None, :]

            c = accumulator.to(tl.float32)
            c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
            tl.store(c_ptrs, c.to(c_type), mask=mask_c)


@triton.autotune(
    configs=STANDARD_CONFIGS,
    key=["M_TOTAL", "N", "K"],
    prune_configs_by={"early_config_prune": early_config_prune},
)
@triton.jit
def _kernel_cg_forward_aligned(
    a_ptr,
    b_ptr,
    c_ptr,
    indices_ptr,
    M_TOTAL: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr = GROUP_SIZE_M,
):
    """
    Contiguous Grouped GEMM kernel forward for aligned inputs.
    """

    pid = tl.program_id(0)

    c_type = c_ptr.dtype.element_ty

    num_m_tiles = tl.cdiv(M_TOTAL, BLOCK_SIZE_M)
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)

    tile_m = pid // num_n_tiles
    tile_n = pid % num_n_tiles

    m_start = tile_m * BLOCK_SIZE_M
    n_start = tile_n * BLOCK_SIZE_N

    if m_start < M_TOTAL:
        offs_m = tl.arange(0, BLOCK_SIZE_M) + m_start
        offs_n = tl.arange(0, BLOCK_SIZE_N) + n_start

        mask_m = offs_m < M_TOTAL
        mask_n = offs_n < N

        group_idx = m_start // GROUP_SIZE_M
        expert_idx = tl.load(indices_ptr + group_idx * GROUP_SIZE_M)

        acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

        for k in range(0, K, BLOCK_SIZE_K):
            offs_k = tl.arange(0, BLOCK_SIZE_K) + k
            mask_k = offs_k < K

            mask_a = mask_m[:, None] & mask_k[None, :]
            mask_b = mask_n[:, None] & mask_k[None, :]

            a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
            a = tl.load(a_ptrs, mask=mask_a, other=0.0)

            b_ptrs = b_ptr + expert_idx * N * K + offs_n[:, None] * K + offs_k[None, :]
            b = tl.load(b_ptrs, mask=mask_b, other=0.0)

            acc += tl.dot(a, b.T)

        c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
        mask_c = mask_m[:, None] & mask_n[None, :]
        tl.store(c_ptrs, acc.to(c_type), mask=mask_c)


def cg_grouped_gemm_forward(
    inputs: torch.Tensor,
    expert_weights: torch.Tensor,
    expert_indices: torch.Tensor,
    group_size_m: int = GROUP_SIZE_M,
) -> torch.Tensor:
    """Contiguous grouped GEMM forward pass for MoE."""

    assert inputs.is_contiguous(), "Input tensor must be contiguous"
    assert expert_weights.is_contiguous(), "Expert weights tensor must be contiguous"
    assert expert_indices.is_contiguous(), "Expert indices tensor must be contiguous"

    M_total, K = inputs.shape
    assert M_total % group_size_m == 0, (
        f"M_total ({M_total}) must be a multiple of group_size_m ({group_size_m})"
    )

    if expert_indices.dtype != torch.int32:
        expert_indices = expert_indices.to(torch.int32)

    num_experts, N, K_weights = expert_weights.shape
    assert K == K_weights, f"Input K ({K}) must match weight K ({K_weights})"
    assert expert_indices.shape[0] == M_total, (
        "Expert indices length must match M_total"
    )

    output = torch.empty((M_total, N), device=inputs.device, dtype=torch.bfloat16)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    grid = (NUM_SMS, 1, 1)
    _kernel_cg_persistent_forward[grid](
        inputs,
        expert_weights,
        output,
        expert_indices,
        M_TOTAL=M_total,
        N=N,
        K=K,
        NUM_EXPERTS=num_experts,
        GROUP_SIZE_M=group_size_m,
        NUM_SMS=NUM_SMS,
    )

    return output


def cg_grouped_gemm_forward_dynamic(
    inputs: torch.Tensor,
    expert_weights: torch.Tensor,
    expert_indices: torch.Tensor,
    group_size_m: int = GROUP_SIZE_M,
) -> torch.Tensor:
    """Contiguous grouped GEMM forward pass for MoE with autotuned launch."""

    assert inputs.is_contiguous(), "Input tensor must be contiguous"
    assert expert_weights.is_contiguous(), "Expert weights tensor must be contiguous"
    assert expert_indices.is_contiguous(), "Expert indices tensor must be contiguous"

    M_total, K = inputs.shape
    assert M_total % group_size_m == 0, (
        f"M_total ({M_total}) must be a multiple of group_size_m ({group_size_m})"
    )

    if expert_indices.dtype != torch.int32:
        expert_indices = expert_indices.to(torch.int32)

    num_experts, N, K_weights = expert_weights.shape
    assert K == K_weights, f"Input K ({K}) must match weight K ({K_weights})"
    assert expert_indices.shape[0] == M_total, (
        "Expert indices length must match M_total"
    )

    output = torch.empty((M_total, N), device=inputs.device, dtype=inputs.dtype)

    grid = lambda meta: (
        triton.cdiv(M_total, meta["BLOCK_SIZE_M"])
        * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )

    _kernel_cg_forward_aligned[grid](
        inputs,
        expert_weights,
        output,
        expert_indices,
        M_TOTAL=M_total,
        N=N,
        K=K,
        NUM_EXPERTS=num_experts,
        GROUP_SIZE_M=group_size_m,
    )

    return output


class ContiguousGroupedGEMM(torch.autograd.Function):
    """Autograd function for contiguous grouped GEMM forward pass only."""

    @staticmethod
    def forward(ctx, inputs, expert_weights, expert_indices, group_size_m=GROUP_SIZE_M):
        return cg_grouped_gemm_forward(
            inputs=inputs,
            expert_weights=expert_weights,
            expert_indices=expert_indices,
            group_size_m=group_size_m,
        )

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover - not implemented
        raise NotImplementedError("Backward pass not implemented")


def cg_grouped_gemm(
    inputs: torch.Tensor,
    expert_weights: torch.Tensor,
    expert_indices: torch.Tensor,
    group_size_m: int = GROUP_SIZE_M,
) -> torch.Tensor:
    """Convenience wrapper for the forward-only autograd function."""

    if expert_indices.dtype != torch.int32:
        expert_indices = expert_indices.to(torch.int32)

    return ContiguousGroupedGEMM.apply(
        inputs, expert_weights, expert_indices, group_size_m
    )
