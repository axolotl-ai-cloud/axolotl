"""Vendored backward pass for Triton contiguous grouped GEMM."""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import triton
import triton.language as tl

from .cg_forward import cg_grouped_gemm_forward
from .tma_cuda_autotune import STANDARD_CONFIGS, early_config_prune

GROUP_SIZE_M = 128


@triton.autotune(
    configs=STANDARD_CONFIGS,
    key=["M_TOTAL", "N", "K"],
    prune_configs_by={"early_config_prune": early_config_prune},
)
@triton.jit
def _kernel_cg_backward_dx(
    grad_output_ptr,
    b_ptr,
    grad_input_ptr,
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
    """Compute gradients with respect to inputs."""

    pid = tl.program_id(0)

    num_m_tiles = tl.cdiv(M_TOTAL, BLOCK_SIZE_M)
    num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

    tile_m = pid // num_k_tiles
    tile_k = pid % num_k_tiles

    m_start = tile_m * BLOCK_SIZE_M
    k_start = tile_k * BLOCK_SIZE_K

    if m_start < M_TOTAL:
        offs_m = tl.arange(0, BLOCK_SIZE_M) + m_start
        offs_k = tl.arange(0, BLOCK_SIZE_K) + k_start

        mask_m = offs_m < M_TOTAL
        mask_k = offs_k < K

        group_idx = m_start // GROUP_SIZE_M
        expert_idx = tl.load(indices_ptr + group_idx * GROUP_SIZE_M)

        grad_input = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_K], dtype=tl.float32)

        for n in range(0, N, BLOCK_SIZE_N):
            offs_n = tl.arange(0, BLOCK_SIZE_N) + n
            mask_n = offs_n < N

            mask_go = mask_m[:, None] & mask_n[None, :]
            mask_w = mask_n[:, None] & mask_k[None, :]

            go_ptrs = grad_output_ptr + offs_m[:, None] * N + offs_n[None, :]
            go = tl.load(go_ptrs, mask=mask_go, other=0.0).to(tl.float32)

            w_ptrs = b_ptr + expert_idx * N * K + offs_n[:, None] * K + offs_k[None, :]
            w = tl.load(w_ptrs, mask=mask_w, other=0.0).to(tl.float32)

            grad_input += tl.dot(go, w)

        grad_input_ptrs = grad_input_ptr + offs_m[:, None] * K + offs_k[None, :]
        mask_gi = mask_m[:, None] & mask_k[None, :]
        tl.store(grad_input_ptrs, grad_input, mask=mask_gi)


@triton.jit
def _kernel_cg_backward_dw(
    grad_output_ptr,
    inputs_ptr,
    grad_weights_ptr,
    indices_ptr,
    M_TOTAL: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    """Simplified kernel for expert weight gradients."""

    pid = tl.program_id(0)

    expert_id = pid // ((N * K) // (BLOCK_SIZE_N * BLOCK_SIZE_K))
    position_id = pid % ((N * K) // (BLOCK_SIZE_N * BLOCK_SIZE_K))

    if expert_id < NUM_EXPERTS:
        n_tiles = K // BLOCK_SIZE_K
        tile_n = position_id // n_tiles
        tile_k = position_id % n_tiles

        n_start = tile_n * BLOCK_SIZE_N
        k_start = tile_k * BLOCK_SIZE_K

        if n_start < N and k_start < K:
            offs_n = tl.arange(0, BLOCK_SIZE_N) + n_start
            offs_k = tl.arange(0, BLOCK_SIZE_K) + k_start

            mask_n = offs_n < N
            mask_k = offs_k < K

            grad_weights = tl.zeros([BLOCK_SIZE_N, BLOCK_SIZE_K], dtype=tl.float32)

            for group_idx in range(0, M_TOTAL // GROUP_SIZE_M):
                group_start = group_idx * GROUP_SIZE_M
                group_expert = tl.load(indices_ptr + group_start)

                if group_expert == expert_id:
                    for m_offset in range(0, GROUP_SIZE_M, BLOCK_SIZE_M):
                        m_start = group_start + m_offset
                        offs_m = tl.arange(0, BLOCK_SIZE_M) + m_start

                        mask_m = offs_m < min(group_start + GROUP_SIZE_M, M_TOTAL)

                        go_ptrs = (
                            grad_output_ptr + offs_m[:, None] * N + offs_n[None, :]
                        )
                        mask_go = mask_m[:, None] & mask_n[None, :]
                        go = tl.load(go_ptrs, mask=mask_go, other=0.0)

                        in_ptrs = inputs_ptr + offs_m[:, None] * K + offs_k[None, :]
                        mask_in = mask_m[:, None] & mask_k[None, :]
                        inp = tl.load(in_ptrs, mask=mask_in, other=0.0)

                        go_t = tl.trans(go)
                        grad_weights += tl.dot(go_t, inp)

            grad_w_ptrs = (
                grad_weights_ptr
                + expert_id * N * K
                + offs_n[:, None] * K
                + offs_k[None, :]
            )
            mask_gw = mask_n[:, None] & mask_k[None, :]
            tl.store(grad_w_ptrs, grad_weights, mask=mask_gw)


def cg_grouped_gemm_backward_weights(
    grad_output: torch.Tensor,
    inputs: torch.Tensor,
    expert_indices: torch.Tensor,
    num_experts: int,
    group_size_m: int = GROUP_SIZE_M,
) -> torch.Tensor:
    """Backward pass for expert weights."""

    assert grad_output.is_contiguous(), "Grad output tensor must be contiguous"
    assert inputs.is_contiguous(), "Inputs tensor must be contiguous"
    assert expert_indices.is_contiguous(), "Expert indices tensor must be contiguous"

    M_total, N = grad_output.shape
    _, K = inputs.shape

    if expert_indices.dtype != torch.int32:
        expert_indices = expert_indices.to(torch.int32)

    grad_weights = torch.zeros(
        (num_experts, N, K), device=grad_output.device, dtype=grad_output.dtype
    )

    block_size_n = min(128, N)
    block_size_k = min(32, K)
    block_size_m = min(32, group_size_m)

    n_tiles = triton.cdiv(N, block_size_n)
    k_tiles = triton.cdiv(K, block_size_k)
    grid = (num_experts * n_tiles * k_tiles,)

    _kernel_cg_backward_dw[grid](
        grad_output,
        inputs,
        grad_weights,
        expert_indices,
        M_TOTAL=M_total,
        N=N,
        K=K,
        NUM_EXPERTS=num_experts,
        GROUP_SIZE_M=group_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        BLOCK_SIZE_M=block_size_m,
    )

    return grad_weights


def cg_grouped_gemm_backward_inputs(
    grad_output: torch.Tensor,
    expert_weights: torch.Tensor,
    expert_indices: torch.Tensor,
    group_size_m: int = GROUP_SIZE_M,
) -> torch.Tensor:
    """Backward pass for inputs."""

    assert grad_output.is_contiguous(), "Grad output tensor must be contiguous"
    assert expert_weights.is_contiguous(), "Expert weights tensor must be contiguous"
    assert expert_indices.is_contiguous(), "Expert indices tensor must be contiguous"

    M_total, N = grad_output.shape
    num_experts, _, K = expert_weights.shape

    assert M_total % group_size_m == 0, (
        f"M_total ({M_total}) must be a multiple of group_size_m ({group_size_m})"
    )

    grad_inputs = torch.zeros(
        (M_total, K), device=grad_output.device, dtype=grad_output.dtype
    )

    grid = lambda meta: (
        triton.cdiv(M_total, meta["BLOCK_SIZE_M"])
        * triton.cdiv(K, meta["BLOCK_SIZE_K"]),
    )

    _kernel_cg_backward_dx[grid](
        grad_output,
        expert_weights,
        grad_inputs,
        expert_indices,
        M_TOTAL=M_total,
        N=N,
        K=K,
        NUM_EXPERTS=num_experts,
        GROUP_SIZE_M=group_size_m,
    )

    return grad_inputs


class ContiguousGroupedGEMM(torch.autograd.Function):
    """Autograd function with full backward support."""

    @staticmethod
    def forward(ctx, inputs, expert_weights, expert_indices, group_size_m=GROUP_SIZE_M):
        ctx.save_for_backward(inputs, expert_weights, expert_indices)
        ctx.group_size_m = group_size_m

        return cg_grouped_gemm_forward(
            inputs=inputs,
            expert_weights=expert_weights,
            expert_indices=expert_indices,
            group_size_m=group_size_m,
        )

    @staticmethod
    def backward(ctx, grad_output):
        inputs, expert_weights, expert_indices = ctx.saved_tensors
        group_size_m = ctx.group_size_m

        grad_output = grad_output.contiguous()
        num_experts = expert_weights.shape[0]

        grad_inputs = cg_grouped_gemm_backward_inputs(
            grad_output=grad_output,
            expert_weights=expert_weights,
            expert_indices=expert_indices,
            group_size_m=group_size_m,
        )

        grad_weights = cg_grouped_gemm_backward_weights(
            grad_output=grad_output,
            inputs=inputs,
            expert_indices=expert_indices,
            num_experts=num_experts,
            group_size_m=group_size_m,
        )

        grad_indices = None
        grad_group_size_m = None

        return grad_inputs, grad_weights, grad_indices, grad_group_size_m
