"""Triton-based Contiguous Grouped GEMM Kernels for MoE."""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# Triton autotune configurations
STANDARD_CONFIGS = [
    triton.Config(
        {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
        },
        num_stages=3,
        num_warps=8,
    ),
    triton.Config(
        {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
        },
        num_stages=4,
        num_warps=4,
    ),
    triton.Config(
        {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
        },
        num_stages=4,
        num_warps=4,
    ),
    triton.Config(
        {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
        },
        num_stages=4,
        num_warps=4,
    ),
    triton.Config(
        {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
        },
        num_stages=4,
        num_warps=4,
    ),
    triton.Config(
        {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
        },
        num_stages=4,
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
        num_stages=5,
        num_warps=2,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
        num_stages=5,
        num_warps=2,
    ),
]


@triton.jit
def _kernel_cg_forward_aligned(
    a_ptr,
    b_ptr,
    c_ptr,
    expert_indices_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bge,
    stride_bgn,
    stride_bgk,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Triton kernel for contiguous grouped GEMM forward pass.
    Computes C = A @ B where B is selected based on expert indices.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Load expert index for this group
    expert_idx = tl.load(expert_indices_ptr + pid_m * BLOCK_SIZE_M)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = (
        b_ptr
        + expert_idx * stride_bge
        + (offs_k[:, None] * stride_bgk + offs_bn[None, :] * stride_bgn)
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bgk

    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.autotune(
    configs=STANDARD_CONFIGS,
    key=["M", "N", "K"],
)
@triton.jit
def _kernel_cg_forward(
    a_ptr,
    b_ptr,
    c_ptr,
    expert_indices_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bge,
    stride_bgn,
    stride_bgk,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Autotuned version of the forward kernel
    """
    _kernel_cg_forward_aligned(
        a_ptr,
        b_ptr,
        c_ptr,
        expert_indices_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bge,
        stride_bgn,
        stride_bgk,
        stride_cm,
        stride_cn,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        GROUP_SIZE_M,
    )


def cg_grouped_gemm_forward(
    inputs: torch.Tensor,
    expert_weights: torch.Tensor,
    expert_indices: torch.Tensor,
    group_size_m: int = 128,
    persistent_kernel: bool = True,
) -> torch.Tensor:
    """
    Perform contiguous grouped GEMM for MoE forward pass.

    Args:
        inputs: Input tensor of shape [M, K]
        expert_weights: Expert weight tensors of shape [num_experts, N, K]
        expert_indices: Expert assignment indices of shape [M]
        group_size_m: Group size for M dimension
        persistent_kernel: Whether to use persistent kernel with L2 cache optimization

    Returns:
        Output tensor of shape [M, N]
    """
    assert inputs.is_contiguous(), "Input must be contiguous"
    assert expert_weights.is_contiguous(), "Expert weights must be contiguous"
    assert inputs.dtype == expert_weights.dtype, "Input and weight dtypes must match"

    M, K = inputs.shape
    num_experts, N, K_w = expert_weights.shape
    assert K == K_w, f"Input K ({K}) must match weight K ({K_w})"
    assert (
        expert_indices.shape[0] == M
    ), "Expert indices must have same length as input rows"

    # Create output tensor
    output = torch.empty((M, N), device=inputs.device, dtype=inputs.dtype)

    # Launch kernel
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    _kernel_cg_forward[grid](
        inputs,
        expert_weights,
        output,
        expert_indices,
        M,
        N,
        K,
        inputs.stride(0),
        inputs.stride(1),
        expert_weights.stride(0),
        expert_weights.stride(1),
        expert_weights.stride(2),
        output.stride(0),
        output.stride(1),
    )

    return output


@triton.jit
def _kernel_cg_backward_dw(
    grad_output_ptr,
    inputs_ptr,
    grad_weights_ptr,
    expert_indices_ptr,
    M_total,
    N,
    K,
    num_experts,
    stride_gom,
    stride_gon,
    stride_im,
    stride_ik,
    stride_gwe,
    stride_gwn,
    stride_gwk,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Triton kernel for computing weight gradients in grouped GEMM.
    """
    pid = tl.program_id(axis=0)

    # Compute block indices
    pid_n = pid % tl.cdiv(N, BLOCK_SIZE_N)
    pid_k = (pid // tl.cdiv(N, BLOCK_SIZE_N)) % tl.cdiv(K, BLOCK_SIZE_K)
    pid_group = pid // (tl.cdiv(N, BLOCK_SIZE_N) * tl.cdiv(K, BLOCK_SIZE_K))

    # Get expert for this group
    group_start_m = pid_group * GROUP_SIZE_M * BLOCK_SIZE_M
    if group_start_m >= M_total:
        return
    expert_idx = tl.load(expert_indices_ptr + group_start_m)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)

    # Compute offsets
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_m = tl.arange(0, BLOCK_SIZE_M)

    # Iterate over M dimension in this group
    for m_block in range(GROUP_SIZE_M):
        m_start = group_start_m + m_block * BLOCK_SIZE_M
        if m_start >= M_total:
            break

        # Check if this block uses the same expert
        block_expert = tl.load(expert_indices_ptr + m_start)
        if block_expert != expert_idx:
            continue

        # Load grad_output and input blocks
        grad_out_ptrs = (
            grad_output_ptr
            + (m_start + offs_m[:, None]) * stride_gom
            + offs_n[None, :] * stride_gon
        )
        input_ptrs = (
            inputs_ptr
            + (m_start + offs_m[:, None]) * stride_im
            + offs_k[None, :] * stride_ik
        )

        mask_m = (m_start + offs_m) < M_total
        grad_out = tl.load(
            grad_out_ptrs, mask=mask_m[:, None] & (offs_n[None, :] < N), other=0.0
        )
        inputs = tl.load(
            input_ptrs, mask=mask_m[:, None] & (offs_k[None, :] < K), other=0.0
        )

        # Accumulate gradient
        accumulator = tl.dot(tl.trans(grad_out), inputs, accumulator)

    # Store gradient for this expert
    grad_weight = accumulator.to(tl.float16)
    grad_w_ptrs = (
        grad_weights_ptr
        + expert_idx * stride_gwe
        + offs_n[:, None] * stride_gwn
        + offs_k[None, :] * stride_gwk
    )
    mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
    tl.store(grad_w_ptrs, grad_weight, mask=mask)


def cg_grouped_gemm_backward_weights(
    grad_output: torch.Tensor,
    inputs: torch.Tensor,
    expert_indices: torch.Tensor,
    num_experts: int,
    group_size_m: int = 128,
) -> torch.Tensor:
    """
    Compute weight gradients for grouped GEMM.

    Args:
        grad_output: Gradient w.r.t. output [M, N]
        inputs: Original input tensor [M, K]
        expert_indices: Expert assignment indices [M]
        num_experts: Total number of experts
        group_size_m: Group size for M dimension

    Returns:
        Gradient w.r.t. weights [num_experts, N, K]
    """
    M, N = grad_output.shape
    _, K = inputs.shape

    # Allocate gradient weights
    grad_weights = torch.zeros(
        (num_experts, N, K), device=grad_output.device, dtype=grad_output.dtype
    )

    # Configure kernel launch
    BLOCK_SIZE_M = min(128, M)
    BLOCK_SIZE_N = min(128, N)
    BLOCK_SIZE_K = min(64, K)

    num_groups = triton.cdiv(M, group_size_m * BLOCK_SIZE_M)
    grid = (num_groups * triton.cdiv(N, BLOCK_SIZE_N) * triton.cdiv(K, BLOCK_SIZE_K),)

    _kernel_cg_backward_dw[grid](
        grad_output,
        inputs,
        grad_weights,
        expert_indices,
        M,
        N,
        K,
        num_experts,
        grad_output.stride(0),
        grad_output.stride(1),
        inputs.stride(0),
        inputs.stride(1),
        grad_weights.stride(0),
        grad_weights.stride(1),
        grad_weights.stride(2),
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        BLOCK_SIZE_M,
        group_size_m // BLOCK_SIZE_M,
    )

    return grad_weights


class ContiguousGroupedGEMM(torch.autograd.Function):
    """
    Autograd function for contiguous grouped GEMM operations.
    """

    @staticmethod
    def forward(ctx, inputs, expert_weights, expert_indices, group_size_m=128):
        output = cg_grouped_gemm_forward(
            inputs, expert_weights, expert_indices, group_size_m
        )
        ctx.save_for_backward(inputs, expert_weights, expert_indices)
        ctx.group_size_m = group_size_m
        ctx.num_experts = expert_weights.shape[0]
        return output

    @staticmethod
    def backward(ctx, grad_output):
        inputs, expert_weights, expert_indices = ctx.saved_tensors[:3]

        # Gradient w.r.t. weights
        grad_weights = cg_grouped_gemm_backward_weights(
            grad_output, inputs, expert_indices, ctx.num_experts, ctx.group_size_m
        )

        # Gradient w.r.t. inputs (using transposed weight multiplication)
        # This would require another kernel implementation
        grad_inputs = None  # TODO: Implement if needed

        return grad_inputs, grad_weights, None, None
