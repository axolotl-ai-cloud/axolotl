# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# credit - flat index forward kernel is derived from FBGemm:
# https://github.com/pytorch/FBGEMM/blob/main/fbgemm_gpu/experimental/gemm/triton_gemm

# pyre-unsafe
import logging
from typing import Tuple

import torch
import triton
import triton.language as tl

from .tma_autotuning import (
    _NV_CONFIGS,
    CudaUtils,
    early_config_prune,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# ==============  Start Triton Kernels ===============


@triton.autotune(
    configs=_NV_CONFIGS,
    key=["G", "M_BUCKET", "N", "K"],
    prune_configs_by={"early_config_prune": early_config_prune},
)
@triton.jit
def _kernel_mg_forward_hopper(
    a_ptr,
    b_ptr,
    c_ptr,
    m_sizes,
    M_TOTAL,
    # problem sizes
    G: tl.constexpr,
    M_BUCKET: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    # config
    NUM_SMS: tl.constexpr,
    TMA_SIZE: tl.constexpr,
    USE_EPILOGUE_SUBTILING: tl.constexpr,
    # tiles
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
) -> None:
    """Flat index style forward kernel for Hopper using tensor descriptors."""
    tbidx = tl.program_id(0)  # thread block index

    c_dtype = c_ptr.dtype.element_ty  # output dtype
    n_size = N // G

    # Compute the total rows spanned by all groups so descriptors cover valid ranges.
    total_rows = M_TOTAL.to(tl.int32)

    stride_k = tl.full([], K, dtype=tl.int32)
    stride_1 = tl.full([], 1, dtype=tl.int32)

    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[total_rows, stride_k],
        strides=[stride_k, stride_1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[tl.full([], N, dtype=tl.int32), stride_k],
        strides=[stride_k, stride_1],
        block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
    )

    M_end = tl.full([], 0, dtype=tl.int32)
    processed_tiles = 0

    for g in range(G):
        M_start = M_end
        m_size = tl.load(m_sizes + g)
        M_end = M_start + m_size

        if m_size > 0:
            group_base = c_ptr + M_start * n_size
            c_desc = tl.make_tensor_descriptor(
                group_base,
                shape=[m_size, tl.full([], n_size, dtype=tl.int32)],
                strides=[tl.full([], n_size, dtype=tl.int32), stride_1],
                block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
            )

            num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
            num_n_tiles = tl.cdiv(n_size, BLOCK_SIZE_N)
            group_num_tiles = num_m_tiles * num_n_tiles

            while (
                tbidx >= processed_tiles and tbidx < processed_tiles + group_num_tiles
            ):
                group_index = tbidx - processed_tiles

                tile_m_index = group_index % num_m_tiles
                tile_n_index = group_index // num_m_tiles

                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

                m_offset = (M_start + tile_m_index * BLOCK_SIZE_M).to(tl.int32)
                n_offset = (tile_n_index * BLOCK_SIZE_N).to(tl.int32)
                global_n_offset = (g * n_size + n_offset).to(tl.int32)

                for k_offset in range(0, K, BLOCK_SIZE_K):
                    a = a_desc.load([m_offset, k_offset])
                    b = b_desc.load([global_n_offset, k_offset])
                    accumulator += tl.dot(a, b.T)

                local_m_offset = (tile_m_index * BLOCK_SIZE_M).to(tl.int32)

                if USE_EPILOGUE_SUBTILING:
                    acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
                    acc = tl.permute(acc, (0, 2, 1))
                    acc0, acc1 = tl.split(acc)
                    c_desc.store([local_m_offset, n_offset], acc0.to(c_dtype))
                    c_desc.store(
                        [local_m_offset, n_offset + BLOCK_SIZE_N // 2],
                        acc1.to(c_dtype),
                    )
                else:
                    c_desc.store([local_m_offset, n_offset], accumulator.to(c_dtype))

                tbidx += NUM_SMS

            processed_tiles += group_num_tiles


"""
Backward pass for grouped GEMM with Triton, where grouping is M*G
We compute gradients with respect to both input (`grad_x`) and weights (`grad_w`).
"""


# ---- dx flat linear indexed ----
@triton.autotune(
    configs=_NV_CONFIGS,
    key=["G", "M_BUCKET", "N", "K"],
    prune_configs_by={"early_config_prune": early_config_prune},
)
@triton.jit
def _kernel_mg_dx_tma(
    grad_output_ptr,
    w_ptr,
    grad_input_ptr,
    m_sizes,
    M_TOTAL,
    # problem sizes
    G: tl.constexpr,
    M_BUCKET: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    # config
    NUM_SMS: tl.constexpr,
    # tiles
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
) -> None:
    """Compute grad_input = grad_output @ w using tensor descriptors."""
    tbidx = tl.program_id(0)

    c_dtype = grad_input_ptr.dtype.element_ty
    stride_1 = tl.full([], 1, dtype=tl.int32)
    stride_n = tl.full([], N, dtype=tl.int32)
    stride_k = tl.full([], K, dtype=tl.int32)

    total_rows = M_TOTAL.to(tl.int32)

    grad_output_desc = tl.make_tensor_descriptor(
        grad_output_ptr,
        shape=[total_rows, stride_n],
        strides=[stride_n, stride_1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )
    w_desc = tl.make_tensor_descriptor(
        w_ptr,
        shape=[stride_n, stride_k],
        strides=[stride_k, stride_1],
        block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
    )

    M_end = tl.full([], 0, dtype=tl.int32)
    processed_tiles = 0

    for g in range(G):
        M_start = M_end
        m_size = tl.load(m_sizes + g)
        M_end = M_start + m_size

        if m_size > 0:
            grad_input_desc = tl.make_tensor_descriptor(
                grad_input_ptr + M_start * K,
                shape=[m_size, stride_k],
                strides=[stride_k, stride_1],
                block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
            )

            num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
            num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
            group_num_tiles = num_m_tiles * num_k_tiles

            while (
                tbidx >= processed_tiles and tbidx < processed_tiles + group_num_tiles
            ):
                group_index = tbidx - processed_tiles
                tile_m_index = group_index % num_m_tiles
                tile_k_index = group_index // num_m_tiles

                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

                m_offset = (M_start + tile_m_index * BLOCK_SIZE_M).to(tl.int32)
                k_offset = (tile_k_index * BLOCK_SIZE_K).to(tl.int32)

                for n_offset in range(0, N, BLOCK_SIZE_N):
                    grad_y = grad_output_desc.load([m_offset, n_offset])
                    w_tile = w_desc.load([n_offset, k_offset])
                    accumulator += tl.dot(grad_y, w_tile)

                local_m_offset = (tile_m_index * BLOCK_SIZE_M).to(tl.int32)
                grad_input_desc.store(
                    [local_m_offset, k_offset], accumulator.to(c_dtype)
                )

                tbidx += NUM_SMS

            processed_tiles += group_num_tiles


@triton.autotune(
    configs=_NV_CONFIGS,
    key=["G", "M_BUCKET", "N", "K"],
    prune_configs_by={"early_config_prune": early_config_prune},
)
@triton.jit
def _kernel_mg_dw_tma(
    x_ptr,
    grad_output_ptr,
    grad_weight_ptr,
    m_sizes,
    M_TOTAL,
    # problem sizes
    G: tl.constexpr,
    M_BUCKET: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    # config
    NUM_SMS: tl.constexpr,
    # tiles
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
) -> None:
    """Compute grad_weight = grad_output.T @ x using tensor descriptors."""
    tbidx = tl.program_id(0)

    c_dtype = grad_weight_ptr.dtype.element_ty
    stride_1 = tl.full([], 1, dtype=tl.int32)
    stride_n = tl.full([], N, dtype=tl.int32)
    stride_k = tl.full([], K, dtype=tl.int32)

    total_rows = M_TOTAL.to(tl.int32)

    x_desc = tl.make_tensor_descriptor(
        x_ptr,
        shape=[total_rows, stride_k],
        strides=[stride_k, stride_1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    grad_output_desc = tl.make_tensor_descriptor(
        grad_output_ptr,
        shape=[total_rows, stride_n],
        strides=[stride_n, stride_1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )
    grad_weight_desc = tl.make_tensor_descriptor(
        grad_weight_ptr,
        shape=[stride_n, stride_k],
        strides=[stride_k, stride_1],
        block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
    )

    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    total_tiles = num_n_tiles * num_k_tiles

    for tile_idx in range(tbidx, total_tiles, NUM_SMS):
        tile_n_idx = tile_idx % num_n_tiles
        tile_k_idx = tile_idx // num_n_tiles

        n_offset = (tile_n_idx * BLOCK_SIZE_N).to(tl.int32)
        k_offset = (tile_k_idx * BLOCK_SIZE_K).to(tl.int32)

        accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)

        M_end = tl.full([], 0, dtype=tl.int32)
        for g in range(G):
            M_start = M_end
            m_size = tl.load(m_sizes + g)
            M_end = M_start + m_size

            if m_size > 0:
                for m_offset in range(0, m_size, BLOCK_SIZE_M):
                    m_global = (M_start + m_offset).to(tl.int32)
                    grad_block = grad_output_desc.load([m_global, n_offset])
                    x_block = x_desc.load([m_global, k_offset])
                    accumulator += tl.dot(
                        grad_block.to(tl.float32).T,
                        x_block.to(tl.float32),
                    )

        grad_weight_desc.store([n_offset, k_offset], accumulator.to(c_dtype))


# ======== End Triton kernels ========
# ======== End Triton kernels ========

# ======== Triton wrapper functions ========

# ----- main forward pass wrapper -----


def grouped_gemm_forward(
    x: torch.Tensor,
    w: torch.Tensor,
    m_sizes: torch.Tensor,
    tma_size: int = 128,
    using_fp8: bool = False,
) -> torch.Tensor:
    """Grouped GEMM forward using Hopper TMA kernels."""
    if not CudaUtils.verify_tma():
        raise NotImplementedError("Grouped GEMM without TMA is not supported yet")
    if using_fp8:
        raise NotImplementedError(
            "FP8 path not implemented with the new Triton API yet"
        )

    G = m_sizes.shape[0]

    assert x.is_contiguous()
    assert w.is_contiguous()
    assert m_sizes.is_contiguous()

    M_total, K = x.shape
    N = w.shape[0]
    assert K == w.shape[1], f"Input K ({K}) must match weight K ({w.shape[1]})"

    y = torch.empty((M_total, N // G), device=x.device, dtype=x.dtype)
    if M_total == 0:
        return y

    NUM_SMS = CudaUtils.get_num_sms()
    USE_EPILOGUE_SUBTILING = False

    def grid(_meta):
        return (NUM_SMS,)

    M_BUCKET = triton.next_power_of_2(M_total)
    _kernel_mg_forward_hopper[grid](
        x,
        w,
        y,
        m_sizes,
        M_total,
        G,
        M_BUCKET,
        N,
        K,
        NUM_SMS,
        TMA_SIZE=tma_size,
        USE_EPILOGUE_SUBTILING=USE_EPILOGUE_SUBTILING,
    )
    return y


# ======== Improved Backward =============
def grouped_gemm_backward(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    m_sizes: torch.Tensor,
    use_tma: bool = True,
    tma_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Unified backward pass for grouped GeMM with M*G grouping.
    Uses optimized TMA-based implementations for both dx and dw when available.

    Args:
        grad_output: Gradient of output, shape [M_total, N]
        x: Input tensor from forward pass, shape [M_total, K]
        w: Weight tensor from forward pass, shape [N, K]
        m_sizes: Group sizes tensor, shape [G]
        use_tma: Whether to try using TMA acceleration (if available)
        tma_size: Size of TMA descriptor in bytes


    Returns:
        Tuple of gradients with respect to x and w: (grad_x, grad_w)
    """
    logging.info("Starting unified grouped_gemm_backward")

    # do this once, seems expensive
    NUM_SMS = CudaUtils.get_num_sms()

    # Basic validation
    M_total, K_x = x.shape
    M_grad, N = grad_output.shape
    N_w, K_w = w.shape

    # Check dimensions
    if K_x != K_w:
        raise ValueError(f"K dimension mismatch: x has K={K_x}, w has K={K_w}")
    if M_total != M_grad:
        raise ValueError(
            f"M dimension mismatch: x has M={M_total}, grad_output has M={M_grad}"
        )

    # Check total M matches sum of group sizes
    sum_m_sizes = m_sizes.sum().item()
    if M_total != sum_m_sizes:
        raise ValueError(
            f"Sum of m_sizes ({sum_m_sizes}) must match M_total ({M_total})"
        )

    # Make sure inputs are contiguous
    grad_output = grad_output.contiguous()
    x = x.contiguous()
    w = w.contiguous()
    m_sizes = m_sizes.contiguous()

    # Check TMA support
    if use_tma and not CudaUtils.verify_tma():
        logging.info("TMA requested but not supported on this device")
        use_tma = False

    # Compute grad_x using flat linear implementation
    try:
        logging.info("Computing grad_x with flat linear kernel")

        # Use TMA-optimized implementation
        grad_x = grouped_gemm_dx_tma(
            grad_output=grad_output,
            w=w,
            m_sizes=m_sizes,
            num_sms=NUM_SMS,
            tma_size=tma_size,
        )

    except Exception as e:
        logging.error(f"Error in grad_x computation: {e}")
        raise

    # Compute grad_w using flat linear style implementation
    try:
        logging.info("Computing grad_w with flat linear kernel")

        grad_w = grouped_gemm_dw_tma(
            x, grad_output, m_sizes, num_sms=NUM_SMS, tma_size=tma_size
        )
    except Exception as e:
        logging.error(f"Error in grad_w computation: {e}")
        raise

    return grad_x, grad_w


# ----- dx backward pass wrapper -----


def grouped_gemm_dx_tma(
    grad_output: torch.Tensor,
    w: torch.Tensor,
    m_sizes: torch.Tensor,
    num_sms: int = 132,
    tma_size: int = 128,
) -> torch.Tensor:
    """Compute grad_x using the Hopper grouped GEMM kernel."""
    if not CudaUtils.verify_tma():
        raise NotImplementedError("Optimized dx computation requires TMA support")

    grad_output = grad_output.contiguous()
    w = w.contiguous()
    m_sizes = m_sizes.contiguous()

    M_total, N = grad_output.shape
    N_w, K = w.shape
    if N != N_w:
        raise ValueError(f"Grad_output N ({N}) must match weight N ({N_w})")

    if m_sizes.sum().item() != M_total:
        raise ValueError("Sum of m_sizes must equal the number of rows in grad_output")

    grad_x = torch.empty(
        (M_total, K), device=grad_output.device, dtype=grad_output.dtype
    )

    NUM_SMS = num_sms

    def grid(_meta):
        return (NUM_SMS,)

    M_BUCKET = triton.next_power_of_2(M_total)
    _kernel_mg_dx_tma[grid](
        grad_output,
        w,
        grad_x,
        m_sizes,
        M_total,
        m_sizes.shape[0],
        M_BUCKET,
        N,
        K,
        NUM_SMS,
    )
    return grad_x


# ======== dw wrapper function ==========


def grouped_gemm_dw_tma(
    x: torch.Tensor,
    grad_output: torch.Tensor,
    m_sizes: torch.Tensor,
    num_sms: int = 132,
    tma_size: int = 128,
) -> torch.Tensor:
    """Compute grad_w using the Hopper grouped GEMM kernel."""
    if not CudaUtils.verify_tma():
        raise RuntimeError("TMA grouped GEMM requested on a device without TMA support")

    x = x.contiguous()
    grad_output = grad_output.contiguous()
    m_sizes = m_sizes.contiguous()

    M_total, K = x.shape
    M_grad, N = grad_output.shape
    if M_total != M_grad:
        raise ValueError("x and grad_output must have matching batch dimension")
    if m_sizes.sum().item() != M_total:
        raise ValueError("Sum of m_sizes must equal the number of rows in the inputs")

    grad_w = torch.zeros((N, K), device=x.device, dtype=x.dtype)

    NUM_SMS = num_sms

    def grid(_meta):
        return (NUM_SMS,)

    M_BUCKET = triton.next_power_of_2(M_total)
    _kernel_mg_dw_tma[grid](
        x,
        grad_output,
        grad_w,
        m_sizes,
        M_total,
        m_sizes.shape[0],
        M_BUCKET,
        N,
        K,
        NUM_SMS,
    )
    return grad_w


# ======== End Backwards Wrapper Functions =============

# ======== PyTorch wrapper functions ========


class GroupedGemmMg(torch.autograd.Function):
    """
    Autograd function for GroupedGEMM with M*G grouping.
    Supports both standard and FP8 quantized operations.
    """

    @staticmethod
    def forward(ctx, x, w, m_sizes, use_tma=True, tma_size=128, using_fp8=False):
        """
        Forward pass of GroupedGEMM.

        Args:
            x: Input tensor, shape [M_total, K]
            w: Weight tensor, shape [N, K]
            m_sizes: Tensor of shape [G] containing the size of each group
            use_tma: Whether to try using TMA acceleration (if available)
            tma_size: Size of TMA descriptor in bytes
            using_fp8: Whether to use FP8 quantization

        Returns:
            Output tensor, shape [M_total, N]
        """

        # Use regular forward without quantization
        output = grouped_gemm_forward(
            x=x, w=w, m_sizes=m_sizes, tma_size=tma_size, using_fp8=False
        )

        # Save inputs and parameters for backward pass
        ctx.save_for_backward(x, w, m_sizes)
        ctx.use_tma = use_tma
        ctx.tma_size = tma_size

        ctx.save_for_backward(x, w, m_sizes)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of M*G GroupedGEMM.

        Args:
            grad_output: Gradient of output, shape [M_total, N]

        Returns:
            Tuple of gradients:
                - grad_x: Gradient with respect to x, shape [M_total, K]
                - grad_w: Gradient with respect to w, shape [N, K]
                - None: Gradient with respect to m_sizes (not differentiable)
                - None: Gradient with respect to use_tma (not differentiable)
                - None: Gradient with respect to tma_size (not differentiable)

        """
        # Retrieve saved tensors and parameters

        x, w, m_sizes = ctx.saved_tensors

        use_tma = ctx.use_tma
        tma_size = ctx.tma_size

        # Compute gradients using the unified implementation
        grad_x, grad_w = grouped_gemm_backward(
            grad_output=grad_output,
            x=x,
            w=w,
            m_sizes=m_sizes,
            use_tma=use_tma,
            tma_size=tma_size,
        )

        # Return gradients for all inputs (None for non-differentiable parameters)
        return grad_x, grad_w, None, None, None, None


def mg_grouped_gemm(
    x: torch.Tensor,
    w: torch.Tensor,
    m_sizes: torch.Tensor,
    use_tma: bool = True,
    tma_size: int = 128,
    using_fp8: bool = False,
) -> torch.Tensor:
    """
    Unified differentiable grouped GEMM operation for M*G grouped GEMM.
    Supports both standard precision and FP8 quantized operations.

    Args:
        x: Input tensor, shape [M_total, K]
        w: Weight tensor, shape [N, K]
        m_sizes: Tensor of shape [G] containing the size of each group
        use_tma: Whether to try using TMA acceleration (if available)
        tma_size: Size of TMA descriptor in bytes
        using_fp8: Whether to use FP8 quantization

    Returns:
        Output tensor, shape [M_total, N]
    """
    return GroupedGemmMg.apply(x, w, m_sizes, use_tma, tma_size, using_fp8)
