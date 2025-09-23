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
    TmaDescriptorHelper,
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
    a_desc_ptr,
    b_desc_ptr,
    c_ptr,
    workspace,
    m_sizes,
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
    """
    Flat index style forward kernel for Hopper.
    For simplicity, we always use TMA Load and TMA Store
    """
    tbidx = tl.program_id(0)  # thread block index

    c_dtype = c_ptr.dtype.element_ty  # output dtype

    c_desc_ptr = workspace + (tbidx * TMA_SIZE)  # for TMA Store

    M_end = 0
    M_start = 0
    processed_tiles = 0
    # Size of individual weight matrix
    n_size = N // G
    n_start = 0

    for g in range(G):
        # Move down along groups
        # reset to new M offset
        M_start = M_end
        m_size = tl.load(m_sizes + g)
        M_end = M_start + m_size
        n_start = n_size * g

        if m_size > 0:
            # Process this group

            # Acquire hold on c_desc_ptr for TMA Store
            tl.extra.cuda.tensormap.create2d(
                desc_ptr=c_desc_ptr,
                global_address=c_ptr + M_start * n_size,
                load_size=[BLOCK_SIZE_M, BLOCK_SIZE_N],
                global_size=[m_size, n_size],
                element_ty=c_dtype,
            )
            tl.extra.cuda.tensormap.fenceproxy_acquire(c_desc_ptr)

            # tiles for this group
            num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
            num_n_tiles = tl.cdiv(n_size, BLOCK_SIZE_N)
            group_num_tiles = num_m_tiles * num_n_tiles

            while tbidx >= processed_tiles and tbidx < (
                processed_tiles + group_num_tiles
            ):
                group_index = tbidx - processed_tiles

                # columnwise
                tile_m_index = group_index % num_m_tiles
                tile_n_index = group_index // num_m_tiles

                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

                m_offset = (M_start + (tile_m_index * BLOCK_SIZE_M)).to(tl.int32)
                n_offset = (tile_n_index * BLOCK_SIZE_N).to(tl.int32)
                global_n_offset = (n_start + n_offset).to(tl.int32)

                for k_offset in range(0, K, BLOCK_SIZE_K):
                    # input block [M,K]
                    a = tl._experimental_descriptor_load(
                        a_desc_ptr,
                        [m_offset, k_offset],
                        [BLOCK_SIZE_M, BLOCK_SIZE_K],
                        c_dtype,
                    )
                    # weight block [N, K]
                    b = tl._experimental_descriptor_load(
                        b_desc_ptr,
                        [global_n_offset, k_offset],
                        [BLOCK_SIZE_N, BLOCK_SIZE_K],
                        c_dtype,
                    )

                    accumulator += tl.dot(a, b.T)

                # Store using TMA

                m_offset = (tile_m_index * BLOCK_SIZE_M).to(tl.int32)

                if USE_EPILOGUE_SUBTILING:
                    acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
                    acc = tl.permute(acc, (0, 2, 1))
                    acc0, acc1 = tl.split(acc)
                    c0 = acc0.to(c_dtype)
                    tl._experimental_descriptor_store(
                        c_desc_ptr, c0, [m_offset, n_offset]
                    )
                    c1 = acc1.to(c_dtype)
                    tl._experimental_descriptor_store(
                        c_desc_ptr, c1, [m_offset, n_offset + BLOCK_SIZE_N // 2]
                    )
                else:
                    tl._experimental_descriptor_store(
                        c_desc_ptr,
                        accumulator.to(c_dtype),
                        [m_offset, n_offset],
                    )
                # move to next tile in group
                tbidx += NUM_SMS
            # Update the total tiles count for the next group
            processed_tiles += group_num_tiles


@triton.autotune(
    configs=_NV_CONFIGS,
    key=["G", "M_BUCKET", "N", "K"],
    prune_configs_by={"early_config_prune": early_config_prune},
)
@triton.jit
def _kernel_mg_forward_tma(
    a_desc_ptr,
    b_desc_ptr,
    c_ptr,
    workspace,
    m_sizes,
    a_scale_ptr,
    b_scale_ptr,
    # problem sizes
    G: tl.constexpr,
    M_BUCKET: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    # config
    NUM_SMS: tl.constexpr,
    USE_TMA_LOAD: tl.constexpr,
    USE_TMA_STORE: tl.constexpr,
    TMA_SIZE: tl.constexpr,
    USE_FP8: tl.constexpr,
    # tiles
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
) -> None:
    """
    Flat index style forward kernel.
    For simplicity, we always use TMA Load and TMA Store
    """
    tbidx = tl.program_id(0)  # thread block index

    c_dtype = c_ptr.dtype.element_ty

    c_desc_ptr = workspace + (tbidx * TMA_SIZE)

    M_end = 0
    processed_tiles = 0

    for g in range(G):
        # Move down along groups
        # reset to new M offset
        M_start = M_end
        m_size = tl.load(m_sizes + g)
        M_end = M_start + m_size

        if m_size > 0:
            # Process this group
            n_size = N

            # TMA Store prep
            tl.extra.cuda.tensormap.create2d(
                desc_ptr=c_desc_ptr,
                global_address=c_ptr + M_start * N,
                load_size=[BLOCK_SIZE_M, BLOCK_SIZE_N],
                global_size=[m_size, n_size],
                element_ty=c_dtype,
            )
            tl.extra.cuda.tensormap.fenceproxy_acquire(c_desc_ptr)

            # tiles for this group
            num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
            num_n_tiles = tl.cdiv(n_size, BLOCK_SIZE_N)
            group_num_tiles = num_m_tiles * num_n_tiles

            while tbidx >= processed_tiles and tbidx < (
                processed_tiles + group_num_tiles
            ):
                group_index = tbidx - processed_tiles

                tile_m_index = group_index % num_m_tiles
                tile_n_index = group_index // num_m_tiles

                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

                m_offset = (M_start + (tile_m_index * BLOCK_SIZE_M)).to(tl.int32)
                n_offset = (tile_n_index * BLOCK_SIZE_N).to(tl.int32)

                for k_offset in range(0, K, BLOCK_SIZE_K):
                    # input block [M,K]
                    a = tl._experimental_descriptor_load(
                        a_desc_ptr,
                        [m_offset, k_offset],
                        [BLOCK_SIZE_M, BLOCK_SIZE_K],
                        c_dtype,
                    )
                    # weight block [N, K]
                    b = tl._experimental_descriptor_load(
                        b_desc_ptr,
                        [n_offset, k_offset],
                        [BLOCK_SIZE_N, BLOCK_SIZE_K],
                        c_dtype,
                    )

                    accumulator += tl.dot(a, b.T)

                # Store using TMA

                m_offset = (tile_m_index * BLOCK_SIZE_M).to(tl.int32)
                # n_offset = (tile_n_index * BLOCK_SIZE_N).to(tl.int32)

                tl._experimental_descriptor_store(
                    c_desc_ptr,
                    accumulator.to(c_dtype),
                    [m_offset, n_offset],
                )

                # Move to the next tile
                tbidx += NUM_SMS
            # Update the total tiles count for the next group
            processed_tiles += group_num_tiles


@triton.autotune(
    configs=_NV_CONFIGS,
    key=["G", "M_BUCKET", "N", "K"],
    prune_configs_by={"early_config_prune": early_config_prune},
)
@triton.jit
def _kernel_mg_forward_no_tma(
    a_ptr,
    b_ptr,
    c_ptr,
    workspace,
    m_sizes,
    # problem sizes
    G: tl.constexpr,
    M_BUCKET: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    # config
    NUM_SMS: tl.constexpr,
    USE_TMA_LOAD: tl.constexpr,
    USE_TMA_STORE: tl.constexpr,
    TMA_SIZE: tl.constexpr,
    # tiles
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
) -> None:
    """
    Flat index style forward kernel.
    For bc and Ampere, we never use TMA Load and TMA Store
    """
    tbidx = tl.program_id(0)  # thread block index

    c_dtype = c_ptr.dtype.element_ty

    M_end = 0
    processed_tiles = 0

    for g in range(G):
        # Move down along groups
        # reset to new M offset
        M_start = M_end
        m_size = tl.load(m_sizes + g)
        M_end = M_start + m_size

        if m_size > 0:
            # Process this group
            n_size = N

            # tiles for this group
            num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
            num_n_tiles = tl.cdiv(n_size, BLOCK_SIZE_N)
            group_num_tiles = num_m_tiles * num_n_tiles

            while tbidx >= processed_tiles and tbidx < (
                processed_tiles + group_num_tiles
            ):
                group_index = tbidx - processed_tiles

                tile_m_index = group_index % num_m_tiles
                tile_n_index = group_index // num_m_tiles

                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

                offs_am = tile_m_index * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                offs_bn = tile_n_index * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                offs_k = tl.arange(0, BLOCK_SIZE_K)

                a_ptrs = a_ptr + (M_start + offs_am[:, None]) * K + offs_k[None, :]
                b_ptrs = b_ptr + (offs_bn[:, None]) * K + offs_k[None, :]

                for _ in range(0, K, BLOCK_SIZE_K):
                    # Load with bounds checking
                    a = tl.load(a_ptrs, mask=offs_am[:, None] < m_size)
                    b = tl.load(b_ptrs, mask=offs_bn[:, None] < n_size)

                    # Main matmul
                    accumulator += tl.dot(a, b.T)

                    # Update pointers for next block
                    a_ptrs += BLOCK_SIZE_K
                    b_ptrs += BLOCK_SIZE_K

                # Store without TMA
                offs_am = tile_m_index * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                offs_bn = tile_n_index * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

                c = accumulator.to(c_dtype)

                tl.store(
                    c_ptr
                    + (M_start + offs_am[:, None]) * N  # Row stride is N
                    + offs_bn[None, :],  # Column offset
                    c,
                    mask=offs_am[:, None] < m_size and offs_bn[None, :] < n_size,
                )
                # Move to the next tile
                tbidx += NUM_SMS
            # Update the total tiles count for the next group
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
    grad_output_desc_ptr,  # [MG, N]
    w_desc_ptr,  # [N, K]
    grad_input_ptr,  # output grad_x [MG, K]
    workspace,  # for TMA store
    m_sizes,  # group sizes [G]
    # problem sizes
    G: tl.constexpr,
    M_BUCKET: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    # config
    NUM_SMS: tl.constexpr,
    USE_TMA_LOAD: tl.constexpr,
    USE_TMA_STORE: tl.constexpr,
    TMA_SIZE: tl.constexpr,
    # tiles
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
) -> None:
    """
    TMA-optimized kernel for computing gradients with respect to input (dx).
    For the forward pass Y = X @ W.T, the backward for input is:
    grad_X = grad_Y @ W

    This maps to [MG, N] @ [N, K] -> [MG, K]

    Key differences from forward:
    1. W is used directly and not transposed
    2. The reduction dimension is now N (not K)
    3. Output is [M, K] instead of [M, N]
    """
    tbidx = tl.program_id(0)  # thread block index

    c_dtype = grad_input_ptr.dtype.element_ty
    c_desc_ptr = workspace + (tbidx * TMA_SIZE)

    M_end = 0
    processed_tiles = 0

    for g in range(G):
        # Move down along groups - same as forward
        M_start = M_end
        m_size = tl.load(m_sizes + g)
        M_end = M_start + m_size

        if m_size > 0:
            # Process this group
            # tiles for this group - now producing [M, K] output
            num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
            num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
            group_num_tiles = num_m_tiles * num_k_tiles

            # TMA Store prep for [M, K] output
            tl.extra.cuda.tensormap.create2d(
                desc_ptr=c_desc_ptr,
                global_address=grad_input_ptr + M_start * K,
                load_size=[BLOCK_SIZE_M, BLOCK_SIZE_K],
                global_size=[m_size, K],
                element_ty=c_dtype,
            )
            tl.extra.cuda.tensormap.fenceproxy_acquire(c_desc_ptr)

            while tbidx >= processed_tiles and tbidx < (
                processed_tiles + group_num_tiles
            ):
                group_index = tbidx - processed_tiles

                # Different tiling scheme for [M, K] output
                tile_m_index = group_index % num_m_tiles
                tile_k_index = group_index // num_m_tiles

                # for grad_input block [M, K]
                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

                # Position in full matrix
                m_offset = (M_start + (tile_m_index * BLOCK_SIZE_M)).to(tl.int32)
                k_offset = (tile_k_index * BLOCK_SIZE_K).to(tl.int32)

                # reduce along N dimension (instead of K in forward)
                for n_offset in range(0, N, BLOCK_SIZE_N):
                    # grad_output block [M, N]
                    grad_output = tl._experimental_descriptor_load(
                        grad_output_desc_ptr,
                        [m_offset, n_offset],
                        [BLOCK_SIZE_M, BLOCK_SIZE_N],
                        c_dtype,
                    )

                    # weight block [N, K] - no transpose needed
                    w = tl._experimental_descriptor_load(
                        w_desc_ptr,
                        [n_offset, k_offset],
                        [BLOCK_SIZE_N, BLOCK_SIZE_K],
                        c_dtype,
                    )

                    # grad_x = grad_output @ w
                    # reducing along N dimension
                    accumulator += tl.dot(grad_output, w)

                # Store using TMA
                m_offset = (tile_m_index * BLOCK_SIZE_M).to(tl.int32)
                # k_offset = (tile_k_index * BLOCK_SIZE_K).to(tl.int32)

                tl._experimental_descriptor_store(
                    c_desc_ptr,
                    accumulator.to(c_dtype),
                    [m_offset, k_offset],
                )

                # Move to the next tile
                tbidx += NUM_SMS

            # Update the total tiles count for the next group
            processed_tiles += group_num_tiles


# ---- dw flat linear indexed ----


@triton.autotune(
    configs=_NV_CONFIGS,
    key=["G", "M_BUCKET", "N", "K"],
    prune_configs_by={"early_config_prune": early_config_prune},
)
@triton.jit
def _kernel_mg_dw_tma(
    x_desc_ptr,  # input descriptor [M_total, K]
    grad_output_desc_ptr,  # grad_output descriptor [M_total, N]
    grad_weight_ptr,  # output grad_w [N, K]
    workspace,  # workspace for TMA store
    m_sizes,  # group sizes [G]
    # problem sizes
    G: tl.constexpr,
    M_BUCKET: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    # config
    NUM_SMS: tl.constexpr,
    USE_TMA_LOAD: tl.constexpr,
    USE_TMA_STORE: tl.constexpr,
    TMA_SIZE: tl.constexpr,
    # tiles
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,  # block size for reduction dimension
) -> None:
    """
    Improved TMA-optimized kernel for computing gradients with respect to weights (dw).
    Uses flat index structure similar to forward.

    For the forward pass Y = X @ W.T,
    the backward for weights is:
    grad_W = grad_Y.T @ X

    Where:
    - grad_Y is [MG, N]
    - X is [MG, K]
    - grad_W is [N, K]
    - we return [N,K]
    """
    # Get thread block index l
    tbidx = tl.program_id(0)

    # Get output data type
    c_dtype = grad_weight_ptr.dtype.element_ty

    # Calculate number of output tiles
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    total_output_tiles = num_n_tiles * num_k_tiles

    # Process tiles in strided manner across SMs
    for tile_idx in range(tbidx, total_output_tiles, NUM_SMS):
        # Calculate tile indices
        tile_n_idx = tile_idx % num_n_tiles
        tile_k_idx = tile_idx // num_n_tiles

        # Calculate global offsets
        n_offset = tile_n_idx * BLOCK_SIZE_N
        k_offset = tile_k_idx * BLOCK_SIZE_K

        # Initialize accumulator for this output tile [N, K]
        accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)

        # Process each group
        M_end = 0
        for g in range(G):
            # Get group boundaries
            M_start = M_end
            m_size = tl.load(m_sizes + g)
            M_end = M_start + m_size

            # Only process if group is non-empty
            if m_size > 0:
                # Process this group in chunks along the M dimension
                for m_offset in range(0, m_size, BLOCK_SIZE_M):
                    # Calculate actual block size (handling boundary)
                    m_block_size = tl.minimum(BLOCK_SIZE_M, m_size - m_offset)

                    # Only process if we have actual work to do
                    if m_block_size > 0:
                        # Global offset for this chunk
                        m_global_offset = M_start + m_offset

                        if USE_TMA_LOAD:
                            # Load input chunk [M_chunk, K] using TMA
                            x_block = tl._experimental_descriptor_load(
                                x_desc_ptr,
                                [m_global_offset, k_offset],
                                [BLOCK_SIZE_M, BLOCK_SIZE_K],
                                c_dtype,
                            )

                            # Load grad_output chunk [M_chunk, N] using TMA
                            grad_output_block = tl._experimental_descriptor_load(
                                grad_output_desc_ptr,
                                [m_global_offset, n_offset],
                                [BLOCK_SIZE_M, BLOCK_SIZE_N],
                                c_dtype,
                            )

                            # Apply masks for valid regions
                            offs_m = tl.arange(0, BLOCK_SIZE_M)
                            m_mask = offs_m < m_block_size

                            # Zero out invalid elements
                            x_block = tl.where(m_mask[:, None], x_block, 0.0)
                            grad_output_block = tl.where(
                                m_mask[:, None], grad_output_block, 0.0
                            )
                        else:
                            # Manual load with bounds checking
                            offs_m = tl.arange(0, BLOCK_SIZE_M)
                            offs_n = tl.arange(0, BLOCK_SIZE_N)
                            offs_k = tl.arange(0, BLOCK_SIZE_K)

                            # Create masks
                            m_mask = offs_m < m_block_size
                            n_mask = offs_n < N - n_offset
                            k_mask = offs_k < K - k_offset

                            # Combined masks
                            mk_mask = m_mask[:, None] & k_mask[None, :]
                            mn_mask = m_mask[:, None] & n_mask[None, :]

                            # Global offsets for loading
                            m_global_offs = m_global_offset + offs_m

                            # Load x block [M_chunk, K]
                            x_block = tl.load(
                                x_desc_ptr
                                + m_global_offs[:, None] * K
                                + (k_offset + offs_k)[None, :],
                                mask=mk_mask,
                                other=0.0,
                            )

                            # Load grad_output block [M_chunk, N]
                            grad_output_block = tl.load(
                                grad_output_desc_ptr
                                + m_global_offs[:, None] * N
                                + (n_offset + offs_n)[None, :],
                                mask=mn_mask,
                                other=0.0,
                            )

                        # Compute partial contribution: grad_W += grad_Y.T @ X
                        # transpose grad_output for the matmul
                        contribution = tl.dot(
                            grad_output_block.to(tl.float32).T,  # [N, M_chunk]
                            x_block.to(tl.float32),  # [M_chunk, K]
                        )

                        # Accumulate
                        accumulator += contribution

        # Store the result
        if USE_TMA_STORE:
            # Store using TMA
            tl._experimental_descriptor_store(
                workspace,  # TMA store descriptor
                accumulator.to(c_dtype),
                [n_offset, k_offset],
            )
        else:
            # Manual store with bounds checking
            offs_n = tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)

            # Create masks for bounds checking
            n_mask = offs_n < N - n_offset
            k_mask = offs_k < K - k_offset
            output_mask = n_mask[:, None] & k_mask[None, :]

            # Store the result
            tl.store(
                grad_weight_ptr
                + (n_offset + offs_n)[:, None] * K
                + (k_offset + offs_k)[None, :],
                accumulator.to(c_dtype),
                mask=output_mask,
            )


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
    """
    M*G style grouped GEMM with TMA and Float8 support.
    # Removed for now - FP8 support is triggered by passing x_scale and w_scale tensors.

    """
    if not CudaUtils.verify_tma():
        raise NotImplementedError("Grouped GEMM without TMA is not supported yet")

    G = m_sizes.shape[0]

    assert x.is_contiguous()
    assert w.is_contiguous()
    assert m_sizes.is_contiguous()

    # Total input size is now [M_total, K] where M_total is the sum of all group sizes
    M_total, K = x.shape
    N = w.shape[0]  # N is now the same for all groups

    assert K == w.shape[1], f"Input K ({K}) must match weight K ({w.shape[1]})"

    # Verify that all group sizes are multiples of ALIGN_SIZE_M
    # This check is commented out because it will involve a GPU-CPU sync
    # assert torch.remainder(m_sizes, ALIGN_SIZE_M).max() == 0, "Group sizes must be a multiple of ALIGN_SIZE_M"

    # Create output tensor with correct shape [M_total, N]
    y = torch.empty((M_total, N // G), device=x.device, dtype=x.dtype)

    if M_total == 0:
        return y

    NUM_SMS = CudaUtils.get_num_sms()
    USE_TMA_LOAD = True
    USE_TMA_STORE = True
    USE_EPILOGUE_SUBTILING = False

    # TMA descriptor helper
    desc_helper = None
    desc_x = x
    desc_w = w
    workspace = None

    if USE_TMA_LOAD:
        desc_helper = TmaDescriptorHelper(tma_size=tma_size)
        desc_helper.init_tma_descriptor("x")
        desc_helper.init_tma_descriptor("w")
        desc_x = desc_helper.get_tma_descriptor_kernel_param("x")
        desc_w = desc_helper.get_tma_descriptor_kernel_param("w")

    if USE_TMA_STORE:
        if desc_helper is None:
            raise RuntimeError(
                "TMA descriptors must be initialized when USE_TMA_STORE is True"
            )
        workspace = torch.empty(
            NUM_SMS * desc_helper.tma_size,
            device=x.device,
            dtype=torch.uint8,
        )

    def grid(META):
        if USE_TMA_LOAD:
            nonlocal desc_helper
            desc_helper.fill_2d_tma_descriptor(
                "x",
                x.data_ptr(),
                M_total,
                K,
                META["BLOCK_SIZE_M"],
                META["BLOCK_SIZE_K"],
                x.element_size(),
            )

            desc_helper.fill_2d_tma_descriptor(
                "w",
                w.data_ptr(),
                N,
                K,
                META["BLOCK_SIZE_N"],
                META["BLOCK_SIZE_K"],
                w.element_size(),
            )
        return (NUM_SMS,)

    M_BUCKET = triton.next_power_of_2(M_total)

    _kernel_mg_forward_hopper[grid](
        desc_x,
        desc_w,
        y,
        workspace,
        m_sizes,
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
    """
    Optimized backward pass wrapper for computing gradient with respect to input (dx)
    using TMA patterns similar to the forward pass.

    Args:
        grad_output: Gradient of output, shape [M_total, N]
        w: Weight tensor, shape [N, K]
        m_sizes: Group sizes tensor, shape [G]
        tma_size: Size of TMA descriptor
        # using_fp8: Whether to use FP8 quantization
        # grad_output_scale: Scale for grad_output in FP8 mode
        # w_scale: Scale for w in FP8 mode

    Returns:
        grad_x: Gradient with respect to x, shape [M_total, K]
    """
    """
    Optimized backward pass for computing gradient with respect to input (dx)
    using TMA patterns similar to the forward pass.

    Args:
        grad_output: Gradient of output, shape [M_total, N]
        w: Weight tensor, shape [N, K]
        m_sizes: Group sizes tensor, shape [G]
        tma_size: Size of TMA descriptor
        using_fp8: Whether to use FP8 quantization
        # grad_output_scale: Scale for grad_output in FP8 mode
        # w_scale: Scale for w in FP8 mode

    Returns:
        grad_x: Gradient with respect to x, shape [M_total, K]
    """
    if not CudaUtils.verify_tma():
        raise NotImplementedError("Optimized dx computation requires TMA support")

    G = m_sizes.shape[0]

    assert grad_output.is_contiguous()
    assert w.is_contiguous()
    assert m_sizes.is_contiguous()

    M_total, N_grad = grad_output.shape
    N_w, K = w.shape

    # Check dimensions
    assert N_grad == N_w, f"Grad_output N ({N_grad}) must match weight N ({N_w})"

    # Verify that the sum of m_sizes matches M_total
    sum_m_sizes = m_sizes.sum().item()
    assert M_total == sum_m_sizes, (
        f"Sum of m_sizes ({sum_m_sizes}) must match M_total ({M_total})"
    )

    # Create output tensor (grad_x) with shape [M_total, K]
    grad_x = torch.empty(
        (M_total, K), device=grad_output.device, dtype=grad_output.dtype
    )

    NUM_SMS = num_sms  # CudaUtils.get_num_sms()
    USE_TMA_LOAD = True
    USE_TMA_STORE = True

    # Set up TMA descriptors
    desc_helper = TmaDescriptorHelper(tma_size=tma_size)
    desc_helper.init_tma_descriptor("grad_output")
    desc_helper.init_tma_descriptor("w")
    desc_grad_output = desc_helper.get_tma_descriptor_kernel_param("grad_output")
    desc_w = desc_helper.get_tma_descriptor_kernel_param("w")

    # Allocate workspace for TMA store
    workspace = torch.empty(
        NUM_SMS * desc_helper.tma_size,
        device=grad_output.device,
        dtype=torch.uint8,
    )

    def grid(META):
        # Fill TMA descriptors with appropriate dimensions
        desc_helper.fill_2d_tma_descriptor(
            "grad_output",
            grad_output.data_ptr(),
            M_total,
            N_grad,
            META["BLOCK_SIZE_M"],
            META["BLOCK_SIZE_N"],
            grad_output.element_size(),
        )

        desc_helper.fill_2d_tma_descriptor(
            "w",
            w.data_ptr(),
            N_w,
            K,
            META["BLOCK_SIZE_N"],
            META["BLOCK_SIZE_K"],
            w.element_size(),
        )
        return (NUM_SMS,)

    M_BUCKET = triton.next_power_of_2(M_total)

    # Launch the flat linear kernel for computing grad_x
    _kernel_mg_dx_tma[grid](
        desc_grad_output,
        desc_w,
        grad_x,
        workspace,
        m_sizes,
        G,
        M_BUCKET,
        N_grad,  # N dimension is now the reduction dimension
        K,
        NUM_SMS,
        USE_TMA_LOAD,
        USE_TMA_STORE,
        TMA_SIZE=tma_size,
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
    """
    Optimized flat linear kernel computation of gradients with respect to weights (dw) using TMA.
    For the forward pass Y = X @ W.T, the backward for weights is:
    grad_W = grad_Y.T @ X

    Args:
        x: Input tensor, shape [M_total, K]
        grad_output: Gradient of output, shape [M_total, N]
        m_sizes: Group sizes tensor, shape [G]
        tma_size: Size of TMA descriptor in bytes


    Returns:
        grad_w: Gradient with respect to weights, shape [N, K]
    """
    # Check TMA support
    if not CudaUtils.verify_tma():
        raise RuntimeError("TMA grouped GEMM requested on a device without TMA support")

    # Get group count
    G = m_sizes.shape[0]

    # Ensure contiguous tensors
    x = x.contiguous()
    grad_output = grad_output.contiguous()
    m_sizes = m_sizes.contiguous()

    # Get dimensions
    M_total, K_x = x.shape
    M_grad, N = grad_output.shape

    # Check dimensions
    assert M_total == M_grad, f"x M ({M_total}) must match grad_output M ({M_grad})"

    # Verify that the sum of m_sizes matches M_total
    sum_m_sizes = m_sizes.sum().item()
    assert sum_m_sizes == M_total, (
        f"Sum of m_sizes ({sum_m_sizes}) must match M_total ({M_total})"
    )

    # Create output tensor (grad_w) with shape [N, K]
    grad_w = torch.zeros((N, K_x), device=x.device, dtype=x.dtype)

    NUM_SMS = num_sms

    # TODO  - hardcoded for now...but should set TMA flags based on hardware support
    USE_TMA_LOAD = True
    USE_TMA_STORE = True

    # Set up TMA descriptors or direct pointers
    if USE_TMA_LOAD or USE_TMA_STORE:
        desc_helper = TmaDescriptorHelper(tma_size=tma_size)

        if USE_TMA_LOAD:
            desc_helper.init_tma_descriptor("x")
            desc_helper.init_tma_descriptor("grad_output")
            x_desc = desc_helper.get_tma_descriptor_kernel_param("x")
            grad_output_desc = desc_helper.get_tma_descriptor_kernel_param(
                "grad_output"
            )
        else:
            x_desc = x
            grad_output_desc = grad_output

        if USE_TMA_STORE:
            desc_helper.init_tma_descriptor("grad_w")
            workspace = desc_helper.get_tma_descriptor_kernel_param("grad_w")
        else:
            workspace = torch.empty(1, device=x.device, dtype=torch.uint8)
    else:
        # If not using TMA, just use the tensors directly
        x_desc = x
        grad_output_desc = grad_output
        workspace = torch.empty(1, device=x.device, dtype=torch.uint8)

    # M_BUCKET for grid size
    M_BUCKET = triton.next_power_of_2(M_total)

    # Define grid for kernel launch
    def grid(META):
        if USE_TMA_LOAD or USE_TMA_STORE:
            if USE_TMA_LOAD:
                desc_helper.fill_2d_tma_descriptor(
                    "x",
                    x.data_ptr(),
                    M_total,
                    K_x,
                    META["BLOCK_SIZE_M"],
                    META["BLOCK_SIZE_K"],
                    x.element_size(),
                )

                desc_helper.fill_2d_tma_descriptor(
                    "grad_output",
                    grad_output.data_ptr(),
                    M_total,
                    N,
                    META["BLOCK_SIZE_M"],
                    META["BLOCK_SIZE_N"],
                    grad_output.element_size(),
                )

            if USE_TMA_STORE:
                desc_helper.fill_2d_tma_descriptor(
                    "grad_w",
                    grad_w.data_ptr(),
                    N,
                    K_x,
                    META["BLOCK_SIZE_N"],
                    META["BLOCK_SIZE_K"],
                    grad_w.element_size(),
                )

        # Return grid size - one block per SM for balanced work distribution
        return (NUM_SMS,)

    # Launch the optimized kernel
    _kernel_mg_dw_tma[grid](
        x_desc,
        grad_output_desc,
        grad_w,
        workspace,
        m_sizes,
        G,
        M_BUCKET,
        N,
        K_x,
        NUM_SMS,
        USE_TMA_LOAD,
        USE_TMA_STORE,
        TMA_SIZE=tma_size,
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
        return grad_x, grad_w, None, None


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
