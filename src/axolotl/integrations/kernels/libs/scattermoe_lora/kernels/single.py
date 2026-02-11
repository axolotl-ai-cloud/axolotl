# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/shawntan/scattermoe
# Copyright (c) Shawn Tan and ScatterMoE Contributors
# Licensed under the Apache License, Version 2.0
# See https://github.com/shawntan/scattermoe/blob/main/LICENSE

import torch
import triton
import triton.language as tl


@triton.jit
def _single2scatter(
    X_ptr,
    stride_xm,
    stride_xk,
    W_ptr,
    stride_we,
    stride_wk,
    stride_wn,
    Y_ptr,
    stride_ym,
    stride_yn,
    expert_idxs_ptr,
    FAN_OUT: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    E: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ACC_TYPE: tl.constexpr,
):
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)

    N_block_id = pid0
    if FAN_OUT == 1:
        in_idx = pid1
    else:
        in_idx = 0
    out_idx = pid1

    K_block = tl.arange(0, BLOCK_K)
    N_block = tl.max_contiguous(
        tl.multiple_of((N_block_id * BLOCK_N + tl.arange(0, BLOCK_N)) % N, BLOCK_N),
        BLOCK_N,
    )
    E_idx = tl.load(expert_idxs_ptr + pid1)
    X_blk_ptrs = X_ptr + in_idx * stride_xm + K_block[:, None] * stride_xk
    W_blk_ptrs = (
        W_ptr
        + E_idx * stride_we
        + K_block[:, None] * stride_wk
        + N_block[None, :] * stride_wn
    )
    acc = tl.zeros((1, BLOCK_N), dtype=ACC_TYPE)
    for _K_block_id in range(0, tl.cdiv(K, BLOCK_K)):
        x = tl.load(X_blk_ptrs)
        w = tl.load(W_blk_ptrs)
        acc += tl.sum(x * w, axis=0)[None, :]
        X_blk_ptrs += BLOCK_K * stride_xk
        W_blk_ptrs += BLOCK_K * stride_wk
    Y_blk_ptrs = Y_ptr + out_idx * stride_ym + N_block[None, :] * stride_yn
    tl.store(Y_blk_ptrs, acc)


def single2scatter(X, W, expert_idxs):
    E, xdim, ydim = W.size()
    k = expert_idxs.size(1)
    assert X.size(0) == k or X.size(0) == 1
    Y = torch.empty((k, ydim), device=X.device, dtype=X.dtype)
    BLOCK_N = 128
    BLOCK_K = 128
    grid = ydim // BLOCK_N, k
    _single2scatter[grid](
        X,
        X.stride(0),
        X.stride(1),
        W,
        W.stride(0),
        W.stride(1),
        W.stride(2),
        Y,
        Y.stride(0),
        Y.stride(1),
        expert_idxs,
        FAN_OUT=Y.size(0) // X.size(0),
        K=xdim,
        N=ydim,
        E=E,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        ACC_TYPE=tl.float32,
    )
    return Y
