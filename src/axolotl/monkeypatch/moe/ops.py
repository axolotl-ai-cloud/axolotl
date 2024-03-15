"""
Adapted from:
https://github.com/shawntan/scattermoe
https://arxiv.org/abs/2403.08245
"""

import torch
import triton
import triton.language as tl
from torch.nn import functional as F

BLOCK_M = 128

@torch.jit.script
def flatten_and_sort(expert_idxs:torch.Tensor):
    flattened_expert_idxs = expert_idxs.flatten()
    sorted_expert_idxs, sorted_scattered_idxs = torch.sort(flattened_expert_idxs)
    return sorted_expert_idxs, sorted_scattered_idxs

@torch.jit.script
def padded_block_indices(sorted_experts_idxs: torch.Tensor, k: int, N_BLOCK_SIZE: int=BLOCK_M) :
    expert_counts = torch.bincount(sorted_experts_idxs, minlength=k)
    padded_block_counts = ((expert_counts - 1) // N_BLOCK_SIZE) + 1
    padded_expert_block_end = padded_block_counts.cumsum(-1)
    expert_boundaries_end = expert_counts.cumsum(-1)
    expert_boundaries_start = expert_boundaries_end - expert_counts
    padded_expert_block_start = padded_expert_block_end - padded_block_counts
    block_idxs = torch.arange(padded_expert_block_end[-1],
                              dtype=sorted_experts_idxs.dtype,
                              device=sorted_experts_idxs.device)
    block_mask = (
        (block_idxs[:, None] < padded_expert_block_start) |
        (block_idxs[:, None] >= padded_expert_block_end)
    )
    expanded_block_idxs = (
        N_BLOCK_SIZE * (block_idxs[:, None] - padded_expert_block_start) +
        expert_boundaries_start
    )
    expanded_block_idxs = expanded_block_idxs.masked_fill(block_mask, 0).sum(-1)
    return expanded_block_idxs, expert_boundaries_end



def _scatter2scatter_configs():
    return [
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
    ]

@triton.autotune(configs=_scatter2scatter_configs(), key=['M', 'N', 'K'], )
@triton.heuristics({
    "NO_K_MASK": lambda args: (args['K'] % args['BLOCK_K']) == 0,
    "NO_N_MASK": lambda args: (args['N'] % args['BLOCK_N']) == 0,
})
@triton.jit
def _scatter2scatter(
    X_ptr, stride_xm, stride_xk,
    W_ptr, stride_we, stride_wk, stride_wn,
    Y_ptr, stride_ym, stride_yn,
    grouped_idx_ptr, expert_idxs_ptr, block_start_idx_ptr,
    FAN_OUT: tl.constexpr,
    M: tl.constexpr, K: tl.constexpr, N: tl.constexpr, E: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ACC_TYPE: tl.constexpr,
    OUT_M: tl.constexpr,
    allow_tf32: tl.constexpr,
    x_grouped: tl.constexpr, y_grouped: tl.constexpr,
    NO_K_MASK: tl.constexpr, NO_N_MASK: tl.constexpr
):
    pid = tl.program_id(axis=0)

    N_BLOCK_COUNT = tl.cdiv(N, BLOCK_N)
    M_block_id = pid // N_BLOCK_COUNT
    N_block_id = pid % N_BLOCK_COUNT
    M_range = tl.arange(0, BLOCK_M)
    block_start_idx = tl.load(block_start_idx_ptr + M_block_id)
    # M_block = tl.max_contiguous((block_start_idx + M_range) % OUT_M, BLOCK_M)
    M_block = tl.max_contiguous(block_start_idx + M_range, BLOCK_M)
    E_idxs = tl.load(expert_idxs_ptr + M_block, mask=M_block < (FAN_OUT * M), other=E)
    E_idx = tl.min(E_idxs)
    E_mask = E_idxs == E_idx
    M_idx = tl.load(grouped_idx_ptr + M_block, mask=E_mask, other=0)
    if x_grouped:
        M_in_idx = M_block
    else:
        M_in_idx = M_idx // FAN_OUT

    if y_grouped:
        M_out_idx = M_block
    else:
        M_out_idx = M_idx

    K_block = tl.arange(0, BLOCK_K)

    N_block = N_block_id * BLOCK_N  + tl.arange(0, BLOCK_N)
    N_mask = N_block < N
    # N_block = tl.max_contiguous(tl.multiple_of(N_block % N, BLOCK_N), BLOCK_N)
    # N_block = N_block_id * BLOCK_N + tl.arange(0, BLOCK_N)

    X_blk_ptrs = X_ptr + M_in_idx[:, None] * stride_xm + K_block[None, :] * stride_xk
    W_blk_ptrs = W_ptr + K_block[:, None] * stride_wk + N_block[None, :] * stride_wn + E_idx * stride_we

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    iters = tl.cdiv(K, BLOCK_K)
    for K_block_id in range(0, iters):
        if NO_K_MASK:
            x = tl.load(X_blk_ptrs, mask=E_mask[:, None])
            if NO_N_MASK:
                w = tl.load(W_blk_ptrs)
            else:
                w = tl.load(W_blk_ptrs, mask=N_mask[None, :])
        else:
            K_mask = (K_block_id * BLOCK_K + K_block) < K
            x = tl.load(X_blk_ptrs, mask=E_mask[:, None] & K_mask[None, :])
            w = tl.load(W_blk_ptrs, mask=K_mask[:, None] & N_mask[None, :])
        X_blk_ptrs += BLOCK_K * stride_xk
        W_blk_ptrs += BLOCK_K * stride_wk
        acc += tl.dot(x, w, allow_tf32=allow_tf32, out_dtype=ACC_TYPE)

    Y_blk_ptrs = Y_ptr + (M_out_idx[:, None] * stride_ym + N_block[None, :] * stride_yn)
    tl.store(Y_blk_ptrs, acc, mask=E_mask[:, None] & N_mask[None, :])

def scatter2scatter(X, W, sorted_expert_idxs, sorted_scattered_idxs, k,
                    padded_block_idxs, x_grouped=False, y_grouped=False,
                    out=None):
    assert sorted_scattered_idxs.size(0) == sorted_expert_idxs.size(0)
    assert sorted_scattered_idxs.size(0) == X.size(0) * k
    # Pre-kernel setup
    x_dim = X.size(-1)
    y_dim = W.size(-1)
    L_scattered = sorted_expert_idxs.size(0)
    if out is None:
        O = torch.empty((L_scattered, y_dim), device=X.device, dtype=X.dtype)
    else:
        assert out.size(0) == L_scattered and out.size(1) == y_dim
        O = out

    def grid(META):
        grid_num = (
            padded_block_idxs.size(0) *
            triton.cdiv(META['N'], META['BLOCK_N']),
        )
        return grid_num
    """
    print("X", X.size(), X.stride(),
          "W", W.size(), W.stride(),
          "O", O.size(), O.stride(),
          "sorted_idxs", sorted_scattered_idxs.size(),
          "FAN_OUT", k,
          "BLOCK_M", BLOCK_M,
          "grouped", (x_grouped, y_grouped))
    """
    _scatter2scatter[grid](
        # X_ptr, stride_xm, stride_xk,
        X, X.stride(0), X.stride(1),
        # W_ptr, stride_we, stride_wk, stride_wn,
        W, W.stride(0), W.stride(1), W.stride(2),
        # Y_ptr, stride_ym, stride_yn,
        O, O.stride(0), O.stride(1),
        grouped_idx_ptr=sorted_scattered_idxs,
        expert_idxs_ptr=sorted_expert_idxs,
        block_start_idx_ptr=padded_block_idxs,
        FAN_OUT=k,
        M=X.size(0),
        K=X.size(1),
        N=O.size(1), E=W.size(0),
        BLOCK_M=BLOCK_M,
        ACC_TYPE=tl.float32,
        OUT_M=O.size(0),
        allow_tf32=True,
        x_grouped=x_grouped, y_grouped=y_grouped,
    )
    return O


def _config_XtY():
    return [
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128, 'BLOCK_M': 32}, num_stages=4, num_warps=4),
    ]

def group_bwd_W(DY, X, expert_offsets, E):
    DWt = torch.zeros((E, DY.size(-1), X.size(-1)), device=DY.device, dtype=DY.dtype)
    DW = DWt.permute(0, 2, 1)
    def grid(META):
        grid = (
            E * triton.cdiv(META['K'], META['BLOCK_K']),
            triton.cdiv(META['N'], META['BLOCK_N']),
        )
        return grid
    _groupXtY[grid](
        # DY_ptr, stride_dym, stride_dyk,
        DY, DY.stride(0), DY.stride(1),
        # X_ptr, stride_xm, stride_xn,
        X, X.stride(0), X.stride(1),
        # DW_ptr, stride_dwe, stride_dwk, stride_dwn,
        DW, DW.stride(0), DW.stride(1), DW.stride(2),
        # expert_offsets_ptr,
        expert_offsets,
        # K: tl.constexpr, N: tl.constexpr,
        M=DY.size(0), N=DY.size(-1), K=X.size(-1),
        # ACC_TYPE: tl.constexpr,
        ACC_TYPE=tl.float32,
        allow_tf32=True
    )
    return DW

@triton.autotune(configs=_config_XtY(), key=['M', 'N', 'K'], )
@triton.heuristics({
    "NO_K_MASK": lambda args: (args['K'] % args['BLOCK_K']) == 0,
    "NO_N_MASK": lambda args: (args['N'] % args['BLOCK_N']) == 0,
})
@triton.jit
def _groupXtY(
    DY_ptr, stride_dym, stride_dyk,
    X_ptr, stride_xm, stride_xn,
    DW_ptr, stride_dwe, stride_dwk, stride_dwn,
    expert_offsets_ptr,
    M: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ACC_TYPE: tl.constexpr,
    allow_tf32: tl.constexpr,
    NO_K_MASK: tl.constexpr, NO_N_MASK: tl.constexpr
):
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)
    num0 = tl.num_programs(0)
    num1 = tl.num_programs(1)
    pid1, pid0 = tl.swizzle2d(pid1, pid0, num1, num0, 128)

    K_BLOCK_COUNT = tl.cdiv(K, BLOCK_K)
    E_idx = pid0 // K_BLOCK_COUNT
    K_block_id = pid0 % K_BLOCK_COUNT
    N_block_id = pid1

    if E_idx == 0:
        start_idx = 0
    else:
        start_idx = tl.load(expert_offsets_ptr + E_idx - 1).to(tl.int32)
    end_idx = tl.load(expert_offsets_ptr + E_idx).to(tl.int32)

    if end_idx > start_idx:
        M_block = tl.max_contiguous(start_idx + tl.arange(0, BLOCK_M), BLOCK_M)

        K_block = K_block_id * BLOCK_K + tl.arange(0, BLOCK_K)
        K_mask = K_block < K
        K_block = tl.max_contiguous(tl.multiple_of(K_block % K, BLOCK_K), BLOCK_K)

        N_block = N_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
        N_mask = N_block < N
        N_block = tl.max_contiguous(tl.multiple_of(N_block % N, BLOCK_N), BLOCK_N)

        M_idxs = M_block
        xt_blk_ptrs = X_ptr + K_block[:, None] * stride_xn + M_idxs[None, :] * stride_xm
        dy_blk_ptrs = DY_ptr + M_idxs[:, None] * stride_dym + N_block[None, :] * stride_dyk

        acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=ACC_TYPE)
        iters = tl.cdiv(end_idx - start_idx, BLOCK_M)
        for i in range(0, iters):
            M_mask = (i * BLOCK_M + M_block) < end_idx
            if NO_K_MASK:
                xt = tl.load(xt_blk_ptrs, mask=M_mask[None, :])
            else:
                xt = tl.load(xt_blk_ptrs, mask=K_mask[:, None] & M_mask[None, :])
            if NO_N_MASK:
                dy = tl.load(dy_blk_ptrs, mask=M_mask[:, None])
            else:
                dy = tl.load(dy_blk_ptrs, mask=M_mask[:, None] & N_mask[None, :])
            acc += tl.dot(xt, dy, out_dtype=ACC_TYPE, allow_tf32=allow_tf32)
            xt_blk_ptrs += BLOCK_M * stride_xm
            dy_blk_ptrs += BLOCK_M * stride_dym


        DW_blk_ptrs = DW_ptr + E_idx * stride_dwe + K_block[:, None] * stride_dwk + N_block[None, :] * stride_dwn
        acc = acc.to(DW_blk_ptrs.dtype.element_ty)
        tl.store(DW_blk_ptrs, acc, mask=K_mask[:, None] & N_mask[None, :])


def _config_grouping():
    return [
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
    ]

def group(A, sorted_expert_idxs, coeff=None, fan_out=1, out=None):
    N = sorted_expert_idxs.size(0)
    K = A.size(1)
    assert A.size(0) * fan_out == N
    if out is not None:
        Y = out
    else:
        Y = torch.empty((N, K), dtype=A.dtype, device=A.device)
        # print("grp init:", Y.size())
    def grid(META):
        grid_num = (triton.cdiv(META['N'], META['BLOCK_N']),)
        return grid_num
    _group[grid](
        # A_ptr, stride_an, stride_ai,
        A, A.stride(0), A.stride(1), coeff is not None, coeff, fan_out,
        # Y_ptr, stride_yn, stride_yk,
        Y, Y.stride(0), Y.stride(1),
        # grouped_idx_ptr,
        sorted_expert_idxs,
        # N: tl.constexpr, K: tl.constexpr,
        N, K
    )
    return Y

@triton.autotune(configs=_config_grouping(), key=['K'])
@triton.heuristics({
    "NO_K_MASK": lambda args: (args['K'] % args['BLOCK_K']) == 0
})
@triton.jit
def _group(
    src_ptr, stride_sn, stride_sk, has_coeff: tl.constexpr, coeff_ptr, FAN_OUT: tl.constexpr,
    tgt_ptr, stride_tn, stride_ti,
    grouped_idx_ptr,
    N: tl.constexpr, K: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NO_K_MASK: tl.constexpr
):
    pid = tl.program_id(axis=0)

    N_block_id = pid
    N_blk = N_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
    N_mask = N_blk < N
    N_blk = tl.max_contiguous(tl.multiple_of(N_blk % N, BLOCK_N), BLOCK_N)
    N_idx = tl.load(grouped_idx_ptr + N_blk, mask=N_mask, other=0)

    K_blk = tl.arange(0, BLOCK_K)
    src_blk_ptrs = src_ptr + (N_idx // FAN_OUT)[:, None] * stride_sn + K_blk[None, :] * stride_sk
    tgt_blk_ptrs = tgt_ptr + N_blk[:, None] * stride_tn + K_blk[None, :] * stride_ti

    if has_coeff:
        c = tl.load(coeff_ptr + N_idx, mask=N_mask)[:, None]

    iters = tl.cdiv(K, BLOCK_K)
    for i in range(0, iters):
        if NO_K_MASK:
            block = tl.load(src_blk_ptrs) # , mask=N_mask[:, None])
            if has_coeff:
                block *= c
            tl.store(tgt_blk_ptrs, block, mask=N_mask[:, None])

        else:
            K_mask = (i * BLOCK_K + K_blk) < K
            mask = N_mask[:, None] & K_mask[None, :]
            block = tl.load(src_blk_ptrs, mask=mask)
            if has_coeff:
                block *= c
            tl.store(tgt_blk_ptrs, block, mask=mask)

        src_blk_ptrs += BLOCK_K * stride_sk
        tgt_blk_ptrs += BLOCK_K * stride_ti