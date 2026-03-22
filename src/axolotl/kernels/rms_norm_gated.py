"""
Fused RMSNorm + SiLU Gate Triton kernel.

Computes: Y = (W + offset) * RMSNorm(X) * silu(G)
where RMSNorm(X) = X / sqrt(mean(X^2) + eps)
and silu(G) = G * sigmoid(G)

Used by Qwen3.5's GatedDeltaNet linear attention layers (Qwen3_5RMSNormGated).
"""

import math
import operator

import torch
import triton
import triton.language as tl
from liger_kernel.ops.utils import (
    calculate_settings,
    compare_version,
    ensure_contiguous,
    torch_to_triton_dtype,
)
from liger_kernel.utils import is_npu_available

if compare_version("triton", operator.ge, "3.0.0") and not is_npu_available():
    try:
        from triton.language.extra.libdevice import rsqrt
    except ModuleNotFoundError:
        from triton.language.extra.cuda.libdevice import rsqrt
else:
    from triton.language.math import rsqrt


@triton.jit
def _rms_norm_gated_forward_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    G_ptr,
    G_row_stride,
    W_ptr,
    W_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    n_cols,
    eps,
    offset,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Y = (W + offset) * (X / RMS(X)) * silu(G)

    All computation done in fp32 (Gemma-style), result cast to input dtype.
    """
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    X_row = tl.load(X_ptr + row_idx * X_row_stride + col_offsets, mask=mask, other=0)
    G_row = tl.load(G_ptr + row_idx * G_row_stride + col_offsets, mask=mask, other=0)
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)

    X_row_dtype = X_row.dtype

    # Cast everything to fp32
    X_fp32 = X_row.to(tl.float32)
    G_fp32 = G_row.to(tl.float32)
    W_fp32 = W_row.to(tl.float32)

    # RMS norm
    mean_sq = tl.sum(X_fp32 * X_fp32, axis=0) / n_cols
    rstd = rsqrt(mean_sq + eps)
    tl.store(RSTD_ptr + row_idx * RSTD_row_stride, rstd)

    X_norm = X_fp32 * rstd

    # SiLU gate: silu(G) = G * sigmoid(G)
    sig_G = tl.sigmoid(G_fp32)
    silu_G = G_fp32 * sig_G

    # Fused output
    Y_row = (offset + W_fp32) * X_norm * silu_G

    tl.store(
        Y_ptr + row_idx * Y_row_stride + col_offsets,
        Y_row.to(X_row_dtype),
        mask=mask,
    )


@triton.jit
def _rms_norm_gated_backward_kernel(
    dY_ptr,
    dY_row_stride,
    dX_ptr,
    dX_row_stride,
    dG_ptr,
    dG_row_stride,
    X_ptr,
    X_row_stride,
    X_dtype: tl.constexpr,
    G_ptr,
    G_row_stride,
    W_ptr,
    W_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    dW_ptr,
    dW_row_stride,
    n_rows,
    n_cols,
    offset,
    rows_per_program,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Backward for Y = (W + offset) * (X * RSTD) * silu(G)

    dW = sum_batch(dY * X_norm * silu(G))
    dG = dY * (W + offset) * X_norm * silu'(G)
       where silu'(G) = sigmoid(G) * (1 + G * (1 - sigmoid(G)))
    dX = RSTD * (m - (1/N) * RSTD^2 * dot(m, X) * X)
       where m = dY * (W + offset) * silu(G)
    """
    row_block_id = tl.program_id(0).to(tl.int64)
    row_start = row_block_id * rows_per_program
    row_end = min((row_block_id + 1) * rows_per_program, n_rows)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dW_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)
    W_row = W_row.to(tl.float32) + offset

    for row_idx in range(row_start, row_end):
        dY_row = tl.load(
            dY_ptr + row_idx * dY_row_stride + col_offsets, mask=mask, other=0.0
        )
        X_row = tl.load(
            X_ptr + row_idx * X_row_stride + col_offsets, mask=mask, other=0.0
        )
        G_row = tl.load(
            G_ptr + row_idx * G_row_stride + col_offsets, mask=mask, other=0.0
        )
        rstd_row = tl.load(RSTD_ptr + row_idx * RSTD_row_stride)

        # Cast to fp32
        dY_fp32 = dY_row.to(tl.float32)
        X_fp32 = X_row.to(tl.float32)
        G_fp32 = G_row.to(tl.float32)

        # Recompute intermediates
        X_norm = X_fp32 * rstd_row
        sig_G = tl.sigmoid(G_fp32)
        silu_G = G_fp32 * sig_G

        # dW: accumulate dY * X_norm * silu(G)
        dW_acc += dY_fp32 * X_norm * silu_G

        # dG: dY * (W + offset) * X_norm * silu'(G)
        # silu'(G) = sigmoid(G) * (1 + G * (1 - sigmoid(G)))
        silu_prime_G = sig_G * (1.0 + G_fp32 * (1.0 - sig_G))
        dG_row = dY_fp32 * W_row * X_norm * silu_prime_G
        tl.store(
            dG_ptr + row_idx * dG_row_stride + col_offsets,
            dG_row.to(X_dtype),
            mask=mask,
        )

        # dX: standard RMSNorm backward with effective gradient m = dY * W * silu(G)
        m = dY_fp32 * W_row * silu_G
        dX_row = rstd_row * m
        dX_row += rstd_row * (
            -(1.0 / n_cols) * rstd_row * rstd_row * tl.sum(m * X_fp32, axis=0) * X_fp32
        )
        tl.store(
            dX_ptr + row_idx * dX_row_stride + col_offsets,
            dX_row.to(X_dtype),
            mask=mask,
        )

    tl.store(
        dW_ptr + row_block_id * dW_row_stride + col_offsets,
        dW_acc,
        mask=mask,
    )


def rms_norm_gated_forward(X, G, W, eps, offset):
    shape = X.shape
    dim = shape[-1]
    X = X.view(-1, dim)
    G = G.view(-1, dim)
    n_rows, n_cols = X.shape
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
    RSTD = torch.empty(n_rows, dtype=torch.float32, device=X.device)

    assert X.shape[1] == W.shape[0], (
        f"Incompatible hidden size: X.shape[1]={X.shape[1]} vs W.shape[0]={W.shape[0]}"
    )
    assert X.shape == G.shape, (
        f"X and G must have same shape, got {X.shape} and {G.shape}"
    )

    _rms_norm_gated_forward_kernel[(n_rows,)](
        Y,
        Y.stride(0),
        X,
        X.stride(0),
        G,
        G.stride(0),
        W,
        W.stride(0),
        RSTD,
        RSTD.stride(0),
        n_cols,
        eps,
        offset,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return Y.view(*shape), X, G, RSTD, BLOCK_SIZE, num_warps


def rms_norm_gated_backward(dY, X, G, W, RSTD, offset, BLOCK_SIZE, num_warps):
    shape = dY.shape
    dim = shape[-1]
    dY = dY.view(-1, dim)
    n_rows, n_cols = dY.shape

    sm_count = torch.cuda.get_device_properties(X.device).multi_processor_count

    _dW = torch.empty((sm_count, n_cols), dtype=torch.float32, device=W.device)
    dX = torch.empty_like(dY)
    dG = torch.empty_like(dY)

    rows_per_program = math.ceil(n_rows / sm_count)
    grid = (sm_count,)

    _rms_norm_gated_backward_kernel[grid](
        dY,
        dY.stride(0),
        dX,
        dX.stride(0),
        dG,
        dG.stride(0),
        X,
        X.stride(0),
        torch_to_triton_dtype[X.dtype],
        G,
        G.stride(0),
        W,
        W.stride(0),
        RSTD,
        RSTD.stride(0),
        _dW,
        _dW.stride(0),
        n_rows,
        n_cols,
        offset,
        rows_per_program,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    dX = dX.view(*shape)
    dG = dG.view(*shape)
    dW = _dW.sum(dim=0).to(W.dtype)
    return dX, dG, dW


class FusedRMSNormGatedFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, X, G, W, eps, offset=0.0):
        """
        X: (B, T, H) or (BxT, H) — input hidden states
        G: (B, T, H) or (BxT, H) — gate tensor
        W: (H,) — weight parameter
        """
        Y, X, G, RSTD, BLOCK_SIZE, num_warps = rms_norm_gated_forward(
            X, G, W, eps, offset
        )
        ctx.offset = offset
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.save_for_backward(X, G, W, RSTD)
        return Y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dY):
        X, G, W, RSTD = ctx.saved_tensors
        dX, dG, dW = rms_norm_gated_backward(
            dY, X, G, W, RSTD, ctx.offset, ctx.BLOCK_SIZE, ctx.num_warps
        )
        return dX, dG, dW, None, None


class FusedRMSNormGated(torch.nn.Module):
    """
    Fused RMSNorm + SiLU Gate.

    Computes: Y = W * RMSNorm(X) * silu(G)

    Drop-in replacement for Qwen3_5RMSNormGated with matching
    init signature: __init__(hidden_size, eps=1e-6, **kwargs)
    and forward signature: forward(hidden_states, gate=None)
    """

    def __init__(self, hidden_size, eps=1e-6, offset=0.0, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.offset = offset

    def forward(self, hidden_states, gate=None):
        if gate is None:
            raise ValueError("FusedRMSNormGated requires a gate tensor")
        if hidden_states.device.type != "cuda":
            raise ValueError(
                f"FusedRMSNormGated requires CUDA tensors, got device={hidden_states.device}"
            )
        return FusedRMSNormGatedFunction.apply(
            hidden_states, gate, self.weight, self.variance_epsilon, self.offset
        )

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
