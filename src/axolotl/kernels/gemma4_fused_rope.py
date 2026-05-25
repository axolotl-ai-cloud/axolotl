"""Fused RMSNorm + (partial) RoPE Triton kernel for Gemma 4 / Qwen3 Q/K paths."""

import math
import operator

import torch
import triton
import triton.language as tl
from liger_kernel.ops.utils import (
    calculate_settings,
    compare_version,
    torch_to_triton_dtype,
)
from liger_kernel.utils import is_npu_available
from torch.library import triton_op, wrap_triton

if compare_version("triton", operator.ge, "3.0.0") and not is_npu_available():
    try:
        from triton.language.extra.libdevice import rsqrt
    except ModuleNotFoundError:
        from triton.language.extra.cuda.libdevice import rsqrt
else:
    from triton.language.math import rsqrt


@triton.jit
def _rms_norm_rope_forward_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,
    COS_ptr,
    COS_row_stride,
    SIN_ptr,
    SIN_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    n_cols,
    n_rot,
    n_heads,
    eps,
    HAS_WEIGHT: tl.constexpr,
    UNIT_OFFSET: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused forward:
      x_norm = x / rms(x) [* weight]   (RMSNorm, full n_cols)
      y[..., :n_rot]  = rope(x_norm[..., :n_rot])
      y[..., n_rot:]  = x_norm[..., n_rot:]   (pass-through for partial rotary)

    rotate_half swaps first/second halves and negates the first, restricted
    to the rotary span [0, n_rot):
      rotate_half([a, b]) = [-b, a]   where len(a) = len(b) = n_rot/2

    For the partial-rotary pass-through region we load cos with default 1.0
    and sin with default 0.0 outside [0, n_rot), so the same formula
    `Y = X_norm * cos + X_rot_norm * sin` collapses to `Y = X_norm`.

    cos/sin are indexed by row_idx // n_heads to handle per-head broadcast
    (cos/sin have shape (B*S, n_rot) while X has shape (B*S*H, n_cols)).
    """
    row_idx = tl.program_id(0).to(tl.int64)
    # cos/sin row: divide by n_heads since cos/sin are (B*S, n_rot)
    cs_row_idx = row_idx // n_heads
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    rot_mask_col = col_offsets < n_rot
    half_rot = n_rot // 2

    # Load input row
    X_row = tl.load(X_ptr + row_idx * X_row_stride + col_offsets, mask=mask, other=0)
    X_dtype = X_row.dtype
    X_fp32 = X_row.to(tl.float32)

    # RMSNorm: compute 1/rms over the full row (rotary + pass-through)
    mean_sq = tl.sum(X_fp32 * X_fp32, axis=0) / n_cols
    rstd = rsqrt(mean_sq + eps)
    tl.store(RSTD_ptr + row_idx * RSTD_row_stride, rstd)

    # Normalize
    X_norm = X_fp32 * rstd

    # Apply weight if present (with_scale=True)
    if HAS_WEIGHT:
        W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
        if UNIT_OFFSET:
            X_norm = X_norm * (W_row + 1.0)
        else:
            X_norm = X_norm * W_row

    # RoPE: load cos/sin (broadcast across heads). For col >= n_rot we get
    # cos=1, sin=0 so the formula leaves X_norm untouched.
    cos_row = tl.load(
        COS_ptr + cs_row_idx * COS_row_stride + col_offsets,
        mask=rot_mask_col,
        other=1.0,
    ).to(tl.float32)
    sin_row = tl.load(
        SIN_ptr + cs_row_idx * SIN_row_stride + col_offsets,
        mask=rot_mask_col,
        other=0.0,
    ).to(tl.float32)

    # rotate_half within [0, n_rot):
    #   for col < half_rot:  take -X_norm[col + half_rot]
    #   for col in [half_rot, n_rot): take  X_norm[col - half_rot]
    # For col >= n_rot the rotation is irrelevant (sin = 0 zeros it out).
    rot_offsets = tl.where(
        col_offsets < half_rot, col_offsets + half_rot, col_offsets - half_rot
    )
    rot_load_mask = (rot_offsets < n_cols) & rot_mask_col
    X_rot = tl.load(
        X_ptr + row_idx * X_row_stride + rot_offsets, mask=rot_load_mask, other=0
    ).to(tl.float32)
    # Re-normalize the rotated values
    X_rot_norm = X_rot * rstd
    if HAS_WEIGHT:
        W_rot = tl.load(W_ptr + rot_offsets, mask=rot_load_mask, other=0).to(tl.float32)
        if UNIT_OFFSET:
            X_rot_norm = X_rot_norm * (W_rot + 1.0)
        else:
            X_rot_norm = X_rot_norm * W_rot

    # Negate the first half (rotate_half negates x2, which becomes the first half)
    sign = tl.where(col_offsets < half_rot, -1.0, 1.0)
    X_rot_norm = X_rot_norm * sign

    # Final RoPE: y = x_norm * cos + rotate_half(x_norm) * sin
    Y_row = X_norm * cos_row + X_rot_norm * sin_row

    tl.store(
        Y_ptr + row_idx * Y_row_stride + col_offsets,
        Y_row.to(X_dtype),
        mask=mask,
    )


@triton.jit
def _rms_norm_rope_backward_kernel(
    dY_ptr,
    dY_row_stride,
    dX_ptr,
    dX_row_stride,
    X_ptr,
    X_row_stride,
    X_dtype: tl.constexpr,
    W_ptr,
    COS_ptr,
    COS_row_stride,
    SIN_ptr,
    SIN_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    dW_ptr,
    dW_row_stride,
    n_rows,
    n_cols,
    n_rot,
    n_heads,
    rows_per_program,
    HAS_WEIGHT: tl.constexpr,
    UNIT_OFFSET: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Backward for Y = RoPE(RMSNorm(X, W)) with optional partial rotary
    (`n_rot <= n_cols`).

    For col < n_rot the standard RoPE adjoint applies. For col >= n_rot the
    output is just the normalized row, so dN[col] = dY[col] (achieved by
    loading cos with default 1.0 and forcing the rotate-half contribution
    to zero outside the rotary span).

    cos/sin indexed by row_idx // n_heads for per-head broadcast.
    """
    row_block_id = tl.program_id(0).to(tl.int64)
    row_start = row_block_id * rows_per_program
    row_end = min((row_block_id + 1) * rows_per_program, n_rows)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    rot_mask_col = col_offsets < n_rot
    half_rot = n_rot // 2

    dW_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    if HAS_WEIGHT:
        W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0).to(tl.float32)

    for row_idx in range(row_start, row_end):
        cs_row_idx = row_idx // n_heads

        dY_row = tl.load(
            dY_ptr + row_idx * dY_row_stride + col_offsets, mask=mask, other=0
        ).to(tl.float32)
        X_row = tl.load(
            X_ptr + row_idx * X_row_stride + col_offsets, mask=mask, other=0
        ).to(tl.float32)
        rstd = tl.load(RSTD_ptr + row_idx * RSTD_row_stride)

        cos_row = tl.load(
            COS_ptr + cs_row_idx * COS_row_stride + col_offsets,
            mask=rot_mask_col,
            other=1.0,
        ).to(tl.float32)

        # dN = dY * cos + rotate_half^T(dY * sin)   (within the rotary span)
        # rotate_half^T([a, b]) = [b, -a]  (adjoint of rotate_half)
        #
        # For col >= n_rot the formula must collapse to dN = dY (since the
        # forward is just a pass-through). cos defaults to 1.0 above; the
        # rotate-half contribution is masked to zero below.
        rot_offsets = tl.where(
            col_offsets < half_rot, col_offsets + half_rot, col_offsets - half_rot
        )
        rot_load_mask = (rot_offsets < n_cols) & rot_mask_col
        dY_rot = tl.load(
            dY_ptr + row_idx * dY_row_stride + rot_offsets,
            mask=rot_load_mask,
            other=0,
        ).to(tl.float32)
        sin_rot = tl.load(
            SIN_ptr + cs_row_idx * SIN_row_stride + rot_offsets,
            mask=rot_load_mask,
            other=0,
        ).to(tl.float32)

        adj_sign = tl.where(col_offsets < half_rot, 1.0, -1.0)
        rotate_term = dY_rot * sin_rot * adj_sign
        # Zero out rotate-half contribution outside the rotary span.
        rotate_term = tl.where(rot_mask_col, rotate_term, 0.0)
        dN = dY_row * cos_row + rotate_term

        # Pre-weight normalized: n = rstd * x
        n = X_row * rstd

        if HAS_WEIGHT:
            dW_acc += dN * n
            if UNIT_OFFSET:
                dm = dN * (W_row + 1.0)
            else:
                dm = dN * W_row
        else:
            dm = dN

        # RMSNorm backward: dX = rstd * (dm - (1/n_cols) * rstd^2 * dot(dm, X) * X)
        dot_dm_x = tl.sum(dm * X_row, axis=0)
        dX_row = rstd * (dm - (1.0 / n_cols) * rstd * rstd * dot_dm_x * X_row)

        tl.store(
            dX_ptr + row_idx * dX_row_stride + col_offsets,
            dX_row.to(X_dtype),
            mask=mask,
        )

    if HAS_WEIGHT:
        tl.store(
            dW_ptr + row_block_id * dW_row_stride + col_offsets,
            dW_acc,
            mask=mask,
        )


@triton_op("axolotl::fused_rms_norm_rope_fwd", mutates_args=())
def _fused_rms_norm_rope_fwd(
    X: torch.Tensor,
    W: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
    n_heads: int,
    n_rot: int,
    unit_offset: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns ``(Y, RSTD)``; ``wrap_triton`` keeps it ``torch.compile``-safe."""
    n_rows, n_cols = X.shape
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    Y = torch.empty_like(X)
    RSTD = torch.empty(n_rows, dtype=torch.float32, device=X.device)
    wrap_triton(_rms_norm_rope_forward_kernel)[(n_rows,)](
        Y,
        Y.stride(0),
        X,
        X.stride(0),
        W,
        cos,
        cos.stride(0),
        sin,
        sin.stride(0),
        RSTD,
        RSTD.stride(0),
        n_cols,
        n_rot,
        n_heads,
        eps,
        HAS_WEIGHT=True,
        UNIT_OFFSET=unit_offset,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return Y, RSTD


@triton_op("axolotl::fused_rms_norm_rope_bwd", mutates_args=())
def _fused_rms_norm_rope_bwd(
    dY: torch.Tensor,
    X: torch.Tensor,
    W: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    RSTD: torch.Tensor,
    n_heads: int,
    n_rot: int,
    unit_offset: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns ``(dX, dW)``."""
    n_rows, n_cols = dY.shape
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    sm_count = torch.cuda.get_device_properties(X.device).multi_processor_count
    rows_per_program = math.ceil(n_rows / sm_count)
    dX = torch.empty_like(X)
    _dW = torch.empty((sm_count, n_cols), dtype=torch.float32, device=X.device)
    wrap_triton(_rms_norm_rope_backward_kernel)[(sm_count,)](
        dY,
        dY.stride(0),
        dX,
        dX.stride(0),
        X,
        X.stride(0),
        torch_to_triton_dtype[X.dtype],
        W,
        cos,
        cos.stride(0),
        sin,
        sin.stride(0),
        RSTD,
        RSTD.stride(0),
        _dW,
        _dW.stride(0),
        n_rows,
        n_cols,
        n_rot,
        n_heads,
        rows_per_program,
        HAS_WEIGHT=True,
        UNIT_OFFSET=unit_offset,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    dW = _dW.sum(dim=0).to(W.dtype)
    return dX, dW


def _fused_rms_norm_rope_setup_context(ctx, inputs, output):
    X, W, cos, sin, _eps, n_heads, n_rot, unit_offset = inputs
    _, RSTD = output
    ctx.save_for_backward(X, W, cos, sin, RSTD)
    ctx.n_heads = n_heads
    ctx.n_rot = n_rot
    ctx.unit_offset = unit_offset


def _fused_rms_norm_rope_backward(ctx, grad_Y, grad_RSTD):
    X, W, cos, sin, RSTD = ctx.saved_tensors
    grad_Y = grad_Y.contiguous()
    dX, dW = _fused_rms_norm_rope_bwd(
        grad_Y, X, W, cos, sin, RSTD, ctx.n_heads, ctx.n_rot, ctx.unit_offset
    )
    return dX, dW, None, None, None, None, None, None


_fused_rms_norm_rope_fwd.register_autograd(
    _fused_rms_norm_rope_backward,
    setup_context=_fused_rms_norm_rope_setup_context,
)


def fused_rms_norm_rope(x, weight, cos, sin, eps=1e-6, unit_offset=False):
    """
    Apply fused RMSNorm + (partial) RoPE.

    Shapes:
        x:      (B, S, H, D) — post-projection
        weight: (D,) — required; use ``fused_rms_norm_noscale`` for the no-weight variant
        cos:    (B, S, n_rot) — ``n_rot`` must be even and ``<= D``; trailing
                ``D - n_rot`` columns are RMSNorm-only (partial rotary).
        sin:    (B, S, n_rot)

    ``unit_offset=True`` scales by ``(weight + 1.0)`` (Gemma-style).
    """
    shape = x.shape  # (B, S, H, D)
    B, S, H, D = shape
    n_rot = cos.shape[-1]
    if sin.shape[-1] != n_rot:
        raise ValueError(
            f"cos and sin must have the same last dim, got cos={cos.shape[-1]} "
            f"sin={sin.shape[-1]}"
        )
    if n_rot > D:
        raise ValueError(f"rotary dim ({n_rot}) cannot exceed head_dim ({D})")
    if n_rot % 2 != 0:
        raise ValueError(f"rotary dim must be even, got {n_rot}")

    x_flat = x.reshape(-1, D).contiguous()
    # Kernel needs a dense (B*S, n_rot) buffer; materialize the batch-broadcast.
    if cos.shape[0] != B:
        if cos.shape[0] != 1:
            raise ValueError(
                f"cos/sin batch dim ({cos.shape[0]}) must be 1 or equal "
                f"to x batch dim ({B})"
            )
        cos = cos.expand(B, S, n_rot)
        sin = sin.expand(B, S, n_rot)
    cos_flat = cos.reshape(B * S, n_rot).contiguous()
    sin_flat = sin.reshape(B * S, n_rot).contiguous()

    y_flat, _ = _fused_rms_norm_rope_fwd(
        x_flat, weight, cos_flat, sin_flat, eps, H, n_rot, unit_offset
    )
    return y_flat.view(shape)


@triton.jit
def _rms_norm_forward_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """RMSNorm without scale weight: y = x / rms(x)"""
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    X_row = tl.load(X_ptr + row_idx * X_row_stride + col_offsets, mask=mask, other=0)
    X_dtype = X_row.dtype
    X_fp32 = X_row.to(tl.float32)

    mean_sq = tl.sum(X_fp32 * X_fp32, axis=0) / n_cols
    rstd = rsqrt(mean_sq + eps)
    tl.store(RSTD_ptr + row_idx * RSTD_row_stride, rstd)

    Y_row = X_fp32 * rstd
    tl.store(Y_ptr + row_idx * Y_row_stride + col_offsets, Y_row.to(X_dtype), mask=mask)


@triton.jit
def _rms_norm_noscale_backward_kernel(
    dY_ptr,
    dY_row_stride,
    dX_ptr,
    dX_row_stride,
    X_ptr,
    X_row_stride,
    X_dtype: tl.constexpr,
    RSTD_ptr,
    RSTD_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Backward for y = x * rstd (no weight)."""
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dY_row = tl.load(
        dY_ptr + row_idx * dY_row_stride + col_offsets, mask=mask, other=0
    ).to(tl.float32)
    X_row = tl.load(
        X_ptr + row_idx * X_row_stride + col_offsets, mask=mask, other=0
    ).to(tl.float32)
    rstd = tl.load(RSTD_ptr + row_idx * RSTD_row_stride)

    dot_dy_x = tl.sum(dY_row * X_row, axis=0)
    dX_row = rstd * (dY_row - (1.0 / n_cols) * rstd * rstd * dot_dy_x * X_row)

    tl.store(
        dX_ptr + row_idx * dX_row_stride + col_offsets, dX_row.to(X_dtype), mask=mask
    )


@triton_op("axolotl::fused_rms_norm_noscale_fwd", mutates_args=())
def _fused_rms_norm_noscale_fwd(
    X: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns ``(Y, RSTD)``."""
    n_rows, n_cols = X.shape
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    Y = torch.empty_like(X)
    RSTD = torch.empty(n_rows, dtype=torch.float32, device=X.device)
    wrap_triton(_rms_norm_forward_kernel)[(n_rows,)](
        Y,
        Y.stride(0),
        X,
        X.stride(0),
        RSTD,
        RSTD.stride(0),
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return Y, RSTD


@triton_op("axolotl::fused_rms_norm_noscale_bwd", mutates_args=())
def _fused_rms_norm_noscale_bwd(
    dY: torch.Tensor, X: torch.Tensor, RSTD: torch.Tensor
) -> torch.Tensor:
    n_rows, n_cols = X.shape
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    dX = torch.empty_like(X)
    wrap_triton(_rms_norm_noscale_backward_kernel)[(n_rows,)](
        dY,
        dY.stride(0),
        dX,
        dX.stride(0),
        X,
        X.stride(0),
        torch_to_triton_dtype[X.dtype],
        RSTD,
        RSTD.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return dX


def _fused_rms_norm_noscale_setup_context(ctx, inputs, output):
    X, _eps = inputs
    _, RSTD = output
    ctx.save_for_backward(X, RSTD)


def _fused_rms_norm_noscale_backward(ctx, grad_Y, grad_RSTD):
    X, RSTD = ctx.saved_tensors
    grad_Y = grad_Y.contiguous()
    dX = _fused_rms_norm_noscale_bwd(grad_Y, X, RSTD)
    return dX, None


_fused_rms_norm_noscale_fwd.register_autograd(
    _fused_rms_norm_noscale_backward,
    setup_context=_fused_rms_norm_noscale_setup_context,
)


def fused_rms_norm_noscale(x, eps=1e-6):
    """RMSNorm without a learned scale (used for v_norm)."""
    shape = x.shape
    x_flat = x.reshape(-1, shape[-1]).contiguous()
    y_flat, _ = _fused_rms_norm_noscale_fwd(x_flat, eps)
    return y_flat.view(shape)
