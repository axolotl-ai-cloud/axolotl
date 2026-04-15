"""
Fused RMSNorm + RoPE Triton kernel for Gemma 4.

Fuses three operations into one kernel launch:
  1. RMSNorm: x_norm = (x / sqrt(mean(x^2) + eps)) * weight
  2. RoPE:    y = x_norm * cos + rotate_half(x_norm) * sin
  3. (optional) RMSNorm without scale (for v_norm)

This eliminates two intermediate tensor materializations per Q/K path;
churn from rotate_half / apply_rotary_pos_emb.

Shapes:
  X:      (rows, head_dim)  — flattened from (batch, seq_len, num_heads, head_dim)
  W:      (head_dim,)       — RMSNorm weight (None for with_scale=False)
  cos:    (rows, head_dim)  — flattened from (batch, seq_len, 1, head_dim) after broadcast
  sin:    (rows, head_dim)  — same as cos
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


def rms_norm_rope_forward(X, W, cos, sin, eps, n_heads, n_rot):
    """
    Args:
        X:   (B*S*H, head_dim) — contiguous, flattened from (B, S, H, D)
        W:   (head_dim,) or None — RMSNorm weight
        cos: (B*S, n_rot) — position embeddings (broadcast across heads)
        sin: (B*S, n_rot) — position embeddings (broadcast across heads)
        eps: float
        n_heads: int — number of attention heads (for cos/sin indexing)
        n_rot: int — rotary dim (== head_dim for full rotary, < head_dim for
            partial rotary). Must be even and ``<= head_dim``.
    Returns:
        Y, X_saved, RSTD, BLOCK_SIZE, num_warps
    """
    n_rows, n_cols = X.shape
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    has_weight = W is not None

    Y = torch.empty_like(X)
    RSTD = torch.empty(n_rows, dtype=torch.float32, device=X.device)

    _rms_norm_rope_forward_kernel[(n_rows,)](
        Y,
        Y.stride(0),
        X,
        X.stride(0),
        W if has_weight else X,  # dummy pointer when no weight
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
        HAS_WEIGHT=has_weight,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return Y, X, RSTD, BLOCK_SIZE, num_warps


def rms_norm_rope_backward(
    dY, X, W, cos, sin, RSTD, n_heads, n_rot, BLOCK_SIZE, num_warps
):
    n_rows, n_cols = dY.shape
    has_weight = W is not None

    sm_count = torch.cuda.get_device_properties(X.device).multi_processor_count
    rows_per_program = math.ceil(n_rows / sm_count)

    dX = torch.empty_like(X)

    if has_weight:
        _dW = torch.empty((sm_count, n_cols), dtype=torch.float32, device=X.device)
    else:
        _dW = torch.empty((1, n_cols), dtype=torch.float32, device=X.device)

    _rms_norm_rope_backward_kernel[(sm_count,)](
        dY,
        dY.stride(0),
        dX,
        dX.stride(0),
        X,
        X.stride(0),
        torch_to_triton_dtype[X.dtype],
        W if has_weight else X,  # dummy
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
        HAS_WEIGHT=has_weight,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    dW = _dW.sum(dim=0).to(W.dtype) if has_weight else None
    return dX, dW


class FusedRMSNormRoPEFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, X, W, cos, sin, eps, n_heads, n_rot):
        """
        X:    (B*S*H, head_dim)
        W:    (head_dim,) or None
        cos:  (B*S, n_rot) — broadcast across heads
        sin:  (B*S, n_rot) — broadcast across heads
        n_heads: int
        n_rot:   int — rotary dim (<= head_dim)
        """
        Y, X_saved, RSTD, BLOCK_SIZE, num_warps = rms_norm_rope_forward(
            X,
            W,
            cos,
            sin,
            eps,
            n_heads,
            n_rot,
        )
        ctx.eps = eps
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.n_heads = n_heads
        ctx.n_rot = n_rot
        ctx.has_weight = W is not None
        ctx.save_for_backward(X_saved, W, cos, sin, RSTD)
        return Y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dY):
        X, W, cos, sin, RSTD = ctx.saved_tensors
        dX, dW = rms_norm_rope_backward(
            dY,
            X,
            W,
            cos,
            sin,
            RSTD,
            ctx.n_heads,
            ctx.n_rot,
            ctx.BLOCK_SIZE,
            ctx.num_warps,
        )
        return dX, dW, None, None, None, None, None


def fused_rms_norm_rope(x, weight, cos, sin, eps=1e-6):
    """
    Apply fused RMSNorm + (partial) RoPE.

    Args:
        x:      (batch, seq_len, num_heads, head_dim) — after projection + view
        weight: (head_dim,) — RMSNorm weight, or None for no-scale norm
        cos:    (batch, seq_len, n_rot) — from RotaryEmbedding. ``n_rot``
                must be even and ``<= head_dim``. When ``n_rot < head_dim``
                the trailing ``head_dim - n_rot`` columns are RMSNorm-only
                (partial-rotary pass-through), matching stock Gemma 4 with
                ``partial_rotary_factor < 1.0``.
        sin:    (batch, seq_len, n_rot) — same shape as ``cos``
        eps:    float — RMSNorm epsilon

    Returns:
        y: (batch, seq_len, num_heads, head_dim) — normalized + rotated
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

    # Flatten to 2D: (B*S*H, D)
    x_flat = x.reshape(-1, D).contiguous()
    # cos/sin may broadcast over the batch dim (e.g. (1, S, n_rot) when
    # all sequences share the same rotary positions). The kernel needs a
    # dense (B*S, n_rot) buffer so that row_idx // n_heads maps cleanly
    # onto a single (b, s) pair, so expand-then-contiguous to materialize
    # the per-batch broadcast. Expand is a no-op when B == cos.shape[0].
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

    y_flat = FusedRMSNormRoPEFunction.apply(
        x_flat, weight, cos_flat, sin_flat, eps, H, n_rot
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


class FusedRMSNormNoScaleFunction(torch.autograd.Function):
    """RMSNorm without learnable scale — used for Gemma4's v_norm."""

    @staticmethod
    @ensure_contiguous
    def forward(ctx, X, eps):
        n_rows, n_cols = X.shape
        BLOCK_SIZE, num_warps = calculate_settings(n_cols)
        Y = torch.empty_like(X)
        RSTD = torch.empty(n_rows, dtype=torch.float32, device=X.device)

        _rms_norm_forward_kernel[(n_rows,)](
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
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.save_for_backward(X, RSTD)
        ctx.n_cols = n_cols
        return Y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dY):
        X, RSTD = ctx.saved_tensors
        n_rows = X.shape[0]
        dX = torch.empty_like(X)
        _rms_norm_noscale_backward_kernel[(n_rows,)](
            dY,
            dY.stride(0),
            dX,
            dX.stride(0),
            X,
            X.stride(0),
            torch_to_triton_dtype[X.dtype],
            RSTD,
            RSTD.stride(0),
            ctx.n_cols,
            BLOCK_SIZE=ctx.BLOCK_SIZE,
            num_warps=ctx.num_warps,
        )
        return dX, None


def fused_rms_norm_noscale(x, eps=1e-6):
    """
    RMSNorm without scale for v_norm.

    Args:
        x: (batch, seq_len, num_heads, head_dim)
    Returns:
        y: same shape, normalized
    """
    shape = x.shape
    x_flat = x.reshape(-1, shape[-1])
    y_flat = FusedRMSNormNoScaleFunction.apply(x_flat, eps)
    return y_flat.view(shape)
