"""Fused per-token int8 activation fake-quant with λ-interpolation.

One program per token: absmax reduction, `[-127, 127]` int8 rounding and the λ blend in a
single kernel, replacing the five elementwise passes the eager path costs.
"""

from __future__ import annotations

import torch

from ...quant import ACT_QMAX, SCALE_EPS
from ._common import HAVE_TRITON, num_warps_for, require_cuda

if HAVE_TRITON:
    import triton
    import triton.language as tl

    try:
        from triton.language.extra.libdevice import div_rn, rint
    except ModuleNotFoundError:  # pragma: no cover
        from triton.language.extra.cuda.libdevice import div_rn, rint

    @triton.jit
    def _blend(x, x_f32, scale, lambda_, QMAX: tl.constexpr, LAMBDA_ONE: tl.constexpr):
        # div_rn, not `/`: only correctly-rounded division reproduces the oracle's ties
        code = rint(div_rn(x_f32, scale))
        code = tl.minimum(tl.maximum(code, -QMAX), QMAX).to(tl.int8)
        # the oracle materializes `code · s_x` in the activation dtype before blending
        deq = (code.to(tl.float32) * scale).to(x.dtype).to(tl.float32)
        if LAMBDA_ONE:
            return deq
        return x_f32 + lambda_ * (deq - x_f32)

    @triton.jit
    def _act_quant_kernel(
        x_ptr,
        out_ptr,
        lambda_,
        in_features,
        EPS: tl.constexpr,
        QMAX: tl.constexpr,
        LAMBDA_ONE: tl.constexpr,
        SINGLE_TILE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        base = row * in_features

        if SINGLE_TILE:
            cols = tl.arange(0, BLOCK)
            mask = cols < in_features
            x = tl.load(x_ptr + base + cols, mask=mask, other=0.0)
            x_f32 = x.to(tl.float32)
            # reciprocal multiply: that is how torch divides by the python scalar 127
            scale = tl.maximum(tl.max(tl.abs(x_f32)) * (1.0 / QMAX), EPS)
            out = _blend(x, x_f32, scale, lambda_, QMAX, LAMBDA_ONE)
            tl.store(out_ptr + base + cols, out.to(x.dtype), mask=mask)
        else:
            amax = 0.0
            for start in range(0, in_features, BLOCK):
                cols = start + tl.arange(0, BLOCK)
                mask = cols < in_features
                x = tl.load(x_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
                amax = tl.maximum(amax, tl.max(tl.abs(x)))
            scale = tl.maximum(amax * (1.0 / QMAX), EPS)
            for start in range(0, in_features, BLOCK):
                cols = start + tl.arange(0, BLOCK)
                mask = cols < in_features
                x = tl.load(x_ptr + base + cols, mask=mask, other=0.0)
                out = _blend(x, x.to(tl.float32), scale, lambda_, QMAX, LAMBDA_ONE)
                tl.store(out_ptr + base + cols, out.to(x.dtype), mask=mask)


def _row_tiling(in_features: int) -> tuple[int, bool]:
    """Return `(block, single_tile)` for a row of `in_features` elements.

    Keeping a whole row resident only pays off when it fills the tile exactly; padded
    tiles waste lanes and registers, so odd widths (2560, 9728) reduce in two passes.
    """
    block = triton.next_power_of_2(in_features)
    if block <= 1024 or (block == in_features and in_features <= 8192):
        return block, True
    return 1024, False


def act_quant(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    """Fused `quant.act_quant`: per-token scale, int8 clip and λ-interp in one pass.

    Args:
        x: Activation on a CUDA device, shape `(..., in_features)`.
        lambda_: Quantization strength in `[0, 1]`.

    Returns:
        Effective activation, same shape and dtype as `x`.

    Raises:
        RuntimeError: If Triton is unavailable or `x` is not on CUDA.
    """
    require_cuda(x)
    x = x.contiguous()
    out = torch.empty_like(x)
    if x.numel() == 0:
        return out

    in_features = x.shape[-1]
    rows = x.numel() // in_features
    block, single_tile = _row_tiling(in_features)
    # no FP contraction: an FMA'd λ-interp drifts from the eager oracle
    _act_quant_kernel[(rows,)](
        x,
        out,
        lambda_,
        in_features,
        EPS=SCALE_EPS,
        QMAX=float(ACT_QMAX),
        LAMBDA_ONE=lambda_ >= 1.0,
        SINGLE_TILE=single_tile,
        BLOCK=block,
        num_warps=num_warps_for(block),
        enable_fp_fusion=False,
    )
    return out
