"""Fused fake-quant for the multi-state grids: `binary`, `dual`, `trit_planes`.

Same split as the single-plane kernel next door: every *reduction* that sets a scale
stays in the eager oracle, so the scales here are bit-identical to it by construction,
and only the elementwise pass is fused. That pass is what the eager path pays several
times over — the five- and nine-state assignments materialize a comparison tensor per
candidate level, which is why a `dual` step costs multiples of an `absmean` one.

The kernels reproduce the oracle exactly, including the parts that look incidental:

- the quantized value is rounded to the *weight* dtype before the λ blend, because
  `dequantize_*` does its cast before `_interp` upcasts again;
- `dual`'s level thresholds are the midpoints `0.5·s_lo` and `0.5·(s_lo + s_hi)`, with
  a magnitude exactly on a boundary taking the lower level;
- `trit_planes` picks the nearest of its five non-negative candidates with ties going
  to the smaller magnitude, which is order-independent and so replays here as a
  running `(distance, magnitude)` comparison rather than a fixed candidate order.

Backward stays the identity `Function` in `quant`: there is no gradient to fuse, the
STE hands the incoming gradient straight to the latent and the scale gradients are
cheap reductions over the codes.
"""

from __future__ import annotations

import torch

from ...quant import (
    SCALE_EPS,
    _ordered_dual_scales,
    binary_scale,
    trit_plane_grid_scales,
)
from ._common import HAVE_TRITON, num_warps_for, require_cuda

if HAVE_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _binary_kernel(
        w_ptr,
        out_ptr,
        scale_ptr,
        lambda_,
        in_features,
        n_groups,
        group_size,
        LAMBDA_ONE: tl.constexpr,
        PER_TENSOR: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """`sign(w) · f16(s)`, λ-blended. No zero state, so no threshold at all."""
        row = tl.program_id(0).to(tl.int64)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < in_features
        offs = row * in_features + cols

        w = tl.load(w_ptr + offs, mask=mask, other=0.0)
        w_f32 = w.to(tl.float32)
        if PER_TENSOR:
            scale = tl.load(scale_ptr).to(tl.float32)
        else:
            scale = tl.load(
                scale_ptr + row * n_groups + cols // group_size, mask=mask, other=1.0
            ).to(tl.float32)
        scale_f16 = scale.to(tl.float16).to(tl.float32)

        # `binary_codes` sends zeros to +1, so this is a `< 0` test, not a sign()
        code = tl.where(w_f32 < 0.0, -1.0, 1.0)
        deq = (code * scale_f16).to(w.dtype).to(tl.float32)

        if LAMBDA_ONE:
            out = deq
        else:
            out = w_f32 + lambda_ * (deq - w_f32)
        tl.store(out_ptr + offs, out.to(w.dtype), mask=mask)

    @triton.jit
    def _dual_kernel(
        w_ptr,
        out_ptr,
        low_ptr,
        high_ptr,
        lambda_,
        in_features,
        LAMBDA_ONE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Nearest of `{0, ±s_lo, ±s_hi}` by midpoint threshold, λ-blended."""
        row = tl.program_id(0).to(tl.int64)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < in_features
        offs = row * in_features + cols

        w = tl.load(w_ptr + offs, mask=mask, other=0.0)
        w_f32 = w.to(tl.float32)
        low = tl.load(low_ptr + row).to(tl.float32)
        high = tl.load(high_ptr + row).to(tl.float32)

        magnitude = tl.abs(w_f32)
        # the oracle's two thresholds; `>` so a boundary magnitude takes the lower level
        level = (magnitude > 0.5 * low).to(tl.int32) + (
            magnitude > 0.5 * (low + high)
        ).to(tl.int32)
        sign = tl.where(w_f32 > 0.0, 1.0, tl.where(w_f32 < 0.0, -1.0, 0.0))

        picked = tl.where(
            level > 1,
            high.to(tl.float16).to(tl.float32),
            low.to(tl.float16).to(tl.float32),
        )
        deq = (tl.where(level > 0, sign * picked, 0.0)).to(w.dtype).to(tl.float32)

        if LAMBDA_ONE:
            out = deq
        else:
            out = w_f32 + lambda_ * (deq - w_f32)
        tl.store(out_ptr + offs, out.to(w.dtype), mask=mask)

    @triton.jit
    def _trit_planes_kernel(
        w_ptr,
        out_ptr,
        first_ptr,
        second_ptr,
        lambda_,
        in_features,
        LAMBDA_ONE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Nearest of the nine free sums `t1·s1 + t2·s2`, λ-blended.

        Only the five non-negative candidates are searched; the sign flips both trits.
        """
        row = tl.program_id(0).to(tl.int64)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < in_features
        offs = row * in_features + cols

        w = tl.load(w_ptr + offs, mask=mask, other=0.0)
        w_f32 = w.to(tl.float32)
        first = tl.load(first_ptr + row).to(tl.float32)
        second = tl.load(second_ptr + row).to(tl.float32)
        magnitude = tl.abs(w_f32)

        # candidate 0 seeds the search, then each is taken when strictly closer, or
        # equally close and smaller — the oracle's order-independent tie rule
        best_value = 0.0 + tl.zeros_like(magnitude)
        best_distance = magnitude
        best_t1 = tl.zeros_like(magnitude)
        best_t2 = tl.zeros_like(magnitude)

        for state in tl.static_range(1, 5):
            if state == 1:
                candidate, t1, t2 = second, 0.0, 1.0
            elif state == 2:
                candidate, t1, t2 = first - second, 1.0, -1.0
            elif state == 3:
                candidate, t1, t2 = first, 1.0, 0.0
            else:
                candidate, t1, t2 = first + second, 1.0, 1.0
            distance = tl.abs(magnitude - candidate)
            closer = (distance < best_distance) | (
                (distance == best_distance) & (candidate < best_value)
            )
            best_value = tl.where(closer, candidate, best_value)
            best_distance = tl.minimum(best_distance, distance)
            best_t1 = tl.where(closer, t1, best_t1)
            best_t2 = tl.where(closer, t2, best_t2)

        sign = tl.where(w_f32 < 0.0, -1.0, 1.0)
        deq = (best_t1 * sign) * first + (best_t2 * sign) * second
        deq = deq.to(w.dtype).to(tl.float32)

        if LAMBDA_ONE:
            out = deq
        else:
            out = w_f32 + lambda_ * (deq - w_f32)
        tl.store(out_ptr + offs, out.to(w.dtype), mask=mask)

    @triton.jit
    def _dual_scale_grad_kernel(
        grad_ptr,
        codes_ptr,
        low_ptr,
        high_ptr,
        lambda_,
        numel,
        BLOCK: tl.constexpr,
    ):
        """Split the incoming gradient by level in one pass over the tensor.

        The eager path walks it about nine times — cast, sign, scale, two level
        masks, two products, two row-wise `where` — allocating a full-size temporary
        each time. The `where` is not here at all: `swapped` is constant per row, so
        selecting before or after the row sum gives the same answer, and the caller
        does it on the two reduced vectors instead.
        """
        pid = tl.program_id(0).to(tl.int64)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < numel

        grad = tl.load(grad_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        codes = tl.load(codes_ptr + offs, mask=mask, other=0).to(tl.int32)

        magnitude = tl.abs(codes)
        sign = tl.where(codes > 0, 1.0, tl.where(codes < 0, -1.0, 0.0))
        scaled = grad * sign * lambda_
        tl.store(low_ptr + offs, tl.where(magnitude == 1, scaled, 0.0), mask=mask)
        tl.store(high_ptr + offs, tl.where(magnitude > 1, scaled, 0.0), mask=mask)

    @triton.jit
    def _trit_scale_grad_kernel(
        grad_ptr,
        plane0_ptr,
        plane1_ptr,
        first_ptr,
        second_ptr,
        lambda_,
        numel,
        BLOCK: tl.constexpr,
    ):
        """Weight the incoming gradient by each trit plane in one pass."""
        pid = tl.program_id(0).to(tl.int64)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < numel

        grad = tl.load(grad_ptr + offs, mask=mask, other=0.0).to(tl.float32) * lambda_
        t1 = tl.load(plane0_ptr + offs, mask=mask, other=0).to(tl.float32)
        t2 = tl.load(plane1_ptr + offs, mask=mask, other=0).to(tl.float32)
        tl.store(first_ptr + offs, grad * t1, mask=mask)
        tl.store(second_ptr + offs, grad * t2, mask=mask)


_BLOCK: int = 1024


def fake_quant_weight_binary(
    weight: torch.Tensor,
    lambda_: float = 1.0,
    group_size: int | None = None,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused `quant.fake_quant_weight_binary`.

    Raises:
        RuntimeError: If Triton is unavailable or `weight` is not on CUDA.
    """
    require_cuda(weight)
    if lambda_ <= 0.0:
        return weight
    if scale is None:
        scale = binary_scale(weight, group_size)
    else:
        scale = scale.to(torch.float32).clamp_min(SCALE_EPS)

    weight = weight.contiguous()
    rows, in_features = _rows_and_cols(weight)
    out = torch.empty_like(weight)
    per_tensor = group_size is None
    scale = scale.contiguous().to(torch.float32)
    _binary_kernel[(rows, triton.cdiv(in_features, _BLOCK))](
        weight,
        out,
        scale,
        lambda_,
        in_features,
        1 if per_tensor else scale.shape[-1],
        group_size or 1,
        LAMBDA_ONE=lambda_ >= 1.0,
        PER_TENSOR=per_tensor,
        BLOCK=_BLOCK,
        num_warps=num_warps_for(_BLOCK),
    )
    return out


def fake_quant_weight_dual(
    weight: torch.Tensor,
    lambda_: float = 1.0,
    scale_lo: torch.Tensor | None = None,
    scale_hi: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused `quant.fake_quant_weight_dual`.

    Raises:
        RuntimeError: If Triton is unavailable, `weight` is not on CUDA, or the pair
            is missing (the statistic is the oracle's job, not the kernel's).
    """
    require_cuda(weight)
    if lambda_ <= 0.0:
        return weight
    if scale_lo is None or scale_hi is None:
        raise RuntimeError("the fused dual kernel needs an explicit (s_lo, s_hi) pair")

    low, high = _ordered_dual_scales(scale_lo, scale_hi)
    weight = weight.contiguous()
    rows, in_features = _rows_and_cols(weight)
    out = torch.empty_like(weight)
    _dual_kernel[(rows, triton.cdiv(in_features, _BLOCK))](
        weight,
        out,
        _row_vector(low, rows),
        _row_vector(high, rows),
        lambda_,
        in_features,
        LAMBDA_ONE=lambda_ >= 1.0,
        BLOCK=_BLOCK,
        num_warps=num_warps_for(_BLOCK),
    )
    return out


def fake_quant_weight_trit_planes(
    weight: torch.Tensor,
    lambda_: float = 1.0,
    scale_1: torch.Tensor | None = None,
    scale_2: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused `quant.fake_quant_weight_trit_planes`.

    Raises:
        RuntimeError: If Triton is unavailable, `weight` is not on CUDA, or the pair
            is missing.
    """
    require_cuda(weight)
    if lambda_ <= 0.0:
        return weight
    if scale_1 is None or scale_2 is None:
        raise RuntimeError(
            "the fused trit_planes kernel needs an explicit (s1, s2) pair"
        )

    # rounded through the weight dtype in the oracle, so the grid is closed under
    # storage — done here, in Python, to keep the scales bit-identical to it
    first, second = trit_plane_grid_scales(scale_1, scale_2, weight.dtype)
    weight = weight.contiguous()
    rows, in_features = _rows_and_cols(weight)
    out = torch.empty_like(weight)
    _trit_planes_kernel[(rows, triton.cdiv(in_features, _BLOCK))](
        weight,
        out,
        _row_vector(first, rows),
        _row_vector(second, rows),
        lambda_,
        in_features,
        LAMBDA_ONE=lambda_ >= 1.0,
        BLOCK=_BLOCK,
        num_warps=num_warps_for(_BLOCK),
    )
    return out


def dual_scale_grad_terms(
    grad_output: torch.Tensor, codes: torch.Tensor, lambda_: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the per-level gradient terms `(low, high)` the dual STE reduces.

    Fuses the elementwise half of the backward only. The reduction stays in torch:
    a Triton row-sum accumulates in a different order, and fp32 addition is not
    associative, so a fused reduction would make the fused and eager gradients
    differ by accumulation noise. The elementwise pass is where the temporaries are
    anyway.

    Raises:
        RuntimeError: If Triton is unavailable or the tensors are not on CUDA.
    """
    require_cuda(grad_output, codes)
    grad_output = grad_output.contiguous()
    codes = codes.contiguous()
    low = torch.empty(grad_output.shape, dtype=torch.float32, device=grad_output.device)
    high = torch.empty_like(low)
    numel = grad_output.numel()
    _dual_scale_grad_kernel[(triton.cdiv(numel, _BLOCK),)](
        grad_output,
        codes,
        low,
        high,
        lambda_,
        numel,
        BLOCK=_BLOCK,
        num_warps=num_warps_for(_BLOCK),
    )
    return low, high


def trit_scale_grad_terms(
    grad_output: torch.Tensor, planes: torch.Tensor, lambda_: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the per-plane gradient terms `(first, second)` the free-sum STE reduces.

    Raises:
        RuntimeError: If Triton is unavailable or the tensors are not on CUDA.
    """
    require_cuda(grad_output, planes)
    grad_output = grad_output.contiguous()
    planes = planes.contiguous()
    first = torch.empty(
        grad_output.shape, dtype=torch.float32, device=grad_output.device
    )
    second = torch.empty_like(first)
    numel = grad_output.numel()
    _trit_scale_grad_kernel[(triton.cdiv(numel, _BLOCK),)](
        grad_output,
        planes[0],
        planes[1],
        first,
        second,
        lambda_,
        numel,
        BLOCK=_BLOCK,
        num_warps=num_warps_for(_BLOCK),
    )
    return first, second


def _rows_and_cols(weight: torch.Tensor) -> tuple[int, int]:
    if weight.ndim != 2:
        raise RuntimeError(
            f"the fused multi-state kernels need a 2D weight, got {tuple(weight.shape)}"
        )
    return int(weight.shape[0]), int(weight.shape[1])


def _row_vector(scale: torch.Tensor, rows: int) -> torch.Tensor:
    """Return a contiguous fp32 `(rows,)` view of a per-row or broadcast scale."""
    dense = scale.to(torch.float32).reshape(-1)
    if dense.numel() == 1:
        dense = dense.expand(rows)
    return dense.contiguous()


__all__ = [
    "dual_scale_grad_terms",
    "fake_quant_weight_binary",
    "fake_quant_weight_dual",
    "fake_quant_weight_trit_planes",
    "trit_scale_grad_terms",
]
