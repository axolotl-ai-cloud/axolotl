"""Fused weight fake-quant: codes, f16-rounded dequant and λ-interpolation in one pass.

The absmean reduction stays in `quant.absmean_scale` (a plain fp32 `torch.mean`) so the
scale is bit-identical to the oracle's; only the elementwise pass — which the eager path
pays four times over — is fused here.
"""

from __future__ import annotations

import torch

from ...quant import SCALE_EPS, absmean_scale
from ._common import HAVE_TRITON, num_warps_for, require_cuda

if HAVE_TRITON:
    import triton
    import triton.language as tl

    try:
        from triton.language.extra.libdevice import div_rn, rint
    except ModuleNotFoundError:  # pragma: no cover
        from triton.language.extra.cuda.libdevice import div_rn, rint

    @triton.jit
    def _fake_quant_kernel(
        w_ptr,
        out_ptr,
        scale_ptr,
        lambda_,
        numel,
        EPS: tl.constexpr,
        LAMBDA_ONE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < numel

        w = tl.load(w_ptr + offs, mask=mask, other=0.0)
        w_f32 = w.to(tl.float32)
        scale = tl.load(scale_ptr).to(tl.float32)
        scale_f16 = scale.to(tl.float16).to(tl.float32)

        # div_rn, not `/`: only correctly-rounded division reproduces the oracle's ties
        code = rint(div_rn(w_f32, tl.maximum(scale, EPS)))
        code = tl.minimum(tl.maximum(code, -1.0), 1.0).to(tl.int8)
        # the oracle materializes `code · s16` in the weight dtype before blending
        deq = (code.to(tl.float32) * scale_f16).to(w.dtype).to(tl.float32)

        if LAMBDA_ONE:
            out = deq
        else:
            out = w_f32 + lambda_ * (deq - w_f32)
        tl.store(out_ptr + offs, out.to(w.dtype), mask=mask)

    @triton.jit
    def _fake_quant_group_kernel(
        w_ptr,
        out_ptr,
        scale_ptr,
        lambda_,
        in_features,
        n_groups,
        group_size,
        EPS: tl.constexpr,
        LAMBDA_ONE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < in_features

        w = tl.load(w_ptr + row * in_features + cols, mask=mask, other=0.0)
        w_f32 = w.to(tl.float32)
        scale = tl.load(
            scale_ptr + row * n_groups + cols // group_size, mask=mask, other=1.0
        ).to(tl.float32)
        scale_f16 = scale.to(tl.float16).to(tl.float32)

        code = rint(div_rn(w_f32, tl.maximum(scale, EPS)))
        code = tl.minimum(tl.maximum(code, -1.0), 1.0).to(tl.int8)
        deq = (code.to(tl.float32) * scale_f16).to(w.dtype).to(tl.float32)

        if LAMBDA_ONE:
            out = deq
        else:
            out = w_f32 + lambda_ * (deq - w_f32)
        tl.store(out_ptr + row * in_features + cols, out.to(w.dtype), mask=mask)


def fake_quant_weight(
    weight: torch.Tensor,
    lambda_: float = 1.0,
    group_size: int | None = None,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused `quant.fake_quant_weight`: absmean → codes → `code · s16` → λ-interp.

    Args:
        weight: Latent weight on a CUDA device.
        lambda_: Quantization strength in `[0, 1]`.
        group_size: Group size for group-scale mode, `None` for per tensor.
        scale: Explicit fp32 scale (learnable-scale mode); derived by
            `quant.absmean_scale` when `None`.

    Returns:
        Effective weight, same shape and dtype as `weight`.

    Raises:
        RuntimeError: If Triton is unavailable or `weight` is not on CUDA.
        ValueError: If `group_size` does not divide the input features.
    """
    require_cuda(weight)
    weight = weight.contiguous()
    if scale is None:
        scale = absmean_scale(weight, group_size)
    scale = scale.to(torch.float32).contiguous()
    out = torch.empty_like(weight)
    if weight.numel() == 0:
        return out

    lambda_one = lambda_ >= 1.0
    if group_size is None:
        block = 2048
        grid = (triton.cdiv(weight.numel(), block),)
        # no FP contraction: an FMA'd λ-interp drifts from the eager oracle
        _fake_quant_kernel[grid](
            weight,
            out,
            scale,
            lambda_,
            weight.numel(),
            EPS=SCALE_EPS,
            LAMBDA_ONE=lambda_one,
            BLOCK=block,
            num_warps=num_warps_for(block),
            enable_fp_fusion=False,
        )
        return out

    in_features = weight.shape[-1]
    if in_features % group_size:
        raise ValueError(
            f"group_size {group_size} does not divide in_features {in_features}"
        )
    rows = weight.numel() // in_features
    block = min(triton.next_power_of_2(in_features), 2048)
    _fake_quant_group_kernel[(rows, triton.cdiv(in_features, block))](
        weight,
        out,
        scale,
        lambda_,
        in_features,
        in_features // group_size,
        group_size,
        EPS=SCALE_EPS,
        LAMBDA_ONE=lambda_one,
        BLOCK=block,
        num_warps=num_warps_for(block),
        enable_fp_fusion=False,
    )
    return out
