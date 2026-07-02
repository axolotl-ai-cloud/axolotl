# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""``backend="fp4_cute"`` ops: torchao NVFP4 base weights on the SM100 W4A4 engine.

Bridges the grouped MoE-LoRA path (``nvfp4.py`` / ``nvfp4_lora.py``, bf16
expert-sorted rows) to ``fp4_cute.GroupedNvfp4Gemm``:

- unpacks a torchao ``NVFP4Tensor`` into the (qdata, block_scale, pts)
  components the engine consumes, with a per-weight engine cache (the base is
  frozen, so weights are packed into the engine exactly once);
- quantizes activations to NVFP4 on the expert-sorted rows (per-16 block
  scales, per-tensor scale 1.0) and builds the dQaccum-padded SFA;
- applies the per-expert weight ``per_tensor_scale`` to the output rows after
  the GEMM, in fp32. Folding it into the e4m3 block scales instead truncates:
  the folded scale is ~``amax_block / 6``, which sits in e4m3's subnormal
  range (< 2^-6) at real LLM weight magnitudes and keeps only the 3 subnormal
  mantissa bits (up to tens of percent block error). Post-scaling keeps SFB
  at the stored full-range scales and makes the forward weights identical to
  the exact dequant the backward uses;
- wraps the grouped GEMM in an autograd.Function whose backward computes dX by
  per-expert chunked dequant matmuls, never through the packed fp4 operand.
"""

from __future__ import annotations

import torch

from .fp4_cute import GroupedNvfp4Gemm
from .nvfp4 import dequantize_expert_slice, is_nvfp4_param
from .nvfp4_quant import quantize_nvfp4_ref
from .sf_layout import build_varlen_sfa


def unpack_nvfp4_components(w) -> tuple:
    """``(qdata u8 [E, N, K/2], block_scale e4m3 [E, N, K/16], pts [E,1,1] | None)``."""
    if not is_nvfp4_param(w):
        raise TypeError(f"expected a torchao NVFP4Tensor, got {type(w).__name__}")
    assert not getattr(w, "is_swizzled_scales", False), (
        "swizzled NVFP4 scales unsupported (our loaders emit row-major)"
    )
    qdata = w.qdata
    if qdata.dtype != torch.uint8:
        qdata = qdata.view(torch.uint8)
    return qdata, w.scale, w.per_tensor_scale


def fp4_cute_dims_ok(w1, w2) -> bool:
    """Both grouped GEMMs run the non-gated engine: K % 32 and N % 8.

    ``w1`` ``[E, 2I, H]`` (K=H), ``w2`` ``[E, H, I]`` (K=I); shapes are logical
    (unpacked) as the tensor subclass reports them.
    """
    n1, k1 = w1.shape[-2], w1.shape[-1]
    n2, k2 = w2.shape[-2], w2.shape[-1]
    return k1 % 32 == 0 and n1 % 8 == 0 and k2 % 32 == 0 and n2 % 8 == 0


# Keyed by the packed storage; safe against pointer reuse because each engine
# holds a view of its weight's qdata, keeping that storage alive.
_ENGINE_CACHE: dict = {}


def _get_engine(weight) -> GroupedNvfp4Gemm:
    qdata, scale, _ = unpack_nvfp4_components(weight)
    key = (qdata.data_ptr(), tuple(qdata.shape), qdata.device.index)
    engine = _ENGINE_CACHE.get(key)
    if engine is None:
        e, n, k2 = qdata.shape
        engine = GroupedNvfp4Gemm(n, k2 * 2, e, gated=False)
        # per_tensor_scale is applied to the output rows in forward(), not
        # folded into SFB (folding truncates in e4m3's subnormal range).
        engine.set_weights(qdata, scale)
        _ENGINE_CACHE[key] = engine
    return engine


def quantize_grouped_rows(x_grouped: torch.Tensor, cu_seqlens: torch.Tensor) -> tuple:
    """NVFP4-quantize expert-sorted rows: ``(a_packed [T, K/2] u8, sfa blocked)``."""
    from .triton_nvfp4 import quantize_nvfp4_triton, triton_available

    if x_grouped.is_cuda and triton_available():
        a_q, a_s = quantize_nvfp4_triton(x_grouped)
    else:
        a_q, a_s, _ = quantize_nvfp4_ref(x_grouped)
    return a_q, build_varlen_sfa(a_s, cu_seqlens)


def grouped_dx_dequant(
    grad_h: torch.Tensor, weight, cu_seqlens: torch.Tensor
) -> torch.Tensor:
    """``dx[start:end] = g_e @ W_e``, never through the packed fp4 operand."""
    from .nvfp4 import dequantize_expert_weight
    from .nvfp4_lora import _use_grouped_mm

    if _use_grouped_mm(grad_h):
        w_dense = dequantize_expert_weight(weight).to(grad_h.dtype)
        return torch._grouped_mm(grad_h, w_dense, offs=cu_seqlens[1:].to(torch.int32))

    cu = cu_seqlens.tolist()
    dx = grad_h.new_empty((grad_h.shape[0], weight.shape[-1]))
    for e in range(len(cu) - 1):
        start, end = cu[e], cu[e + 1]
        if end <= start:
            continue
        w_e = dequantize_expert_slice(weight, e)
        dx[start:end] = grad_h[start:end] @ w_e.to(grad_h.dtype)
    return dx


class _GroupedNvfp4Linear(torch.autograd.Function):
    """``y_e = quant16(x_e) @ W_e^T`` with a frozen NVFP4 ``W``.

    Backward treats the activation quantization as identity (straight-through)
    and computes ``dx`` by chunked dequant matmuls; ``W`` gets no gradient.
    """

    @staticmethod
    def forward(ctx, x_grouped, weight, cu_seqlens):
        engine = _get_engine(weight)
        a_q, sfa = quantize_grouped_rows(x_grouped, cu_seqlens)
        out = engine.forward(a_q, sfa, cu_seqlens)
        pts = weight.per_tensor_scale
        if pts is not None:
            counts = (cu_seqlens[1:] - cu_seqlens[:-1]).long()
            row_pts = torch.repeat_interleave(pts.view(-1), counts)
            out = (out.float() * row_pts.unsqueeze(1)).to(x_grouped.dtype)
        ctx.save_for_backward(weight, cu_seqlens)
        return out.to(x_grouped.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        weight, cu_seqlens = ctx.saved_tensors
        dx = grouped_dx_dequant(grad_out.contiguous(), weight, cu_seqlens)
        return dx, None, None


def grouped_nvfp4_linear(
    x_grouped: torch.Tensor, weight, end_offsets: torch.Tensor
) -> torch.Tensor:
    """Per-expert ``x_e @ w[e]^T`` on the SM100 engine.

    ``end_offsets`` ``[E]`` are cumulative row ends (``_grouped_gemm``'s
    convention); the leading zero is prepended to form ``cu_seqlens``.
    """
    assert x_grouped.shape[-1] == weight.shape[-1], (
        f"x K={x_grouped.shape[-1]} vs weight K={weight.shape[-1]}"
    )
    cu = torch.cat([end_offsets.new_zeros(1), end_offsets]).to(torch.int32)
    return _GroupedNvfp4Linear.apply(x_grouped, weight, cu)
