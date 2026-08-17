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
- handles the per-expert weight ``per_tensor_scale`` EXACTLY: the engine keeps
  the stored e4m3 block scales and multiplies the full pts values into the
  fp32 GEMM output (post-matmul), via a per-row fp32 colvec
  (``pts`` repeated over each expert's rows, sync-free). Single bf16 rounding
  at the cast, and it composes with ``add_to_output`` (the multiply runs
  before the delta add). The old lossy scheme (fold the pts_e/pts_ref RATIOS
  into SFB, ``alpha = pts_ref``; re-rounds SFB in e4m3) stays reachable via
  ``AXOLOTL_SONICMOE_NVFP4_PTS_FOLD=1`` for A/B numerics debugging;
- the backward dequant (:func:`dequantize_engine_weight`) therefore uses the
  plain stored-scales-times-pts semantics (``dequantize_expert_weight``), so
  the dense weights the dX path sees are identical to the operands the forward
  kernel consumed; with the fold env set it uses the folded scales and alpha;
- wraps the grouped GEMM in an autograd.Function whose backward computes dX by
  per-expert chunked dequant matmuls, never through the packed fp4 operand.
"""

from __future__ import annotations

import os
from typing import NamedTuple, Optional

import torch

from .fp4_cute import GroupedNvfp4Gemm
from .nvfp4 import dequantize_expert_slice, is_nvfp4_param
from .nvfp4_quant import quantize_nvfp4_ref
from .sf_layout import build_varlen_sfa, fold_per_tensor_scale


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


class _EngineEntry(NamedTuple):
    engine: GroupedNvfp4Gemm
    # alpha scales the fp32 accumulator. colvec_pts [E] fp32 means the engine
    # keeps the stored scales and pts is applied exactly per row in the
    # epilogue; folded_scale means SFB holds the pts_e/pts_ref ratios (old
    # scheme). fused=True means no post-GEMM row scaling is needed.
    alpha: float
    folded_scale: Optional[torch.Tensor]
    fold_rel_err: float
    fused: bool
    colvec_pts: Optional[torch.Tensor]


# Keyed by the packed storage; safe against pointer reuse because each engine
# holds a view of its weight's qdata, keeping that storage alive.
_ENGINE_CACHE: dict = {}


def _engine_key(qdata: torch.Tensor) -> tuple:
    return (qdata.data_ptr(), tuple(qdata.shape), qdata.device.index)


def _get_engine(weight) -> _EngineEntry:
    qdata, scale, pts = unpack_nvfp4_components(weight)
    key = _engine_key(qdata)
    entry = _ENGINE_CACHE.get(key)
    if entry is None:
        e, n, k2 = qdata.shape
        engine = GroupedNvfp4Gemm(n, k2 * 2, e, gated=False)
        alpha, folded, rel_err, colvec_pts = 1.0, None, 0.0, None
        # Default (unset/0): exact scheme, stored scales + per-row pts colvec
        # multiplied into the fp32 accumulator in the epilogue. "1": old lossy
        # SFB ratio fold (re-rounds SFB in e4m3, ~6% max block-scale err on
        # real checkpoints), kept for A/B numerics debugging.
        fold_enabled = os.environ.get("AXOLOTL_SONICMOE_NVFP4_PTS_FOLD", "0") == "1"
        if pts is not None:
            if fold_enabled:
                # One-time host sync per weight, at engine build.
                pts_ref = float(pts.float().max())
                if pts_ref > 0:
                    try:
                        folded, rel_err = fold_per_tensor_scale(
                            scale,
                            pts.float().reshape(-1) / pts_ref,
                            allow_underflow=True,
                        )
                        alpha = pts_ref
                    except ValueError:
                        folded = None
            else:
                colvec_pts = pts.float().reshape(-1).contiguous()
        engine.set_weights(qdata, scale if folded is None else folded)
        fused = pts is None or folded is not None or colvec_pts is not None
        entry = _EngineEntry(engine, alpha, folded, rel_err, fused, colvec_pts)
        _ENGINE_CACHE[key] = entry
    return entry


class _GatedEngineEntry(NamedTuple):
    engine: GroupedNvfp4Gemm
    colvec_pts: Optional[torch.Tensor]


_GATED_ENGINE_CACHE: dict = {}


def _get_gated_engine(weight, activation: str = "swiglu") -> _GatedEngineEntry:
    """Gated (fused-activation) engine for a concat-layout ``[E, 2I, H]`` NVFP4
    up-projection weight. ``concat_b=True``: the packed qdata is consumed
    zero-copy (the kernel views concat rows as interleaved); pts is always
    applied exactly via the per-row colvec (no fold variant here)."""
    qdata, scale, pts = unpack_nvfp4_components(weight)
    key = _engine_key(qdata)
    entry = _GATED_ENGINE_CACHE.get(key)
    if entry is None:
        e, n, k2 = qdata.shape
        engine = GroupedNvfp4Gemm(
            n, k2 * 2, e, gated=True, activation=activation, concat_b=True
        )
        engine.set_weights(qdata, scale)
        colvec_pts = pts.float().reshape(-1).contiguous() if pts is not None else None
        entry = _GatedEngineEntry(engine, colvec_pts)
        _GATED_ENGINE_CACHE[key] = entry
    return entry


def gated_nvfp4_forward(
    x_grouped: torch.Tensor,
    weight,
    cu_seqlens: torch.Tensor,
    aux: Optional[torch.Tensor],
    activation: str = "swiglu",
) -> tuple:
    """Fused up-GEMM + gated activation (+ optional preact aux add).

    ``aux`` ``[T, 2I]`` bf16 in INTERLEAVED gate/up layout (the LoRA delta;
    compute it against row-permuted LoRA-B factors) is added to the fp32
    accumulator after the exact per-row pts multiply, before the activation.
    Returns ``(postact [T, I] bf16, preact [T, 2I] bf16)`` where the preact
    memory layout is also INTERLEAVED (``preact.view(T, I, 2)`` puts gate at
    ``[..., 0]`` and up at ``[..., 1]``). No autograd."""
    entry = _get_gated_engine(weight, activation)
    a_q, sfa = quantize_grouped_rows(x_grouped, cu_seqlens)
    colvec = None
    if entry.colvec_pts is not None:
        counts = (cu_seqlens[1:] - cu_seqlens[:-1]).long()
        colvec = torch.repeat_interleave(
            entry.colvec_pts, counts, output_size=x_grouped.shape[0]
        )
    return entry.engine.forward(
        a_q,
        sfa,
        cu_seqlens,
        colvec=colvec,
        aux=aux,
    )


def dequantize_engine_weight(weight) -> torch.Tensor:
    """Dense weights matching the operands the fp4_cute forward consumed.

    The default colvec path applies the FULL pts exactly in the forward
    epilogue, so plain ``dequantize_expert_weight`` (stored scales times pts)
    already matches; only when the engine folded the pts ratios into SFB
    (``AXOLOTL_SONICMOE_NVFP4_PTS_FOLD=1``) does this dequantize with the
    folded scales and alpha instead.
    """
    from .nvfp4 import dequantize_expert_weight

    if not is_nvfp4_param(weight):
        return weight
    qdata, _, _ = unpack_nvfp4_components(weight)
    entry = _ENGINE_CACHE.get(_engine_key(qdata))
    if entry is None or entry.folded_scale is None:
        return dequantize_expert_weight(weight)
    from .triton_nvfp4 import dequant_nvfp4_triton, triton_available

    alpha_t = torch.tensor([entry.alpha], dtype=torch.float32, device=qdata.device)
    if qdata.is_cuda and triton_available():
        return dequant_nvfp4_triton(
            qdata, entry.folded_scale, alpha_t, weight.orig_dtype
        )
    from .nvfp4_quant import dequantize_nvfp4_ref

    return (dequantize_nvfp4_ref(qdata, entry.folded_scale) * entry.alpha).to(
        weight.orig_dtype
    )


def quantize_grouped_rows(x_grouped: torch.Tensor, cu_seqlens: torch.Tensor) -> tuple:
    """NVFP4-quantize expert-sorted rows: ``(a_packed [T, K/2] u8, sfa blocked)``."""
    from .triton_nvfp4 import quantize_rows_fused_sfa_triton, triton_available

    if (
        x_grouped.is_cuda
        and triton_available()
        and (x_grouped.shape[-1] // 16) % 4 == 0
    ):
        return quantize_rows_fused_sfa_triton(x_grouped, cu_seqlens)
    a_q, a_s, _ = quantize_nvfp4_ref(x_grouped)
    return a_q, build_varlen_sfa(a_s, cu_seqlens)


def _base_gemm_forward(
    entry: _EngineEntry,
    weight,
    x_grouped: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: torch.Tensor | None = None,
    add_to_output: bool = False,
) -> torch.Tensor:
    a_q, sfa = quantize_grouped_rows(x_grouped, cu_seqlens)
    colvec = None
    if entry.colvec_pts is not None:
        # Per-row FULL pts, sync-free: counts stay on device, output_size is
        # known host-side from the packed activation rows.
        counts = (cu_seqlens[1:] - cu_seqlens[:-1]).long()
        colvec = torch.repeat_interleave(
            entry.colvec_pts, counts, output_size=x_grouped.shape[0]
        )
    if entry.fused:
        return entry.engine.forward(
            a_q,
            sfa,
            cu_seqlens,
            alpha=entry.alpha,
            colvec=colvec,
            out=out,
            add_to_output=add_to_output,
        )
    assert not add_to_output
    result = entry.engine.forward(a_q, sfa, cu_seqlens)
    pts = weight.per_tensor_scale
    counts = (cu_seqlens[1:] - cu_seqlens[:-1]).long()
    row_pts = torch.repeat_interleave(pts.view(-1).float(), counts)
    from .triton_nvfp4 import rowscale_inplace_triton, triton_available

    if result.is_cuda and triton_available() and result.dtype == x_grouped.dtype:
        return rowscale_inplace_triton(result.contiguous(), row_pts)
    return (result.float() * row_pts.unsqueeze(1)).to(x_grouped.dtype)


def grouped_dx_dequant(
    grad_h: torch.Tensor, weight, cu_seqlens: torch.Tensor
) -> torch.Tensor:
    """``dx[start:end] = g_e @ W_e``, never through the packed fp4 operand."""
    from .fp8_bwd import fp8_dx_supported, grouped_fp8_dx
    from .nvfp4_lora import _use_grouped_mm

    if fp8_dx_supported(grad_h, weight):
        return grouped_fp8_dx(grad_h, weight, cu_seqlens)

    if _use_grouped_mm(grad_h):
        w_dense = dequantize_engine_weight(weight).to(grad_h.dtype)
        return torch._grouped_mm(grad_h, w_dense, offs=cu_seqlens[1:].to(torch.int32))

    cu = cu_seqlens.tolist()
    dx = grad_h.new_empty((grad_h.shape[0], weight.shape[-1]))
    qdata, _, _ = unpack_nvfp4_components(weight)
    entry = _ENGINE_CACHE.get(_engine_key(qdata))
    folded = entry.folded_scale if entry is not None else None
    alpha = entry.alpha if entry is not None else 1.0
    for e in range(len(cu) - 1):
        start, end = cu[e], cu[e + 1]
        if end <= start:
            continue
        if folded is not None:
            from .nvfp4_quant import dequantize_nvfp4_ref

            w_e = dequantize_nvfp4_ref(qdata[e], folded[e]) * alpha
        else:
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
        entry = _get_engine(weight)
        out = _base_gemm_forward(entry, weight, x_grouped, cu_seqlens)
        ctx.save_for_backward(weight, cu_seqlens)
        return out.to(x_grouped.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        weight, cu_seqlens = ctx.saved_tensors
        dx = grouped_dx_dequant(grad_out.contiguous(), weight, cu_seqlens)
        return dx, None, None


def _cu_seqlens(end_offsets: torch.Tensor) -> torch.Tensor:
    return torch.cat([end_offsets.new_zeros(1), end_offsets]).to(torch.int32)


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
    return _GroupedNvfp4Linear.apply(x_grouped, weight, _cu_seqlens(end_offsets))


def grouped_nvfp4_linear_add_delta(
    x_grouped: torch.Tensor,
    weight,
    end_offsets: torch.Tensor,
    delta: torch.Tensor,
) -> torch.Tensor:
    """``x_e @ w[e]^T + delta``, accumulated in the GEMM epilogue when possible.

    No autograd: callers (the grouped LoRA autograd.Functions) own the
    backward. ``delta`` ``[T, N]`` may be overwritten and returned.
    """
    assert x_grouped.shape[-1] == weight.shape[-1], (
        f"x K={x_grouped.shape[-1]} vs weight K={weight.shape[-1]}"
    )
    cu = _cu_seqlens(end_offsets)
    entry = _get_engine(weight)
    if entry.fused and delta.dtype == torch.bfloat16:
        out = delta.contiguous()
        _base_gemm_forward(entry, weight, x_grouped, cu, out=out, add_to_output=True)
        return out
    return _base_gemm_forward(entry, weight, x_grouped, cu) + delta
