# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Optional fp8 backward-dX engine over DeepGEMM for the frozen NVFP4 base.

Opt-in via ``AXOLOTL_SONICMOE_NVFP4_BWD=deepgemm``. The default bf16 backward
dequantizes each expert weight every pass and runs ``torch._grouped_mm``; this
caches the weights once as fp8 e4m3 (from the same folded dequant the forward
consumed) and runs dX through DeepGEMM's m-grouped contiguous fp8 GEMM. About
1.5-1.8x over the bf16 grouped GEMM at Qwen3-30B backward shapes (dX rel err
~7e-4), but the cache is a full fp8 copy of every expert weight (+27 GiB on
Qwen3-30B), so it stays off by default.

``torch._scaled_grouped_mm`` is deliberately not a fallback: its rowwise fp8
grouped kernel device-aborts on SM100 (torch 2.10+cu130, not built for sm100a)
and kills the CUDA context. DeepGEMM's contiguous layout needs every expert
segment padded to its M block alignment (128); padded rows carry
``m_indices == -1`` (skipped) and are gathered back after. All index math stays
on device (buffer sized ``T + E * align``) so no host sync is needed.
"""

from __future__ import annotations

import os
from typing import NamedTuple

import torch

from .nvfp4 import is_nvfp4_param

_DEEP_GEMM = None


def _deep_gemm():
    """Import deep_gemm once; torch must own the CUDA context first."""
    global _DEEP_GEMM
    if _DEEP_GEMM is None:
        try:
            torch.zeros(1, device="cuda")
            import deep_gemm

            # 128 over the theoretical 224: every padded-layout cost (quant,
            # zero rows, output buffer) scales with E * alignment, and the
            # GEMM measured slightly FASTER at 128 on B200 (fewer pad rows).
            deep_gemm.set_mk_alignment_for_contiguous_layout(128)
            _DEEP_GEMM = deep_gemm
        except Exception:
            _DEEP_GEMM = False
    return _DEEP_GEMM if _DEEP_GEMM is not False else None


def fp8_dx_supported(grad_h: torch.Tensor, base_weight) -> bool:
    # explicit opt-in only: the fp8 cache costs a full fp8 copy of the expert
    # weights (+27 GiB on Qwen3-30B), unacceptable as a default
    if os.environ.get("AXOLOTL_SONICMOE_NVFP4_BWD") != "deepgemm":
        return False
    if not (
        grad_h.is_cuda
        and grad_h.dtype == torch.bfloat16
        and is_nvfp4_param(base_weight)
        and base_weight.shape[-2] % 128 == 0  # K of the dX GEMM (dim1)
        and torch.cuda.get_device_capability(grad_h.device)[0] == 10
    ):
        raise RuntimeError(
            "AXOLOTL_SONICMOE_NVFP4_BWD=deepgemm requires a CUDA bf16 grad, "
            "a packed NVFP4 base with dim1 % 128 == 0, and an SM100 GPU"
        )
    if _deep_gemm() is None:
        raise RuntimeError(
            "AXOLOTL_SONICMOE_NVFP4_BWD=deepgemm but deep_gemm is not "
            "importable; build it from source "
            "(https://github.com/deepseek-ai/DeepGEMM)"
        )
    return True


_PAD_QUANT_KERNEL = None


def _pad_quant_kernel():
    """Fused pad + per-token fp8 quant (1x128 ue8m0 scales) in one pass.

    The torch chain (zero-fill the padded buffer, scatter, per_token_cast's
    fill + copy + amax + mul) moves ~5x the real data and erased the GEMM win;
    this reads only real rows (padding writes zeros without reading) and its
    rounding is bit-identical to per_token_cast_to_fp8 because ue8m0 scales
    are powers of two.
    """
    global _PAD_QUANT_KERNEL
    if _PAD_QUANT_KERNEL is None:
        import triton
        import triton.language as tl

        @triton.jit
        def _kernel(
            grad_ptr,
            src_ptr,
            q_ptr,
            sf_ptr,
            N_BLOCKS: tl.constexpr,
            BLK: tl.constexpr,
        ):
            pid = tl.program_id(0)
            m = pid // N_BLOCKS
            b = pid % N_BLOCKS
            src = tl.load(src_ptr + m)
            offs = b * BLK + tl.arange(0, BLK)
            if src >= 0:
                x = tl.load(grad_ptr + src.to(tl.int64) * (N_BLOCKS * BLK) + offs).to(
                    tl.float32
                )
            else:
                x = tl.zeros([BLK], dtype=tl.float32)
            amax = tl.maximum(tl.max(tl.abs(x)), 1e-4)
            sf = amax / 448.0
            bits = sf.to(tl.int32, bitcast=True)
            exp = ((bits >> 23) & 0xFF) + tl.where((bits & 0x7FFFFF) != 0, 1, 0)
            exp = tl.minimum(tl.maximum(exp, 1), 254)
            sf = (exp << 23).to(tl.float32, bitcast=True)
            q = x * (1.0 / sf)
            tl.store(
                q_ptr + pid.to(tl.int64) * BLK + tl.arange(0, BLK),
                q.to(q_ptr.dtype.element_ty),
            )
            tl.store(sf_ptr + m.to(tl.int64) * N_BLOCKS + b, sf)

        _PAD_QUANT_KERNEL = _kernel
    return _PAD_QUANT_KERNEL


def _pad_and_cast_fp8(
    grad_h: torch.Tensor, src: torch.Tensor, dest: torch.Tensor, m_max: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(a_fp8 [m_max, K], a_sf fp32 [m_max, K/128])`` from real rows of grad_h.

    ``src`` maps padded row -> source row (-1 for padding); ``dest`` is the
    inverse map. Falls back to the torch scatter + per_token_cast chain when
    triton is unavailable.
    """
    K = grad_h.shape[1]
    from .triton_nvfp4 import triton_available

    if triton_available() and K % 128 == 0:
        q = torch.empty((m_max, K), dtype=torch.float8_e4m3fn, device=grad_h.device)
        sf = torch.empty((m_max, K // 128), dtype=torch.float32, device=grad_h.device)
        grad_c = grad_h.contiguous()
        n_blocks = K // 128
        _pad_quant_kernel()[(m_max * n_blocks,)](
            grad_c, src, q, sf, N_BLOCKS=n_blocks, BLK=128
        )
        return q, sf

    from deep_gemm.utils import per_token_cast_to_fp8

    a_pad = grad_h.new_zeros((m_max, K))
    a_pad[dest] = grad_h
    return per_token_cast_to_fp8(a_pad, use_ue8m0=True)


class _Fp8Entry(NamedTuple):
    # transposed weight [E, dim2, dim1] as fp8 with 128x128-block ue8m0 scales,
    # so dX = grad_h @ W runs as the NT grouped GEMM A @ B^T
    w_fp8: torch.Tensor
    w_sf: torch.Tensor
    alignment: int


# Keyed like fp4_cute_ops._ENGINE_CACHE; each entry only lives as long as the
# frozen weight it was built from (the qdata view keeps the storage alive).
_FP8_CACHE: dict = {}


def _cache_key(base_weight) -> tuple:
    qdata = base_weight.qdata
    return (qdata.data_ptr(), tuple(qdata.shape), qdata.device.index)


def _get_fp8_cache(base_weight) -> _Fp8Entry:
    key = _cache_key(base_weight)
    entry = _FP8_CACHE.get(key)
    if entry is None:
        dg = _deep_gemm()
        from deep_gemm.utils import ceil_div, per_block_cast_to_fp8

        from .fp4_cute_ops import dequantize_engine_weight

        # one-time per frozen weight; must match the forward's folded operands
        w = dequantize_engine_weight(base_weight)
        wt = w.transpose(1, 2).contiguous()  # [E, dim2, dim1]
        e, n, k = wt.shape
        w_fp8 = torch.empty_like(wt, dtype=torch.float8_e4m3fn)
        w_sf = torch.empty(
            (e, ceil_div(n, 128), ceil_div(k, 128)),
            dtype=torch.float32,
            device=wt.device,
        )
        for i in range(e):
            w_fp8[i], w_sf[i] = per_block_cast_to_fp8(wt[i], use_ue8m0=True)
        alignment = dg.get_mk_alignment_for_contiguous_layout()
        entry = _Fp8Entry(w_fp8, w_sf, alignment)
        _FP8_CACHE[key] = entry
    return entry


def _dx_plan(expert_offsets: torch.Tensor, T: int, E: int, align: int) -> tuple:
    """``(dest, m_indices, src, m_max)`` for the padded contiguous layout.

    Cached as an attribute on the offsets tensor: the up and down projections
    of one layer save the SAME offsets object for backward, so the plan builds
    once per layer per step and dies with the tensor (no stale-pointer risk).
    The dozen tiny index kernels here cost more than the GEMM otherwise.
    """
    plan = getattr(expert_offsets, "_dg_dx_plan", None)
    if plan is not None and plan[3] == T + E * align:
        return plan

    device = expert_offsets.device
    offsets = expert_offsets.to(dtype=torch.int64)
    counts = offsets[1:] - offsets[:-1]
    padded = torch.div(counts + (align - 1), align, rounding_mode="floor") * align
    pad_starts = torch.cat([padded.new_zeros(1), padded.cumsum(0)])[:-1]

    # worst case each segment wastes < align rows; sized statically, no sync
    m_max = T + E * align
    row = torch.arange(T, device=device)
    eid = torch.repeat_interleave(torch.arange(E, device=device), counts, output_size=T)
    dest = pad_starts[eid] + (row - offsets[eid])

    maps = torch.full((2, m_max), -1, dtype=torch.int32, device=device)
    m_indices, src = maps[0], maps[1]
    m_indices[dest] = eid.to(torch.int32)
    src[dest] = row.to(torch.int32)

    plan = (dest, m_indices, src, m_max)
    expert_offsets._dg_dx_plan = plan
    return plan


def grouped_fp8_dx(
    grad_h: torch.Tensor, base_weight, expert_offsets: torch.Tensor
) -> torch.Tensor:
    """``dx[start:end] = g_e @ W_e`` through the fp8 weight cache.

    ``expert_offsets`` is ``[E+1]`` cumulative (device). Returns ``[T, dim2]``
    bf16, row-aligned with ``grad_h`` ``[T, dim1]``.
    """
    dg = _deep_gemm()
    entry = _get_fp8_cache(base_weight)

    T = grad_h.shape[0]
    E, N, K = entry.w_fp8.shape
    dest, m_indices, src, m_max = _dx_plan(expert_offsets, T, E, entry.alignment)

    a_fp8, a_sf = _pad_and_cast_fp8(grad_h, src, dest, m_max)
    d = torch.empty((m_max, N), dtype=torch.bfloat16, device=grad_h.device)
    dg.m_grouped_fp8_gemm_nt_contiguous(
        (a_fp8, a_sf),
        (entry.w_fp8, entry.w_sf),
        d,
        m_indices,
        disable_ue8m0_cast=False,
    )
    return d.index_select(0, dest)
