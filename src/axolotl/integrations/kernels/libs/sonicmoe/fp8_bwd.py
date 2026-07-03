# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Optional fp8 backward-dX engine over DeepGEMM for the frozen NVFP4 base.

The bf16 backward dequantizes the whole expert weight every backward pass and
runs ``torch._grouped_mm``. This module instead caches the weights once as
fp8 e4m3 (128x128-block ue8m0 scales, built from the SAME folded dequant the
forward consumed) and runs dX through DeepGEMM's m-grouped contiguous fp8
GEMM (1x128 per-token dY scales), DeepSeek's validated training recipe.
Measured 1.5-1.8x over the bf16 grouped GEMM at Qwen3-30B backward shapes,
before counting the skipped per-backward dequant; dX rel err ~7e-4.

Engine chain: DeepGEMM if importable, else the bf16 dequant path (caller's
fallback). ``torch._scaled_grouped_mm`` is deliberately NOT in the chain: its
rowwise fp8 grouped kernel device-aborts on SM100 (torch 2.10+cu130, kernel
not built for sm100a) and kills the CUDA context. See ``DEEPGEMM.md`` for the
install steps (source build; CUTLASS symlinks into ``deep_gemm/include`` are
mandatory or every JIT compile fails).

``AXOLOTL_SONICMOE_NVFP4_BWD``: unset/``auto`` uses DeepGEMM when available,
``bf16`` disables it, ``deepgemm`` raises if unavailable.

DeepGEMM's contiguous layout needs every expert segment padded to its M block
alignment (224 on B200); rows are scattered into a padded buffer whose
padding carries ``m_indices == -1`` (skipped by the kernel) and gathered back
after. All index math stays on device; the buffer is sized ``T + E * align``
so no host sync is needed.
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

            alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
            deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
            _DEEP_GEMM = deep_gemm
        except Exception:
            _DEEP_GEMM = False
    return _DEEP_GEMM if _DEEP_GEMM is not False else None


def fp8_dx_supported(grad_h: torch.Tensor, base_weight) -> bool:
    mode = os.environ.get("AXOLOTL_SONICMOE_NVFP4_BWD", "auto")
    if mode == "bf16":
        return False
    if not (
        grad_h.is_cuda
        and grad_h.dtype == torch.bfloat16
        and is_nvfp4_param(base_weight)
        and base_weight.shape[-2] % 128 == 0  # K of the dX GEMM (dim1)
        and torch.cuda.get_device_capability(grad_h.device)[0] == 10
    ):
        if mode == "deepgemm":
            raise RuntimeError(
                "AXOLOTL_SONICMOE_NVFP4_BWD=deepgemm requires a CUDA bf16 grad, "
                "a packed NVFP4 base with dim1 % 128 == 0, and an SM100 GPU"
            )
        return False
    if _deep_gemm() is None:
        if mode == "deepgemm":
            raise RuntimeError(
                "AXOLOTL_SONICMOE_NVFP4_BWD=deepgemm but deep_gemm is not "
                "importable; see DEEPGEMM.md for the source-build steps"
            )
        return False
    return True


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


def grouped_fp8_dx(
    grad_h: torch.Tensor, base_weight, expert_offsets: torch.Tensor
) -> torch.Tensor:
    """``dx[start:end] = g_e @ W_e`` through the fp8 weight cache.

    ``expert_offsets`` is ``[E+1]`` cumulative (device). Returns ``[T, dim2]``
    bf16, row-aligned with ``grad_h`` ``[T, dim1]``.
    """
    dg = _deep_gemm()
    entry = _get_fp8_cache(base_weight)
    from deep_gemm.utils import per_token_cast_to_fp8

    T = grad_h.shape[0]
    E, N, K = entry.w_fp8.shape
    align = entry.alignment
    device = grad_h.device

    offsets = expert_offsets.to(device=device, dtype=torch.int64)
    counts = offsets[1:] - offsets[:-1]
    padded = torch.div(counts + (align - 1), align, rounding_mode="floor") * align
    pad_starts = torch.cat([padded.new_zeros(1), padded.cumsum(0)])[:-1]

    # worst case each segment wastes < align rows; sized statically, no sync
    m_max = T + E * align
    row = torch.arange(T, device=device)
    eid = torch.repeat_interleave(torch.arange(E, device=device), counts, output_size=T)
    dest = pad_starts[eid] + (row - offsets[eid])

    a_pad = grad_h.new_zeros((m_max, K))
    a_pad[dest] = grad_h
    m_indices = torch.full((m_max,), -1, dtype=torch.int32, device=device)
    m_indices[dest] = eid.to(torch.int32)

    a_fp8, a_sf = per_token_cast_to_fp8(a_pad, use_ue8m0=True)
    d = torch.empty((m_max, N), dtype=torch.bfloat16, device=device)
    dg.m_grouped_fp8_gemm_nt_contiguous(
        (a_fp8, a_sf),
        (entry.w_fp8, entry.w_sf),
        d,
        m_indices,
        disable_ue8m0_cast=False,
    )
    return d.index_select(0, dest)
