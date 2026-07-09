# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Triton NVFP4 codecs for the sonicmoe grouped path.

Two memory-bound kernels replacing multi-kernel pure-torch chains:

- :func:`dequant_nvfp4_triton`: packed E2M1 + e4m3 block scales (+ optional
  per-expert scale) -> bf16, one kernel. torchao's ``NVFP4Tensor.dequantize()``
  does this through a long-index table gather over every element and dominates
  the fp4_cute backward (the dX path dequantizes the whole weight).
- :func:`quantize_nvfp4_triton`: bf16 rows -> (packed u8, e4m3 block scales),
  one kernel. Replaces the ~10-kernel ``quantize_nvfp4_ref`` chain on the
  activation-quant hot path. Same rounding rules as the reference (scale =
  amax/6 clamped to 448, encode against the STORED e4m3 scale, ties at the
  E2M1 midpoints round down), so the fp32 oracle stays valid.

Both fall back to the callers' torch paths when triton is unavailable.
"""

from __future__ import annotations

import functools

import torch


@functools.lru_cache(maxsize=1)
def triton_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401
        import triton.language as tl  # noqa: F401
    except ImportError:
        return False
    return True


def _kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def _e2m1_value(m):
        # m in [0, 8): (0, 0.5, 1, 1.5, 2, 3, 4, 6)
        mant = tl.where(m % 2 == 1, 1.5, 1.0)
        exp = (m // 2 - 1).to(tl.float32)
        val = mant * tl.exp2(exp)
        val = tl.where(m == 0, 0.0, val)
        val = tl.where(m == 1, 0.5, val)
        return val

    @triton.jit
    def _dequant_kernel(
        q_ptr,
        s_ptr,
        pts_ptr,
        out_ptr,
        rows_per_expert,
        K2: tl.constexpr,  # bytes per row (K/2)
        SF_K: tl.constexpr,  # scales per row (K/16)
        HAS_PTS: tl.constexpr,
        BLOCK_B: tl.constexpr,  # bytes per program
    ):
        row = tl.program_id(0)
        blk = tl.program_id(1)
        offs_b = blk * BLOCK_B + tl.arange(0, BLOCK_B)
        mask_b = offs_b < K2
        b = tl.load(q_ptr + row * K2 + offs_b, mask=mask_b, other=0).to(tl.uint8)

        lo = (b & 0x0F).to(tl.int32)
        hi = ((b >> 4) & 0x0F).to(tl.int32)
        v_lo = _e2m1_value(lo & 7) * tl.where(lo >= 8, -1.0, 1.0)
        v_hi = _e2m1_value(hi & 7) * tl.where(hi >= 8, -1.0, 1.0)

        # byte i covers values 2i / 2i+1 -> scale index (2i)//16 == i//8
        offs_s = offs_b // 8
        sc_u8 = tl.load(s_ptr + row * SF_K + offs_s, mask=offs_s < SF_K, other=0)
        sc = sc_u8.to(tl.uint8).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        if HAS_PTS:
            sc = sc * tl.load(pts_ptr + row // rows_per_expert).to(tl.float32)

        out_lo = (v_lo * sc).to(out_ptr.dtype.element_ty)
        out_hi = (v_hi * sc).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + row * (2 * K2) + 2 * offs_b, out_lo, mask=mask_b)
        tl.store(out_ptr + row * (2 * K2) + 2 * offs_b + 1, out_hi, mask=mask_b)

    @triton.jit
    def _quant_kernel(
        x_ptr,
        q_ptr,
        s_ptr,
        K: tl.constexpr,
        SF_K: tl.constexpr,
        BLOCK_SF: tl.constexpr,  # scale blocks per program
    ):
        row = tl.program_id(0)
        blk = tl.program_id(1)
        offs_sf = blk * BLOCK_SF + tl.arange(0, BLOCK_SF)
        mask_sf = offs_sf < SF_K
        # [BLOCK_SF, 8, 2] element offsets: (block, byte-in-block, nibble)
        offs_v = (
            offs_sf[:, None, None] * 16
            + tl.arange(0, 8)[None, :, None] * 2
            + tl.arange(0, 2)[None, None, :]
        )
        mask_v = mask_sf[:, None, None]
        x = tl.load(x_ptr + row * K + offs_v, mask=mask_v, other=0.0).to(tl.float32)

        amax = tl.max(tl.max(tl.abs(x), axis=2), axis=1)
        scale = tl.minimum(amax / 6.0, 448.0)
        scale_e4m3 = scale.to(tl.float8e4nv)
        tl.store(
            s_ptr + row * SF_K + offs_sf,
            scale_e4m3.to(tl.uint8, bitcast=True),
            mask=mask_sf,
        )
        # encode against the STORED (rounded) scale, like the reference.
        # div_rn: triton's `/` may lower to reciprocal-multiply, which flips
        # values sitting exactly on E2M1 bucket boundaries vs the torch ref.
        sdec = scale_e4m3.to(tl.float32)
        sdec = tl.where(sdec == 0, 1.0, sdec)
        q = tl.math.div_rn(x, tl.broadcast_to(sdec[:, None, None], x.shape))

        a = tl.abs(q)
        idx = (
            (a > 0.25).to(tl.int32)
            + (a > 0.75).to(tl.int32)
            + (a > 1.25).to(tl.int32)
            + (a > 1.75).to(tl.int32)
            + (a > 2.5).to(tl.int32)
            + (a > 3.5).to(tl.int32)
            + (a > 5.0).to(tl.int32)
        )
        code = tl.where(q < 0, idx + 8, idx)

        # pack low nibble first: byte = lo + hi * 16, via a weighted sum over
        # the nibble axis (triton tensors do not support axis slicing)
        weight = tl.where(tl.arange(0, 2)[None, None, :] == 0, 1, 16)
        packed = tl.sum(code * weight, axis=2).to(tl.uint8)
        offs_p = offs_sf[:, None] * 8 + tl.arange(0, 8)[None, :]
        tl.store(q_ptr + row * (K // 2) + offs_p, packed, mask=mask_sf[:, None])

    @triton.jit
    def _quant_sfa_kernel(
        x_ptr,
        q_ptr,
        sfa_ptr,
        dest_ptr,  # [T] padded destination row per source row
        K: tl.constexpr,
        SF_K: tl.constexpr,
        RK: tl.constexpr,  # ceil(SF_K / 4), 512-byte tiles per row-tile
        BLOCK_SF: tl.constexpr,
    ):
        # identical quantization math to _quant_kernel, but the e4m3 scales are
        # stored straight into the dQaccum-padded swizzled SFA layout: padded
        # row p, sf col c -> tile (p//128, c//4), byte (p%32)*16 + ((p//32)%4)*4 + c%4
        row = tl.program_id(0)
        blk = tl.program_id(1)
        offs_sf = blk * BLOCK_SF + tl.arange(0, BLOCK_SF)
        mask_sf = offs_sf < SF_K
        offs_v = (
            offs_sf[:, None, None] * 16
            + tl.arange(0, 8)[None, :, None] * 2
            + tl.arange(0, 2)[None, None, :]
        )
        x = tl.load(
            x_ptr + row * K + offs_v, mask=mask_sf[:, None, None], other=0.0
        ).to(tl.float32)

        amax = tl.max(tl.max(tl.abs(x), axis=2), axis=1)
        scale = tl.minimum(amax / 6.0, 448.0)
        scale_e4m3 = scale.to(tl.float8e4nv)

        p = tl.load(dest_ptr + row)
        addr = (
            (p // 128) * (RK * 512)
            + (offs_sf // 4) * 512
            + (p % 32) * 16
            + ((p // 32) % 4) * 4
            + (offs_sf % 4)
        )
        tl.store(sfa_ptr + addr, scale_e4m3.to(tl.uint8, bitcast=True), mask=mask_sf)

        sdec = scale_e4m3.to(tl.float32)
        sdec = tl.where(sdec == 0, 1.0, sdec)
        q = tl.math.div_rn(x, tl.broadcast_to(sdec[:, None, None], x.shape))

        a = tl.abs(q)
        idx = (
            (a > 0.25).to(tl.int32)
            + (a > 0.75).to(tl.int32)
            + (a > 1.25).to(tl.int32)
            + (a > 1.75).to(tl.int32)
            + (a > 2.5).to(tl.int32)
            + (a > 3.5).to(tl.int32)
            + (a > 5.0).to(tl.int32)
        )
        code = tl.where(q < 0, idx + 8, idx)
        weight = tl.where(tl.arange(0, 2)[None, None, :] == 0, 1, 16)
        packed = tl.sum(code * weight, axis=2).to(tl.uint8)
        offs_p = offs_sf[:, None] * 8 + tl.arange(0, 8)[None, :]
        tl.store(q_ptr + row * (K // 2) + offs_p, packed, mask=mask_sf[:, None])

    @triton.jit
    def _rowscale_kernel(
        x_ptr,
        pts_ptr,  # fp32 [rows]
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        # in-place x[r, :] = bf16(f32(x[r, :]) * pts[r]); one pass instead of
        # float() -> mul -> to(bf16)
        row = tl.program_id(0)
        blk = tl.program_id(1)
        offs = blk * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)
        p = tl.load(pts_ptr + row)
        tl.store(x_ptr + row * N + offs, (x * p).to(x_ptr.dtype.element_ty), mask=mask)

    return _dequant_kernel, _quant_kernel, _quant_sfa_kernel, _rowscale_kernel


@functools.lru_cache(maxsize=1)
def _get_kernels():
    return _kernels()


def dequant_nvfp4_triton(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    per_tensor_scale: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """``qdata [..., K/2] u8`` + ``scale [..., K/16] e4m3`` -> ``[..., K]``.

    ``per_tensor_scale``, if given, has one entry per leading-dim slice (the
    ``[E, N, K]`` expert convention: entry ``e`` scales rows of slice ``e``).
    """
    dequant_kernel, _, _, _ = _get_kernels()
    if qdata.dtype != torch.uint8:
        qdata = qdata.view(torch.uint8)
    k2 = qdata.shape[-1]
    q2 = qdata.reshape(-1, k2).contiguous()
    s2 = scale.reshape(-1, scale.shape[-1]).contiguous().view(torch.uint8)
    rows = q2.shape[0]
    rows_per_expert = (
        rows // per_tensor_scale.numel() if per_tensor_scale is not None else rows
    )
    pts = (
        per_tensor_scale.reshape(-1).float().contiguous()
        if per_tensor_scale is not None
        else q2.new_zeros(1, dtype=torch.float32)
    )
    out = torch.empty(rows, 2 * k2, dtype=out_dtype, device=qdata.device)
    BLOCK_B = 512
    grid = (rows, (k2 + BLOCK_B - 1) // BLOCK_B)
    dequant_kernel[grid](
        q2,
        s2,
        pts,
        out,
        rows_per_expert,
        K2=k2,
        SF_K=scale.shape[-1],
        HAS_PTS=per_tensor_scale is not None,
        BLOCK_B=BLOCK_B,
    )
    return out.view(*qdata.shape[:-1], 2 * k2)


def rowscale_inplace_triton(x: torch.Tensor, row_scale: torch.Tensor) -> torch.Tensor:
    """In-place ``x[r] = x.dtype(f32(x[r]) * row_scale[r])``; bit-identical to
    ``(x.float() * row_scale[:, None]).to(x.dtype)`` in one pass."""
    _, _, _, rowscale_kernel = _get_kernels()
    assert x.dim() == 2 and x.is_contiguous()
    rows, n = x.shape
    BLOCK_N = 1024
    grid = (rows, (n + BLOCK_N - 1) // BLOCK_N)
    rowscale_kernel[grid](x, row_scale.contiguous(), N=n, BLOCK_N=BLOCK_N)
    return x


def quantize_nvfp4_triton(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``x [T, K]`` (K % 16 == 0) -> ``(packed u8 [T, K/2], scale e4m3 [T, K/16])``."""
    _, quant_kernel, _, _ = _get_kernels()
    assert x.dim() == 2 and x.shape[-1] % 16 == 0
    t, k = x.shape
    sf_k = k // 16
    x = x.contiguous()
    packed = torch.empty(t, k // 2, dtype=torch.uint8, device=x.device)
    scale_u8 = torch.empty(t, sf_k, dtype=torch.uint8, device=x.device)
    BLOCK_SF = 64
    grid = (t, (sf_k + BLOCK_SF - 1) // BLOCK_SF)
    quant_kernel[grid](x, packed, scale_u8, K=k, SF_K=sf_k, BLOCK_SF=BLOCK_SF)
    return packed, scale_u8.view(torch.float8_e4m3fn)


def quantize_rows_fused_sfa_triton(
    x: torch.Tensor, cu_seqlens: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize expert-sorted rows AND write the dQaccum-padded swizzled SFA in
    one kernel: ``(packed u8 [T, K/2], sfa e4m3 (1, rm, rk, 512))``.

    Byte-identical to ``quantize_nvfp4_triton`` + ``sf_layout.build_varlen_sfa``,
    minus the scatter and the pack/permute chain. Requires ``(K/16) % 4 == 0``.
    """
    from .sf_layout import SF_TILE_ROWS, varlen_padded_num_row_tiles

    _, _, quant_sfa_kernel, _ = _get_kernels()
    assert x.dim() == 2 and x.shape[-1] % 16 == 0
    t, k = x.shape
    sf_k = k // 16
    assert sf_k % 4 == 0
    rk = sf_k // 4
    num_experts = cu_seqlens.numel() - 1

    cu = cu_seqlens.to(device=x.device, dtype=torch.long)
    starts = cu[:-1]
    counts = cu[1:] - starts
    seg = torch.repeat_interleave(torch.arange(num_experts, device=x.device), counts)
    dest = (
        (starts[seg] // SF_TILE_ROWS + seg) * SF_TILE_ROWS
        + torch.arange(t, device=x.device)
        - starts[seg]
    ).to(torch.int32)

    total_rm = varlen_padded_num_row_tiles(t, num_experts)
    x = x.contiguous()
    packed = torch.empty(t, k // 2, dtype=torch.uint8, device=x.device)
    sfa = torch.zeros(1, total_rm, rk, 512, dtype=torch.uint8, device=x.device)
    BLOCK_SF = 64
    grid = (t, (sf_k + BLOCK_SF - 1) // BLOCK_SF)
    quant_sfa_kernel[grid](
        x, packed, sfa, dest, K=k, SF_K=sf_k, RK=rk, BLOCK_SF=BLOCK_SF
    )
    return packed, sfa.view(torch.float8_e4m3fn)
