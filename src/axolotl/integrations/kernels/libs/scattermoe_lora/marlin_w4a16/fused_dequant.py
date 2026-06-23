"""Fused Triton kernels: marlin int32 layout -> bf16 / fp8_e4m3, without intermediate reconstruction.

The backward dequant in the marlin training path needs to reconstruct the expert weights from the
marlin qweight cache (int32, tile-scattered nibbles) + the original NVFP4 block scales. A two-step
approach (reconstruct nibble array + call nvfp4_dequant_bf16) is ~12x slower due to the scattered
gather; these kernels fold both steps into one Triton pass with only sequential reads on qweight and
sequential writes on output, bringing cost to ~1.2-1.6x the direct nvfp4_dequant.

The marlin tile layout (N_t=64, K_t=16) scatters nibbles according to gptq_marlin_repack. A 1024-entry
LUT (base scatter, built once per process from _build_base_scatter in backend.py) maps each
(local_n, local_k) position within a base tile to its flat nibble position in the marlin output.
Larger (N, K) tile: offset = (k_tile * N_TILES + n_tile) * 1024 nibbles.
"""

from __future__ import annotations

import triton
import triton.language as tl

# Grid (C, N // 64, K // 16): one block per (expert-chunk, n-tile, k-tile), each a 64x16 tile.


@triton.jit
def _marlin_dequant_bf16_kernel(
    QW,  # [C, K*N//8] int32  (marlin flat)
    SC,  # [C, N, K//16] fp8_e4m3
    PT,  # [C] float32  per-tensor scale
    SCATTER,  # [1024] int32  base scatter LUT
    CB,  # [16] float32  fp4 codebook
    OUT,  # [C, N, K] bfloat16
    qw_stride_c: tl.constexpr,
    sc_stride_c: tl.constexpr,
    sc_stride_n: tl.constexpr,
    out_stride_c: tl.constexpr,
    out_stride_n: tl.constexpr,
    N_TILES: tl.constexpr,
    N_t: tl.constexpr,  # 64
    K_t: tl.constexpr,  # 16
    TILE_SIZE: tl.constexpr,  # N_t * K_t = 1024
):
    c_id = tl.program_id(0)
    n_tile = tl.program_id(1)
    k_tile = tl.program_id(2)

    local_idx = tl.arange(0, TILE_SIZE)
    local_n = local_idx // K_t
    local_k = local_idx % K_t

    local_marlin_pos = tl.load(SCATTER + local_n * K_t + local_k)
    tile_off = (k_tile * N_TILES + n_tile) * TILE_SIZE
    nib_pos = tile_off + local_marlin_pos

    words = tl.load(QW + c_id * qw_stride_c + nib_pos // 8)
    nibs = (words >> ((nib_pos % 8) * 4)) & 0xF

    pt = tl.load(PT + c_id)
    global_n = n_tile * N_t + local_n
    sc = tl.load(SC + c_id * sc_stride_c + global_n * sc_stride_n + k_tile).to(
        tl.float32
    )
    cb_val = tl.load(CB + nibs)

    out_val = (cb_val * sc * pt).to(tl.bfloat16)
    tl.store(
        OUT + c_id * out_stride_c + global_n * out_stride_n + k_tile * K_t + local_k,
        out_val,
    )


@triton.jit
def _marlin_dequant_fp8_kernel(
    QW,
    SC,
    PT,
    SCATTER,
    CB,
    OUT,
    qw_stride_c: tl.constexpr,
    sc_stride_c: tl.constexpr,
    sc_stride_n: tl.constexpr,
    out_stride_c: tl.constexpr,
    out_stride_n: tl.constexpr,
    N_TILES: tl.constexpr,
    N_t: tl.constexpr,
    K_t: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    c_id = tl.program_id(0)
    n_tile = tl.program_id(1)
    k_tile = tl.program_id(2)

    local_idx = tl.arange(0, TILE_SIZE)
    local_n = local_idx // K_t
    local_k = local_idx % K_t

    local_marlin_pos = tl.load(SCATTER + local_n * K_t + local_k)
    tile_off = (k_tile * N_TILES + n_tile) * TILE_SIZE
    nib_pos = tile_off + local_marlin_pos

    words = tl.load(QW + c_id * qw_stride_c + nib_pos // 8)
    nibs = (words >> ((nib_pos % 8) * 4)) & 0xF

    pt = tl.load(PT + c_id)
    global_n = n_tile * N_t + local_n
    sc = tl.load(SC + c_id * sc_stride_c + global_n * sc_stride_n + k_tile).to(
        tl.float32
    )
    cb_val = tl.load(CB + nibs)

    out_val = (cb_val * sc * pt).to(tl.float8e4nv)
    tl.store(
        OUT + c_id * out_stride_c + global_n * out_stride_n + k_tile * K_t + local_k,
        out_val,
    )


def marlin_dequant_bf16(qw_flat, orig_scale, pt, scatter_lut, cb, N, K, C):
    """Dequantize marlin int32 weight -> bf16 using the original NVFP4 block scales.

    qw_flat: [C, K*N//8] int32  (marlin flat layout)
    orig_scale: [C, N, K//16] fp8_e4m3  (original block scales, NOT marlin-processed)
    pt: [C] float32  per-tensor scale
    scatter_lut: [1024] int32  base scatter LUT (from _build_base_scatter)
    cb: [16] float32  fp4 codebook
    Returns: [C, N, K] bfloat16"""
    import torch

    out = torch.empty(C, N, K, device=qw_flat.device, dtype=torch.bfloat16)
    N_t, K_t = 64, 16
    N_TILES = N // N_t
    grid = (C, N_TILES, K // K_t)
    _marlin_dequant_bf16_kernel[grid](
        qw_flat,
        orig_scale,
        pt,
        scatter_lut,
        cb,
        out,
        int(qw_flat.stride(0)),
        int(orig_scale.stride(0)),
        int(orig_scale.stride(1)),
        int(out.stride(0)),
        int(out.stride(1)),
        N_TILES,
        N_t,
        K_t,
        N_t * K_t,
    )
    return out


def marlin_dequant_fp8(qw_flat, orig_scale, pt, scatter_lut, cb, N, K, C):
    """Dequantize marlin int32 weight -> fp8_e4m3 using the original NVFP4 block scales.

    Same signature as marlin_dequant_bf16; output dtype is float8_e4m3fn."""
    import torch

    out = torch.empty(C, N, K, device=qw_flat.device, dtype=torch.float8_e4m3fn)
    N_t, K_t = 64, 16
    N_TILES = N // N_t
    grid = (C, N_TILES, K // K_t)
    _marlin_dequant_fp8_kernel[grid](
        qw_flat,
        orig_scale,
        pt,
        scatter_lut,
        cb,
        out,
        int(qw_flat.stride(0)),
        int(orig_scale.stride(0)),
        int(orig_scale.stride(1)),
        int(out.stride(0)),
        int(out.stride(1)),
        N_TILES,
        N_t,
        K_t,
        N_t * K_t,
    )
    return out
