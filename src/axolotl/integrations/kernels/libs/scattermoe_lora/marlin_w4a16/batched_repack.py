"""Batched one-pass marlin repack (Triton): all experts in one launch.

``gptq_marlin_repack`` scatters a (64, 16) base tile into fixed output positions; the map is
probed once per process from the CUDA kernel itself (bijective, bit-exact). Repacking per expert
costs one kernel launch + transpose copies each; these kernels do every expert in one launch with
coalesced slab loads and an in-register permute, bit-identical to the loop.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

N_T, K_T = 64, 16  # marlin base tile — matches the 4-bit fused_dequant geometry
TILE = N_T * K_T

_REPACK4_INV_LUT: dict[torch.device, torch.Tensor] = {}


def _build_repack4_inv_lut(dev: torch.device) -> torch.Tensor:
    """Inverse of the 4-bit (64,16)-tile nibble scatter (``_build_base_scatter`` in backend.py):
    output nibble position -> source nibble position, so the kernel can build each output byte
    from two gathers (no read-modify-write on nibbles)."""
    if dev in _REPACK4_INV_LUT:
        return _REPACK4_INV_LUT[dev]
    from .backend import _build_base_scatter

    scatter = _build_base_scatter(dev).to(torch.int64)  # src nib -> out nib
    inv = torch.empty_like(scatter)
    inv[scatter] = torch.arange(TILE, device=dev)
    lut = inv.to(torch.int32)
    _REPACK4_INV_LUT[dev] = lut
    return lut


@triton.jit
def _batched_repack4_kernel(
    QD,  # [E, size_n, size_k/2] uint8 (packed nibbles, low-nibble-first)
    INV,  # [1024] int32: out nibble pos -> src nibble pos (n*16 + k within the tile)
    OUT,  # [E, size_n*size_k/2] uint8 (int32-viewable marlin pack)
    qd_se,
    qd_sn,
    qd_sk,
    out_se,
    NT_OUT: tl.constexpr,  # size_n // 64
    HALF_TILE: tl.constexpr,  # 512 output bytes per tile
    TILES_K: tl.constexpr,  # adjacent k-tiles per program (slab width)
):
    e = tl.program_id(0).to(tl.int64)
    t = tl.program_id(1).to(tl.int64)
    nt = t % NT_OUT
    kt0 = (t // NT_OUT) * TILES_K  # first of TILES_K adjacent k-tiles in this slab

    # load a (64 x TILES_K*8-byte) slab ONCE, coalesced (rows are contiguous byte runs)
    rows = tl.arange(0, 64)
    cols = tl.arange(0, TILES_K * 8)
    b = tl.load(
        QD
        + e * qd_se
        + (nt * 64 + rows)[:, None] * qd_sn
        + (kt0 * 8 + cols)[None, :] * qd_sk
    )
    # nibble order (low-nibble-first along k): flat position (n, s, k) = n*TILES_K*16 + s*16 + k
    nibs = tl.reshape(tl.interleave(b & 0xF, b >> 4), [64 * TILES_K * 16])
    inv = tl.load(
        INV + tl.arange(0, HALF_TILE * 2)
    )  # out nib -> (n*16 + k) within a tile
    n_src = inv // 16
    k_src = inv % 16
    for s in tl.static_range(TILES_K):
        idx = n_src * (TILES_K * 16) + s * 16 + k_src
        out_nibs = tl.gather(nibs, idx, 0)
        pairs = tl.reshape(out_nibs, [HALF_TILE, 2])
        lo, hi = tl.split(pairs)
        j = tl.arange(0, HALF_TILE)
        tile_off = ((kt0 + s) * NT_OUT + nt) * HALF_TILE
        tl.store(OUT + e * out_se + tile_off + j, lo | (hi << 4))


def batched_repack4(qdata: torch.Tensor, size_n: int, size_k: int) -> torch.Tensor:
    """Marlin 4-bit repack for ALL experts in one launch (bit-identical to the per-expert
    ``gptq_marlin_repack(..., 4, False)`` loop). ``qdata [E, size_n, size_k//2]`` uint8 packed
    low-nibble-first -> int32 pack ``[E, size_k//16, size_n*2]``."""
    E = qdata.size(0)
    assert size_n % N_T == 0 and size_k % K_T == 0, (size_n, size_k)
    lut = _build_repack4_inv_lut(qdata.device)
    out = torch.empty(E, size_n * size_k // 2, dtype=torch.uint8, device=qdata.device)
    nt_out = size_n // N_T
    k_tiles = size_k // K_T
    # slab several k-tiles per program: 1-tile programs are 512B of work — launch-bound at scale
    tiles_k = 8
    while k_tiles % tiles_k:
        tiles_k //= 2
    grid = (E, nt_out * (k_tiles // tiles_k))
    _batched_repack4_kernel[grid](
        qdata,
        lut,
        out,
        qdata.stride(0),
        qdata.stride(1),
        qdata.stride(2),
        out.stride(0),
        NT_OUT=nt_out,
        HALF_TILE=TILE // 2,
        TILES_K=tiles_k,
    )
    return out.view(torch.int32).reshape(E, size_k // 16, size_n * 2)
