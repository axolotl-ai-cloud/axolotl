"""Batched one-pass marlin 8-bit repack (Triton): all experts, transposed read, one launch.

``gptq_marlin_repack`` scatters a (64, 16) base tile of bytes into 1024 fixed output positions;
the map is probed once per process (bijective, bit-exact — see ``_build_repack8_lut``). This
kernel fuses, for every expert at once, the W^T transpose (a strided READ from the original
``[E, N, K]`` fp8 weight — no intermediate ``.contiguous()`` copies) with the LUT scatter,
replacing the per-expert loop of 2 transpose copies + repack launch. Same output bit-for-bit.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

N_T, K_T = 64, 16  # marlin base tile (bytes) — matches the 4-bit fused_dequant geometry
TILE = N_T * K_T

_REPACK8_LUT: dict[torch.device, torch.Tensor] = {}


def _build_repack8_lut(dev: torch.device, ext) -> torch.Tensor:
    """Probe the (64, 16)-tile byte scatter of ``gptq_marlin_repack(num_bits=8)`` once per
    process: set one byte, find it in the output. Bijective by construction (asserted)."""
    if dev in _REPACK8_LUT:
        return _REPACK8_LUT[dev]
    perm = torch.empty(0, dtype=torch.int, device=dev)
    scatter = torch.full((TILE,), -1, dtype=torch.int64)
    for n in range(N_T):
        for k in range(K_T):
            qd = torch.zeros(N_T, K_T, dtype=torch.uint8, device=dev)
            qd[n, k] = 0xFF
            qw_i = qd.view(torch.int32).T.contiguous()
            out = ext.gptq_marlin_repack(qw_i, perm, K_T, N_T, 8, False)
            pos = (out.view(torch.uint8).flatten() == 0xFF).nonzero()
            assert pos.numel() == 1, f"repack8 probe ({n},{k}): {pos.numel()} hits"
            scatter[n * K_T + k] = pos.item()
    assert (scatter >= 0).all() and scatter.unique().numel() == TILE
    lut = scatter.to(dev, dtype=torch.int32)
    _REPACK8_LUT[dev] = lut
    return lut


@triton.jit
def _batched_repack8_t_kernel(
    WQ,  # [E, N, K] uint8 (fp8 bytes) — the FORWARD-layout weight
    LUT,  # [1024] int32
    OUT,  # [E, N*K] uint8 (int32-viewable marlin pack of W^T)
    wq_se,
    wq_sn,
    wq_sk,
    out_se,
    NT_OUT: tl.constexpr,  # size_n(=K) // 64 tiles
    TILE_SZ: tl.constexpr,  # 1024
    NPROG: tl.constexpr,  # tiles per expert
):
    # flat 1-D grid: 2-D (E, tiles) exceeds the 65535 grid-Y limit once N/16 * K/64 > 65535
    # (hit at [12288, 6144]). int64: e*stride_e exceeds 2^31 at E=256 x [4096,4096] (same
    # overflow class as the nvfp4_dequant_bf16 E=256 bug).
    pid = tl.program_id(0).to(tl.int64)
    e = pid // NPROG
    t = pid % NPROG
    nt = t % NT_OUT
    kt = t // NT_OUT
    idx = tl.arange(0, TILE_SZ)
    ln = idx // 16  # local n within the tile (dx-weight n = fwd K index)
    lk = idx % 16  # local k within the tile (dx-weight k = fwd N index)
    # dx-weight element (n, k) = W[k, n]: transposed gather from the forward layout
    src = WQ + e * wq_se + (kt * 16 + lk) * wq_sn + (nt * 64 + ln) * wq_sk
    val = tl.load(src)
    pos = tl.load(LUT + idx)
    tile_off = (kt * NT_OUT + nt) * TILE_SZ
    tl.store(OUT + e * out_se + tile_off + pos, val)


@triton.jit
def _batched_repack8_fwd_kernel(
    WQ,  # [E, N, K] uint8 (fp8 bytes) — the FORWARD-layout weight
    LUT,  # [1024] int32
    OUT,  # [E, N*K] uint8 (int32-viewable marlin pack, forward orientation)
    wq_se,
    wq_sn,
    wq_sk,
    out_se,
    NT_OUT: tl.constexpr,  # size_n(=N) // 64 tiles
    TILE_SZ: tl.constexpr,  # 1024
    TILES_KT: tl.constexpr,  # adjacent k-tiles per program (slab width)
    NPROG: tl.constexpr,  # slabs per expert
):
    # Forward orientation: marlin B(k, n) = W[n, k], so a tile needs a (64 n-rows x 16 k-cols)
    # patch of W — 16-byte strided segments if gathered tile-by-tile. Slab TILES_KT adjacent
    # k-tiles instead: ONE coalesced (64 x TILES_KT*16 contiguous bytes) load, then per-tile
    # register permutes (same shape that made the 4-bit slab kernel 3x faster).
    pid = tl.program_id(0).to(tl.int64)
    e = pid // NPROG
    t = pid % NPROG
    nt = t % NT_OUT
    kt0 = (t // NT_OUT) * TILES_KT
    rows = tl.arange(0, 64)
    cols = tl.arange(0, TILES_KT * 16)
    b = tl.load(
        WQ
        + e * wq_se
        + (nt * 64 + rows)[:, None] * wq_sn
        + (kt0 * 16 + cols)[None, :] * wq_sk
    )
    flat = tl.reshape(b, [64 * TILES_KT * 16])
    idx = tl.arange(0, TILE_SZ)
    ln = idx // 16
    lk = idx % 16
    pos = tl.load(LUT + idx)
    for s in tl.static_range(TILES_KT):
        val = tl.gather(flat, ln * (TILES_KT * 16) + s * 16 + lk, 0)
        tile_off = ((kt0 + s) * NT_OUT + nt) * TILE_SZ
        tl.store(OUT + e * out_se + tile_off + pos, val)


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
    NPROG: tl.constexpr,  # slabs per expert
):
    # flat 1-D grid: 2-D (E, slabs) exceeds the 65535 grid-Y limit at large [N, K]
    pid = tl.program_id(0).to(tl.int64)
    e = pid // NPROG
    t = pid % NPROG
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
    nprog = nt_out * (k_tiles // tiles_k)
    _batched_repack4_kernel[(E * nprog,)](
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
        NPROG=nprog,
    )
    return out.view(torch.int32).reshape(E, size_k // 16, size_n * 2)


def batched_repack8_fwd(wq: torch.Tensor, ext) -> torch.Tensor:
    """Marlin 8-bit repack of ``W`` in FORWARD orientation for ALL experts in one launch.

    ``wq [E, N, K]`` float8/uint8 -> int32 pack shaped ``[E, K//16, N*4]``, bit-identical to
    ``gptq_marlin_repack(B, ..., num_bits=8)`` per expert where ``B[k, n] = W[n, k]`` (i.e. the
    ``x [Mt, K] -> [Mt, N]`` GEMM operand). Coalesced slab loads — ~2x the strided-view path."""
    E, N, K = wq.shape
    assert N % N_T == 0 and K % K_T == 0, f"repack8 fwd tiling: {(N, K)}"
    lut = _build_repack8_lut(wq.device, ext)
    wq_b = wq.view(torch.uint8)
    out = torch.empty(E, N * K, dtype=torch.uint8, device=wq.device)
    nt_out = N // N_T
    k_tiles = K // K_T
    tiles_kt = 8
    while k_tiles % tiles_kt:
        tiles_kt //= 2
    nprog = nt_out * (k_tiles // tiles_kt)
    _batched_repack8_fwd_kernel[(E * nprog,)](
        wq_b,
        lut,
        out,
        wq_b.stride(0),
        wq_b.stride(1),
        wq_b.stride(2),
        out.stride(0),
        NT_OUT=nt_out,
        TILE_SZ=TILE,
        TILES_KT=tiles_kt,
        NPROG=nprog,
    )
    return out.view(torch.int32).reshape(E, K // 16, N * 4)


def batched_repack8_transposed(wq: torch.Tensor, ext) -> torch.Tensor:
    """Marlin 8-bit repack of ``W^T`` for ALL experts in one launch.

    ``wq [E, N, K]`` float8/uint8 (forward layout) -> int32 pack ``[E, (K//16), N*4]``-shaped
    exactly as ``gptq_marlin_repack(W^T_bytes, ..., num_bits=8)`` per expert (bit-identical)."""
    E, N, K = wq.shape
    assert K % N_T == 0 and N % K_T == 0, (
        f"repack8 tiling: K%{N_T}, N%{K_T} of {(N, K)}"
    )
    lut = _build_repack8_lut(wq.device, ext)
    wq_b = wq.view(torch.uint8)
    out = torch.empty(E, N * K, dtype=torch.uint8, device=wq.device)
    nt_out = K // N_T
    nprog = nt_out * (N // K_T)
    _batched_repack8_t_kernel[(E * nprog,)](
        wq_b,
        lut,
        out,
        wq_b.stride(0),
        wq_b.stride(1),
        wq_b.stride(2),
        out.stride(0),
        NT_OUT=nt_out,
        TILE_SZ=TILE,
        NPROG=nprog,
    )
    # match the per-expert repack's output shape: [E, size_k // 16, size_n * 4] int32
    return out.view(torch.int32).reshape(E, N // 16, K * 4)
