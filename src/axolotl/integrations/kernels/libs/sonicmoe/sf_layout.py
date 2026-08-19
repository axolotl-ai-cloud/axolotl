"""Scale-factor (SF) layout prep for the SM100 blockscaled GEMM.

The quack/tcgen05 kernel consumes block scales in a ``(l, rm, rk, 512)`` layout:
each contiguous 512-byte block is one hardware-swizzled atom covering 128 MN
rows x 4 SF columns, with row r of a tile at byte
``(r % 32) * 16 + ((r // 32) % 4) * 4 + (sf_k % 4)``.

For varlen_m (grouped, per-expert M) the SFA storage is "dQaccum padded":
expert ``i``'s scale rows start at padded row ``(cu_seqlens[i] // 128 + i) * 128``
and the allocation is ``ceil(total_m / 128) + (E - 1)`` row-tiles. Operand data
(A/B/D) stays packed and unpadded; only the SF storage pads.

These mirror quack's ``pack_scale_2d_to_blocked_contig`` /
``create_blockscaled_varlen_m_operands`` (verified at quack source f4f54db0;
runtime pin quack-kernels==0.5.0).
"""

from __future__ import annotations

import torch

SF_TILE_ROWS = 128
SF_TILE_COLS = 4


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def pack_scales_blocked(scale_2d: torch.Tensor) -> torch.Tensor:
    """Rearrange ``(mn, sf_k)`` or ``(l, mn, sf_k)`` scales into ``(l, rm, rk, 512)``.

    Works for any 1-byte scale dtype (e4m3 for NVFP4, e8m0 for MX). Pads mn to
    a multiple of 128 and sf_k to a multiple of 4 with zeros.
    """
    if scale_2d.dim() == 2:
        scale_2d = scale_2d.unsqueeze(0)
    assert scale_2d.dim() == 3, f"expected (l, mn, sf_k), got {tuple(scale_2d.shape)}"
    assert scale_2d.element_size() == 1, "scale dtype must be 1 byte (e4m3/e8m0/u8)"
    orig_dtype = scale_2d.dtype
    l, mn, sf_k = scale_2d.shape
    rm = ceil_div(mn, SF_TILE_ROWS)
    rk = ceil_div(sf_k, SF_TILE_COLS)
    mn_pad, sf_k_pad = rm * SF_TILE_ROWS, rk * SF_TILE_COLS
    u8 = scale_2d.contiguous().view(torch.uint8)
    if (mn_pad, sf_k_pad) != (mn, sf_k):
        padded = torch.zeros(l, mn_pad, sf_k_pad, device=u8.device, dtype=torch.uint8)
        padded[:, :mn, :sf_k] = u8
    else:
        padded = u8
    blocks = padded.view(l, rm, SF_TILE_ROWS, rk, SF_TILE_COLS).permute(0, 1, 3, 2, 4)
    # split the 128 rows into (4 outer, 32 inner), then swap to (32, 4)
    blocks = blocks.reshape(l, rm, rk, 4, 32, SF_TILE_COLS).transpose(3, 4).contiguous()
    return blocks.view(l, rm, rk, 512).view(orig_dtype)


def varlen_padded_num_row_tiles(total_m: int, num_experts: int) -> int:
    return ceil_div(total_m, SF_TILE_ROWS) + (num_experts - 1)


def build_varlen_sfa(
    scale_rows: torch.Tensor, cu_seqlens: torch.Tensor
) -> torch.Tensor:
    """Build the dQaccum-padded SFA for varlen_m.

    scale_rows: ``(total_m, sf_k)`` per-row block scales, expert-sorted and packed
    (row order matching the A operand). cu_seqlens: ``(E+1,)`` int.

    Returns ``(1, total_padded_rm, rk, 512)`` in the blocked layout, where
    expert ``i``'s rows sit at padded tile offset ``cu_seqlens[i] // 128 + i``.
    """
    assert scale_rows.dim() == 2
    total_m, sf_k = scale_rows.shape
    num_experts = cu_seqlens.numel() - 1
    total_padded_rm = varlen_padded_num_row_tiles(total_m, num_experts)
    padded = torch.zeros(
        total_padded_rm * SF_TILE_ROWS,
        sf_k,
        dtype=scale_rows.dtype,
        device=scale_rows.device,
    )

    if scale_rows.is_cuda:
        # One scatter instead of a host-synced per-expert copy loop.
        cu = cu_seqlens.to(device=scale_rows.device, dtype=torch.long)
        starts = cu[:-1]
        counts = cu[1:] - starts
        seg = torch.repeat_interleave(
            torch.arange(num_experts, device=scale_rows.device), counts
        )
        row_in_seg = torch.arange(total_m, device=scale_rows.device) - starts[seg]
        dest = (starts[seg] // SF_TILE_ROWS + seg) * SF_TILE_ROWS + row_in_seg
        padded[dest] = scale_rows
        return pack_scales_blocked(padded.unsqueeze(0))

    cu = cu_seqlens.tolist()
    assert cu[0] == 0 and cu[-1] == total_m, (
        f"bad cu_seqlens {cu} for total_m={total_m}"
    )
    for i in range(num_experts):
        start, end = cu[i], cu[i + 1]
        row0 = (start // SF_TILE_ROWS + i) * SF_TILE_ROWS
        padded[row0 : row0 + (end - start)] = scale_rows[start:end]
    return pack_scales_blocked(padded.unsqueeze(0))


def fold_per_tensor_scale(
    block_scale: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    *,
    allow_underflow: bool = False,
) -> tuple[torch.Tensor, float]:
    """Fold a per-expert fp32 scale into e4m3 block scales: ``scale * pts -> e4m3``.

    block_scale: ``(E, N, sf_k)`` float8_e4m3fn. per_tensor_scale: ``(E,)``,
    ``(E,1,1)`` or scalar fp32. Returns ``(folded e4m3, max relative roundtrip
    error)``. Raises if any folded value saturates e4m3 (|v| > 448) or, unless
    ``allow_underflow``, rounds a nonzero scale to zero (an underflowed block
    dequantizes to exact zeros and counts as rel err 1.0); in that case pts
    must stay outside the block scale (per-expert alpha, a kernel epilogue
    change).
    """
    e4m3_max = 448.0
    pts = per_tensor_scale.float().reshape(-1)
    if pts.numel() == 1:
        pts = pts.expand(block_scale.shape[0])
    assert pts.numel() == block_scale.shape[0], (
        f"pts has {pts.numel()} entries for {block_scale.shape[0]} experts"
    )
    folded_f32 = block_scale.float() * pts.view(-1, 1, 1)
    if bool((folded_f32.abs() > e4m3_max).any()):
        raise ValueError("per_tensor_scale folding saturates e4m3 (>448)")
    folded = folded_f32.to(torch.float8_e4m3fn)
    roundtrip = folded.float()
    if not allow_underflow and bool(((roundtrip == 0) & (folded_f32 != 0)).any()):
        raise ValueError("per_tensor_scale folding underflows e4m3 to zero")
    nonzero = folded_f32 != 0
    rel_err = 0.0
    if bool(nonzero.any()):
        rel_err = float(
            (
                (roundtrip[nonzero] - folded_f32[nonzero]).abs()
                / folded_f32[nonzero].abs()
            ).max()
        )
    return folded, rel_err


def gate_up_interleave_perm(n2: int, device=None) -> torch.Tensor:
    """Row permutation mapping concat ``[gate; up]`` (N = 2I) to interleaved
    ``[g0, u0, g1, u1, ...]``, the pairing quack's gated epilogue consumes
    (postact col j = act(row 2j, row 2j+1))."""
    assert n2 % 2 == 0
    i = n2 // 2
    perm = torch.empty(n2, dtype=torch.long, device=device)
    perm[0::2] = torch.arange(i, device=device)
    perm[1::2] = torch.arange(i, device=device) + i
    return perm
