"""Packed-snapshot code statistics: 4-codes-per-byte packing, flip and zero counts.

Monitoring compares two 2-bit snapshots instead of two dequantized weights, so the whole
cost is `numel / 4` bytes of memory and one small reduction every N steps.
"""

from __future__ import annotations

import torch

from ...quant import CODES_PER_BYTE
from ._common import HAVE_TRITON, num_warps_for, require_cuda

if HAVE_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _pack_codes_kernel(
        codes_ptr,
        out_ptr,
        numel,
        n_bytes,
        LANES: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        byte_idx = pid * BLOCK + tl.arange(0, BLOCK)
        byte_mask = byte_idx < n_bytes

        lanes = tl.arange(0, LANES)
        offs = byte_idx[:, None] * LANES + lanes[None, :]
        # padding lanes read as -1 so they store as 0, the same value `unpack` drops
        codes = tl.load(codes_ptr + offs, mask=offs < numel, other=-1).to(tl.int32)
        packed = tl.sum((codes + 1) << (2 * lanes)[None, :], axis=1)

        tl.store(out_ptr + byte_idx, packed.to(tl.uint8), mask=byte_mask)

    @triton.jit
    def _code_stats_kernel(
        prev_ptr,
        now_ptr,
        flips_ptr,
        zeros_ptr,
        n_bytes,
        LANES: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        byte_idx = pid * BLOCK + tl.arange(0, BLOCK)
        mask = byte_idx < n_bytes

        # out-of-range bytes read as 0: identical in both snapshots, and never a zero code
        prev = tl.load(prev_ptr + byte_idx, mask=mask, other=0).to(tl.int32)
        now = tl.load(now_ptr + byte_idx, mask=mask, other=0).to(tl.int32)

        shifts = 2 * tl.arange(0, LANES)[None, :]
        prev_codes = (prev[:, None] >> shifts) & 3
        now_codes = (now[:, None] >> shifts) & 3

        tl.store(flips_ptr + pid, tl.sum((prev_codes != now_codes).to(tl.int32)))
        tl.store(zeros_ptr + pid, tl.sum((now_codes == 1).to(tl.int32)))


def pack_codes(codes: torch.Tensor) -> torch.Tensor:
    """Fused `quant.pack_codes`: pack ternary codes 4-per-byte into uint8.

    Args:
        codes: int8 codes in `{-1, 0, +1}` on a CUDA device.

    Returns:
        Flat uint8 tensor of length `ceil(codes.numel() / CODES_PER_BYTE)`.

    Raises:
        RuntimeError: If Triton is unavailable or `codes` is not on CUDA.
    """
    require_cuda(codes)
    codes = codes.contiguous()
    n_bytes = -(-codes.numel() // CODES_PER_BYTE)
    packed = torch.empty(n_bytes, dtype=torch.uint8, device=codes.device)
    if n_bytes == 0:
        return packed

    block = 1024
    _pack_codes_kernel[(triton.cdiv(n_bytes, block),)](
        codes,
        packed,
        codes.numel(),
        n_bytes,
        LANES=CODES_PER_BYTE,
        BLOCK=block,
        num_warps=num_warps_for(block),
    )
    return packed


def code_stats(
    packed_prev: torch.Tensor, packed_now: torch.Tensor, numel: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Count changed codes and zero codes between two packed snapshots in one pass.

    Args:
        packed_prev: Packed snapshot from a previous step.
        packed_now: Packed snapshot from the current step, same length.
        numel: Number of codes the snapshots hold.

    Returns:
        `(flips, zeros, total)` as 0-dim int64 tensors; padding lanes contribute to
        neither count.

    Raises:
        RuntimeError: If Triton is unavailable or a snapshot is not on CUDA.
        ValueError: If the snapshots have different lengths.
    """
    require_cuda(packed_prev, packed_now)
    if packed_prev.shape != packed_now.shape:
        raise ValueError(
            f"snapshot length mismatch: {packed_prev.shape} vs {packed_now.shape}"
        )
    packed_prev = packed_prev.contiguous()
    packed_now = packed_now.contiguous()
    # torch.full, not torch.tensor: a host scalar copy would sync the stream every call
    total = torch.full((), numel, dtype=torch.int64, device=packed_now.device)
    n_bytes = packed_now.numel()
    if n_bytes == 0:
        zero = torch.zeros((), dtype=torch.int64, device=packed_now.device)
        return zero, zero.clone(), total

    block = 1024
    n_programs = triton.cdiv(n_bytes, block)
    flips = torch.empty(n_programs, dtype=torch.int32, device=packed_now.device)
    zeros = torch.empty_like(flips)
    _code_stats_kernel[(n_programs,)](
        packed_prev,
        packed_now,
        flips,
        zeros,
        n_bytes,
        LANES=CODES_PER_BYTE,
        BLOCK=block,
        num_warps=num_warps_for(block),
    )
    return flips.sum(dtype=torch.int64), zeros.sum(dtype=torch.int64), total


def flip_count(packed_prev: torch.Tensor, packed_now: torch.Tensor) -> torch.Tensor:
    """Fused `quant.flip_count`: number of codes differing between two snapshots."""
    flips, _, _ = code_stats(
        packed_prev, packed_now, packed_now.numel() * CODES_PER_BYTE
    )
    return flips
