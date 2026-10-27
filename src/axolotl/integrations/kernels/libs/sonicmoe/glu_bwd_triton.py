# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""``dh += glu'(h)[d]`` on sonic-moe's interleaved ``[gate0, up0, gate1, up1, ...]`` preact.

Adds the gate/up gradient of an extra upstream ``d [TK, I]`` to the ``dh [TK, 2I]`` that quack's
``gemm_dgated`` wrote, with quack's ``dswiglu`` / ``dgeglu`` (tanh-approximate GELU) math in
fp32. Rows at or past ``cu_seqlens[-1]`` (EP sentinel pairs) are neither read nor written.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _tanh(z):
    return 2.0 * tl.sigmoid(2.0 * z) - 1.0


@triton.jit
def _glu_bwd_add_kernel(
    dh_ptr,
    h_ptr,
    d_ptr,
    cu_ptr,
    num_slots,
    num_experts,
    inter,
    stride_h,
    stride_d,
    GEGLU: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_i = tl.program_id(1)
    num_valid = tl.load(cu_ptr + num_experts)
    rows = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    cols = (pid_i * BLOCK_I + tl.arange(0, BLOCK_I)).to(tl.int64)
    mask = ((rows < num_valid) & (rows < num_slots))[:, None] & (cols < inter)[None, :]

    gate_off = rows[:, None] * stride_h + 2 * cols[None, :]
    gate = tl.load(h_ptr + gate_off, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(h_ptr + gate_off + 1, mask=mask, other=0.0).to(tl.float32)
    d = tl.load(
        d_ptr + rows[:, None] * stride_d + cols[None, :], mask=mask, other=0.0
    ).to(tl.float32)

    if GEGLU:
        c1 = 0.7978845608028654
        c2 = 0.044715 * c1
        gate_sq = gate * gate
        t = _tanh(gate * (c1 + c2 * gate_sq))
        half = 0.5 + 0.5 * t
        act = gate * half
        dact = half + 0.5 * gate * (1.0 - t * t) * (c1 + 3.0 * c2 * gate_sq)
    else:
        sig = tl.sigmoid(gate)
        act = gate * sig
        dact = sig + act * (1.0 - sig)

    dgate = tl.load(dh_ptr + gate_off, mask=mask, other=0.0).to(tl.float32)
    dup = tl.load(dh_ptr + gate_off + 1, mask=mask, other=0.0).to(tl.float32)
    out_ty = dh_ptr.dtype.element_ty
    tl.store(dh_ptr + gate_off, (dgate + d * up * dact).to(out_ty), mask=mask)
    tl.store(dh_ptr + gate_off + 1, (dup + d * act).to(out_ty), mask=mask)


def glu_bwd_add_(
    dh: torch.Tensor,
    h: torch.Tensor,
    d: torch.Tensor,
    cu_seqlens: torch.Tensor,
    activation: str,
) -> None:
    if activation not in ("swiglu", "geglu"):
        raise ValueError(f"glu_bwd_add_ supports swiglu and geglu, got {activation!r}")
    num_slots, two_i = h.shape
    inter = two_i // 2
    assert dh.shape == h.shape and d.shape == (num_slots, inter)
    assert h.stride(1) == 1 and dh.stride() == h.stride() and d.stride(1) == 1
    if num_slots == 0:
        return
    block_m = 32
    block_i = min(128, triton.next_power_of_2(inter))
    grid = (triton.cdiv(num_slots, block_m), triton.cdiv(inter, block_i))
    _glu_bwd_add_kernel[grid](
        dh,
        h,
        d,
        cu_seqlens,
        num_slots,
        cu_seqlens.numel() - 1,
        inter,
        h.stride(0),
        d.stride(0),
        GEGLU=activation == "geglu",
        BLOCK_M=block_m,
        BLOCK_I=block_i,
        num_warps=4,
    )
