# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Expert-parallel merge-aware NVFP4 forward (``grouped_moe_merge_aware_ep_forward``).

After DeepEP dispatch each rank sees only its local experts, with routed ids in
``[0, E_local)`` and ``-1`` for slots owned by other ranks. These tests prove, on
CPU, that the sentinel-aware local forward (a) matches the validated single-GPU
reference when nothing is sharded, and (b) the per-rank sharded forwards sum back
to the single-rank forward, i.e. training merge-aware LoRA under EP computes the
same thing as on one GPU.

Runs entirely on CPU (the merge-aware snap falls back to torchao's reference).
"""

import pytest
import torch

from axolotl.integrations.kernels.libs.sonicmoe.nvfp4_lora import (
    grouped_moe_merge_aware_ep_forward,
    grouped_moe_reference_forward,
    set_merge_aware_enabled,
)


def _make_nvfp4_base(E, n, k):
    """Packed per-expert NVFP4 base ``[E, n, k]`` with per-expert pts, orig f32."""
    from torchao.prototype.mx_formats.nvfp4_tensor import (
        NVFP4Tensor,
        per_tensor_amax_to_scale,
    )

    w = torch.randn(E, n, k) * k**-0.5
    qd, sc, pts = [], [], []
    for e in range(E):
        p = per_tensor_amax_to_scale(w[e].abs().max())
        nv = NVFP4Tensor.to_nvfp4(w[e].contiguous(), per_tensor_scale=p)
        qd.append(nv.qdata)
        sc.append(nv.scale)
        pts.append(p)
    return NVFP4Tensor(
        torch.stack(qd),
        torch.stack(sc),
        16,
        torch.float32,
        per_tensor_scale=torch.stack(pts).view(-1, 1, 1),
    )


def _slice_experts(nv, lo, hi):
    """This rank's local-expert slice of a packed NVFP4 base (dim-0)."""
    return type(nv)(
        nv.qdata[lo:hi],
        nv.scale[lo:hi],
        nv.block_size,
        nv.orig_dtype,
        per_tensor_scale=nv.per_tensor_scale[lo:hi],
    )


def _make_lora(E, dim1, dim2, r):
    """PEFT rank-major LoRA: A ``[r*E, dim2]``, B ``[dim1, r*E]``."""
    return torch.randn(r * E, dim2), torch.randn(dim1, r * E)


def _slice_lora(A, B, lo, hi, E, r):
    """Local-expert slice in PEFT rank-major layout (mirrors _ep_local_expert_lora)."""
    a = A[lo * r : hi * r, :]
    out = B.shape[0]
    b = B.reshape(out, r, E)[:, :, lo:hi].reshape(out, r * (hi - lo))
    return a, b


def _routing(T, K, E, seed):
    g = torch.Generator().manual_seed(seed)
    idx = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(T)])
    wts = torch.rand(T, K, generator=g)
    return idx, wts / wts.sum(-1, keepdim=True)


@pytest.fixture(autouse=True)
def _merge_aware_on():
    set_merge_aware_enabled(True)
    yield
    set_merge_aware_enabled(False)


def test_ep_forward_matches_reference_without_sharding():
    """E_local == E_global, no sentinels: EP forward == the single-GPU reference."""
    pytest.importorskip("torchao")
    torch.manual_seed(0)
    E, H, I, r, K, T = 4, 32, 16, 2, 2, 13  # noqa: E741
    s1, s2 = 0.5, 0.75
    w1, w2 = _make_nvfp4_base(E, 2 * I, H), _make_nvfp4_base(E, H, I)
    A1, B1 = _make_lora(E, 2 * I, H, r)
    A2, B2 = _make_lora(E, H, I, r)
    x = torch.randn(T, H)
    idx, wts = _routing(T, K, E, seed=1)

    ref = grouped_moe_reference_forward(
        x,
        idx,
        wts,
        w1,
        None,
        w2,
        None,
        (A1, B1),
        (A2, B2),
        E,
        act="silu",
        backend="dequant",
        concat=True,
        scaling1=s1,
        scaling2=s2,
    )
    ep = grouped_moe_merge_aware_ep_forward(
        x,
        idx,
        wts,
        w1,
        None,
        w2,
        None,
        (A1, B1),
        (A2, B2),
        E,
        act="silu",
        concat=True,
        scaling1=s1,
        scaling2=s2,
    )
    assert torch.allclose(ref, ep, atol=1e-5, rtol=1e-4)


def test_ep_sharded_forwards_sum_to_single_rank():
    """ep_size=2: per-rank local forwards (remote experts -> -1) sum to the full forward."""
    pytest.importorskip("torchao")
    torch.manual_seed(2)
    E, H, I, r, K, T, ep_size = 4, 32, 16, 2, 2, 17, 2  # noqa: E741
    s1, s2 = 0.5, 0.75
    E_local = E // ep_size
    w1, w2 = _make_nvfp4_base(E, 2 * I, H), _make_nvfp4_base(E, H, I)
    A1, B1 = _make_lora(E, 2 * I, H, r)
    A2, B2 = _make_lora(E, H, I, r)
    x = torch.randn(T, H)
    idx, wts = _routing(T, K, E, seed=3)

    full = grouped_moe_reference_forward(
        x,
        idx,
        wts,
        w1,
        None,
        w2,
        None,
        (A1, B1),
        (A2, B2),
        E,
        act="silu",
        backend="dequant",
        concat=True,
        scaling1=s1,
        scaling2=s2,
    )

    acc = torch.zeros_like(full)
    for rk in range(ep_size):
        lo, hi = rk * E_local, (rk + 1) * E_local
        # global id in this rank's block -> local id; everything else -> sentinel.
        local_idx = torch.where(
            (idx >= lo) & (idx < hi), idx - lo, idx.new_full(idx.shape, -1)
        )
        w1l, w2l = _slice_experts(w1, lo, hi), _slice_experts(w2, lo, hi)
        A1l, B1l = _slice_lora(A1, B1, lo, hi, E, r)
        A2l, B2l = _slice_lora(A2, B2, lo, hi, E, r)
        acc = acc + grouped_moe_merge_aware_ep_forward(
            x,
            local_idx,
            wts,
            w1l,
            None,
            w2l,
            None,
            (A1l, B1l),
            (A2l, B2l),
            E_local,
            act="silu",
            concat=True,
            scaling1=s1,
            scaling2=s2,
        )

    assert torch.allclose(full, acc, atol=1e-5, rtol=1e-4)
