# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""End-to-end tests for the sonicmoe NVFP4 grouped forward path.

Validates routing gather/combine + the grouped LoRA MLP orchestration against a
naive per-token MoE oracle (materialized W_eff), forward and backward, on CPU.
"""

import pytest
import torch
import torch.nn.functional as F

from axolotl.integrations.kernels.libs.sonicmoe.nvfp4_lora import (
    combine_expert_outputs,
    grouped_moe_reference_forward,
    route_and_group,
)


def _make_peft_lora(E, r, dim1, dim2, *, dtype, seed):
    """Build (lora_A, lora_B) in PEFT rank-major layout + per-expert 3d views."""
    g = torch.Generator().manual_seed(seed)
    A_3d = torch.randn(E, r, dim2, generator=g, dtype=dtype, requires_grad=True)
    B_3d = torch.randn(E, dim1, r, generator=g, dtype=dtype, requires_grad=True)
    return A_3d, B_3d


def _peft_layout(A_3d, B_3d):
    E, r, dim2 = A_3d.shape
    _, dim1, _ = B_3d.shape
    lora_A = A_3d.reshape(E * r, dim2)
    lora_B = B_3d.permute(1, 2, 0).reshape(dim1, E * r)
    return lora_A, lora_B


def _swiglu_concat(h):
    i = h.shape[-1] // 2
    return h[..., i:] * F.silu(h[..., :i])


def _oracle_forward(
    hidden_states,
    top_k_index,
    top_k_weights,
    w1,
    w2,
    A1_3d,
    B1_3d,
    s1,
    A2_3d,
    B2_3d,
    s2,
):
    """Naive per-token MoE with materialized W_eff per expert."""
    T, K = top_k_index.shape
    H = hidden_states.shape[-1]
    out = hidden_states.new_zeros((T, H))
    for t in range(T):
        for k in range(K):
            e = int(top_k_index[t, k])
            x = hidden_states[t]
            w1_eff = w1[e] + s1 * (B1_3d[e] @ A1_3d[e])  # [2I, H]
            h = x @ w1_eff.transpose(0, 1)  # [2I]
            a = _swiglu_concat(h)  # [I]
            w2_eff = w2[e] + s2 * (B2_3d[e] @ A2_3d[e])  # [H, I]
            y = a @ w2_eff.transpose(0, 1)  # [H]
            out[t] = out[t] + top_k_weights[t, k] * y
    return out


def test_route_and_group_roundtrip():
    torch.manual_seed(0)
    T, K, H, E = 6, 2, 4, 3
    hs = torch.randn(T, H, dtype=torch.float64)
    idx = torch.randint(0, E, (T, K))
    w = torch.rand(T, K, dtype=torch.float64)

    x_g, offsets, gather_idx, w_g = route_and_group(hs, idx, w, E)

    assert offsets.shape == (E + 1,)
    assert int(offsets[-1]) == T * K
    # rows are sorted by expert
    counts = torch.bincount(idx.reshape(-1), minlength=E)
    assert torch.equal(offsets[1:] - offsets[:-1], counts)
    # combine with unit outputs and unit weights reconstructs per-token top-k counts
    ones = torch.ones(T * K, H, dtype=torch.float64)
    combined = combine_expert_outputs(
        ones, gather_idx, torch.ones(T * K, dtype=torch.float64), T
    )
    assert torch.allclose(combined, torch.full((T, H), float(K), dtype=torch.float64))


def test_nvfp4_forward_rejects_expert_parallelism():
    # EP is unfinished: base sharded to local experts, routing/LoRA still global.
    from types import SimpleNamespace

    from axolotl.integrations.kernels.libs.sonicmoe.experts import (
        _sonicmoe_nvfp4_forward,
    )

    fake_self = SimpleNamespace(
        is_transposed=False, num_experts=4, num_experts_global=8
    )
    dummy = torch.zeros(2, 4)
    with pytest.raises(NotImplementedError, match="expert parallelism"):
        _sonicmoe_nvfp4_forward(
            fake_self, dummy, dummy, dummy, dummy, None, dummy, None, None, None
        )


def test_route_and_group_rejects_out_of_range_expert():
    # A routed id >= num_experts (e.g. global ids under expert parallelism) would
    # otherwise silently truncate the offsets; the path must reject it loudly.
    hs = torch.randn(4, 3, dtype=torch.float64)
    idx = torch.tensor([[0, 5], [1, 2], [0, 1], [2, 0]])  # id 5 >= num_experts=3
    w = torch.rand(4, 2, dtype=torch.float64)
    with pytest.raises(NotImplementedError, match="expert parallelism"):
        route_and_group(hs, idx, w, num_experts=3)


def test_nvfp4_forward_matches_oracle_fwd_bwd():
    torch.manual_seed(1)
    T, K, H, I, E, r = 5, 2, 6, 4, 3, 2
    dtype = torch.float64
    s1, s2 = 0.5, 0.25

    hs = torch.randn(T, H, dtype=dtype, requires_grad=True)
    idx = torch.randint(0, E, (T, K))
    tkw = torch.rand(T, K, dtype=dtype, requires_grad=True)
    w1 = torch.randn(E, 2 * I, H, dtype=dtype)  # frozen base
    w2 = torch.randn(E, H, I, dtype=dtype)

    A1_3d, B1_3d = _make_peft_lora(E, r, 2 * I, H, dtype=dtype, seed=2)
    A2_3d, B2_3d = _make_peft_lora(E, r, H, I, dtype=dtype, seed=3)

    # clone leaves for the two independent graphs (module path vs oracle)
    hs_o = hs.detach().clone().requires_grad_(True)
    tkw_o = tkw.detach().clone().requires_grad_(True)
    A1o, B1o = (
        A1_3d.detach().clone().requires_grad_(True),
        B1_3d.detach().clone().requires_grad_(True),
    )
    A2o, B2o = (
        A2_3d.detach().clone().requires_grad_(True),
        B2_3d.detach().clone().requires_grad_(True),
    )

    lora_A1, lora_B1 = _peft_layout(A1_3d, B1_3d)
    lora_A2, lora_B2 = _peft_layout(A2_3d, B2_3d)

    out = grouped_moe_reference_forward(
        hs,
        idx,
        tkw,
        w1,
        None,
        w2,
        None,
        (lora_A1, lora_B1),
        (lora_A2, lora_B2),
        E,
        act="swiglu",
        backend="torch",
        concat=True,
        scaling1=s1,
        scaling2=s2,
    )
    ref = _oracle_forward(hs_o, idx, tkw_o, w1, w2, A1o, B1o, s1, A2o, B2o, s2)

    assert torch.allclose(out, ref, atol=1e-9), (out - ref).abs().max()

    g = torch.randn_like(out)
    out.backward(g)
    ref.backward(g)

    assert torch.allclose(hs.grad, hs_o.grad, atol=1e-8)
    assert torch.allclose(tkw.grad, tkw_o.grad, atol=1e-8)
    assert torch.allclose(A1_3d.grad, A1o.grad, atol=1e-8)
    assert torch.allclose(B1_3d.grad, B1o.grad, atol=1e-8)
    assert torch.allclose(A2_3d.grad, A2o.grad, atol=1e-8)
    assert torch.allclose(B2_3d.grad, B2o.grad, atol=1e-8)


def test_nvfp4_forward_no_lora_and_frozen_base():
    torch.manual_seed(4)
    T, K, H, I, E = 4, 2, 6, 4, 3
    dtype = torch.float64
    hs = torch.randn(T, H, dtype=dtype)
    idx = torch.randint(0, E, (T, K))
    tkw = torch.rand(T, K, dtype=dtype)
    w1 = torch.randn(E, 2 * I, H, dtype=dtype, requires_grad=True)
    w2 = torch.randn(E, H, I, dtype=dtype, requires_grad=True)

    out = grouped_moe_reference_forward(
        hs,
        idx,
        tkw,
        w1,
        None,
        w2,
        None,
        None,
        None,
        E,
        act="swiglu",
        backend="torch",
        concat=True,
        scaling1=1.0,
        scaling2=1.0,
    )
    out.sum().backward()
    # No LoRA Function is invoked, so the base GEMM is a plain differentiable op
    # and w1 does get a grad here.
    assert w1.grad is not None
    # sanity: matches oracle with zero LoRA
    A0_1 = torch.zeros(E, 1, H, dtype=dtype)
    B0_1 = torch.zeros(E, 2 * I, 1, dtype=dtype)
    A0_2 = torch.zeros(E, 1, I, dtype=dtype)
    B0_2 = torch.zeros(E, H, 1, dtype=dtype)
    ref = _oracle_forward(
        hs, idx, tkw, w1.detach(), w2.detach(), A0_1, B0_1, 0.0, A0_2, B0_2, 0.0
    )
    assert torch.allclose(out, ref, atol=1e-9)


def test_nvfp4_forward_clamped_swiglu_limit():
    torch.manual_seed(8)
    T, K, H, I, E = 5, 2, 6, 4, 3
    dtype = torch.float64
    L = 10.0
    # Large weights so the pre-activation exceeds the clamp bound.
    hs = torch.randn(T, H, dtype=dtype) * 3.0
    idx = torch.randint(0, E, (T, K))
    tkw = torch.rand(T, K, dtype=dtype)
    w1 = torch.randn(E, 2 * I, H, dtype=dtype) * 3.0
    w2 = torch.randn(E, H, I, dtype=dtype)

    out = grouped_moe_reference_forward(
        hs,
        idx,
        tkw,
        w1,
        None,
        w2,
        None,
        None,
        None,
        E,
        act="swiglu",
        backend="torch",
        concat=True,
        scaling1=1.0,
        scaling2=1.0,
        limit=L,
    )

    def clamped_oracle():
        acc = hs.new_zeros((T, H))
        for t in range(T):
            for k in range(K):
                e = int(idx[t, k])
                h = hs[t] @ w1[e].transpose(0, 1)
                gate = h[:I].clamp(max=L)
                up = h[I:].clamp(min=-L, max=L)
                a = up * F.silu(gate)
                acc[t] = acc[t] + tkw[t, k] * (a @ w2[e].transpose(0, 1))
        return acc

    torch.testing.assert_close(out, clamped_oracle(), atol=1e-9, rtol=1e-9)
    # Without the limit the outputs differ (clamp actually fires).
    unclamped = grouped_moe_reference_forward(
        hs,
        idx,
        tkw,
        w1,
        None,
        w2,
        None,
        None,
        None,
        E,
        act="swiglu",
        backend="torch",
        concat=True,
        scaling1=1.0,
        scaling2=1.0,
    )
    assert not torch.allclose(out, unclamped)


def test_dequant_backend_identity_on_dense():
    torch.manual_seed(5)
    T, K, H, I, E = 4, 2, 6, 4, 3
    dtype = torch.float64
    hs = torch.randn(T, H, dtype=dtype)
    idx = torch.randint(0, E, (T, K))
    tkw = torch.rand(T, K, dtype=dtype)
    w1 = torch.randn(E, 2 * I, H, dtype=dtype)
    w2 = torch.randn(E, H, I, dtype=dtype)

    common = dict(act="swiglu", concat=True, scaling1=1.0, scaling2=1.0)
    y_torch = grouped_moe_reference_forward(
        hs, idx, tkw, w1, None, w2, None, None, None, E, backend="torch", **common
    )
    y_deq = grouped_moe_reference_forward(
        hs, idx, tkw, w1, None, w2, None, None, None, E, backend="dequant", **common
    )
    assert torch.allclose(y_torch, y_deq, atol=1e-12)
