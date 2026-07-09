# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Gradient correctness tests for the sonicmoe NVFP4 frozen-base grouped LoRA.

Runs entirely on CPU.
"""

import pytest
import torch
import torch.nn.functional as F

from axolotl.integrations.kernels.libs.sonicmoe.nvfp4_lora import (
    GroupedDownProjLoRA,
    GroupedUpProjLoRA,
    grouped_expert_mlp_lora,
)


def _ref_gated_activation(h, act, *, concat):
    """SwiGLU gated activation over the last (``2I``) dim -> ``I``.

    ``concat=True``: split-halves (``[gate | up]``). ``concat=False``:
    interleaved (``gate`` = even lanes, ``up`` = odd lanes).
    """
    assert act.lower() == "silu"
    if concat:
        gate, up = h.chunk(2, dim=-1)
    else:
        gate, up = h[..., 0::2], h[..., 1::2]
    return F.silu(gate) * up


# =============================================================================
# Helpers
# =============================================================================


def _rand_offsets(E, T, device="cpu"):
    """Random ascending expert offsets covering ``T`` tokens across ``E`` experts."""
    counts = torch.zeros(E, dtype=torch.long)
    for _ in range(T):
        counts[torch.randint(0, E, (1,)).item()] += 1
    offsets = torch.zeros(E + 1, dtype=torch.long, device=device)
    offsets[1:] = torch.cumsum(counts, 0)
    return offsets


def _make_lora(E, dim1, dim2, r, dtype):
    """PEFT rank-major LoRA: A ``[r*E, dim2]``, B ``[dim1, r*E]``."""
    A = torch.randn(r * E, dim2, dtype=dtype, requires_grad=True)
    B = torch.randn(dim1, r * E, dtype=dtype, requires_grad=True)
    return A, B


def _materialize_expert(W_e, A, B, e, r, E, scaling):
    """W_eff_e = W_e + scaling * (B_e @ A_e) in PEFT rank-major layout."""
    dim1, dim2 = W_e.shape
    A_e = A.reshape(E, r, dim2)[e]  # [r, dim2]
    B_e = B.reshape(dim1, r, E).permute(2, 0, 1)[e]  # [dim1, r]
    return W_e + scaling * (B_e @ A_e)


def _oracle_moe(
    x_grouped, expert_offsets, w1, b1, w2, b2, A1, B1, A2, B2, r, E, s1, s2, concat
):
    """Naive per-expert MoE on MATERIALIZED W_eff (up -> swiglu -> down)."""
    y = x_grouped.new_zeros((x_grouped.shape[0], w2.shape[1]))
    for e in range(E):
        st, en = int(expert_offsets[e]), int(expert_offsets[e + 1])
        if en <= st:
            continue
        x_e = x_grouped[st:en]
        W1e = _materialize_expert(w1[e], A1, B1, e, r, E, s1)
        h_e = F.linear(x_e, W1e)  # x_e @ W1e^T
        if b1 is not None:
            h_e = h_e + b1[e]
        a_e = _ref_gated_activation(h_e, "silu", concat=concat)
        W2e = _materialize_expert(w2[e], A2, B2, e, r, E, s2)
        y_e = F.linear(a_e, W2e)  # a_e @ W2e^T
        if b2 is not None:
            y_e = y_e + b2[e]
        y[st:en] = y_e
    return y


# =============================================================================
# Oracle test: forward + hand-written backward vs materialized autograd oracle
# =============================================================================


@pytest.mark.parametrize("concat", [True, False])
def test_oracle_forward_and_backward(concat):
    torch.manual_seed(0)
    dtype = torch.float64
    E, H, I, r, T = 3, 8, 6, 2, 17  # noqa: E741
    two_i = 2 * I
    s1, s2 = 0.5, 0.75

    offsets = _rand_offsets(E, T)

    w1 = torch.randn(E, two_i, H, dtype=dtype)  # frozen base
    w2 = torch.randn(E, H, I, dtype=dtype)  # frozen base
    b1 = torch.randn(E, two_i, dtype=dtype)
    b2 = torch.randn(E, H, dtype=dtype)

    A1, B1 = _make_lora(E, two_i, H, r, dtype)
    A2, B2 = _make_lora(E, H, I, r, dtype)

    x = torch.randn(T, H, dtype=dtype, requires_grad=True)

    # ---- grouped path ----
    y = grouped_expert_mlp_lora(
        x,
        offsets,
        w1,
        b1,
        w2,
        b2,
        (A1, B1),
        (A2, B2),
        act="silu",
        backend="torch",
        concat=concat,
        scaling1=s1,
        scaling2=s2,
    )

    # ---- oracle path ----
    xo = x.detach().clone().requires_grad_(True)
    A1o = A1.detach().clone().requires_grad_(True)
    B1o = B1.detach().clone().requires_grad_(True)
    A2o = A2.detach().clone().requires_grad_(True)
    B2o = B2.detach().clone().requires_grad_(True)
    yo = _oracle_moe(
        xo, offsets, w1, b1, w2, b2, A1o, B1o, A2o, B2o, r, E, s1, s2, concat
    )

    assert torch.allclose(y, yo, atol=1e-10), (y - yo).abs().max()

    # Backward with a random cotangent so every output element contributes.
    g = torch.randn_like(y)
    y.backward(g)
    yo.backward(g.clone())

    assert torch.allclose(x.grad, xo.grad, atol=1e-9), (x.grad - xo.grad).abs().max()
    assert torch.allclose(A1.grad, A1o.grad, atol=1e-9), (
        (A1.grad - A1o.grad).abs().max()
    )
    assert torch.allclose(B1.grad, B1o.grad, atol=1e-9), (
        (B1.grad - B1o.grad).abs().max()
    )
    assert torch.allclose(A2.grad, A2o.grad, atol=1e-9), (
        (A2.grad - A2o.grad).abs().max()
    )
    assert torch.allclose(B2.grad, B2o.grad, atol=1e-9), (
        (B2.grad - B2o.grad).abs().max()
    )


# =============================================================================
# gradcheck on the full chain
# =============================================================================


@pytest.mark.parametrize("concat", [True, False])
def test_gradcheck_full_chain(concat):
    torch.manual_seed(1)
    dtype = torch.float64
    E, H, I, r, T = 2, 6, 4, 2, 9  # noqa: E741
    two_i = 2 * I
    s1, s2 = 0.5, 1.0

    offsets = _rand_offsets(E, T)
    w1 = torch.randn(E, two_i, H, dtype=dtype)
    w2 = torch.randn(E, H, I, dtype=dtype)

    A1, B1 = _make_lora(E, two_i, H, r, dtype)
    A2, B2 = _make_lora(E, H, I, r, dtype)
    x = torch.randn(T, H, dtype=dtype, requires_grad=True)

    def fn(x_, A1_, B1_, A2_, B2_):
        return grouped_expert_mlp_lora(
            x_,
            offsets,
            w1,
            None,
            w2,
            None,
            (A1_, B1_),
            (A2_, B2_),
            act="silu",
            backend="torch",
            concat=concat,
            scaling1=s1,
            scaling2=s2,
        )

    assert torch.autograd.gradcheck(
        fn, (x, A1, B1, A2, B2), eps=1e-6, atol=1e-5, rtol=1e-3
    )


# =============================================================================
# None LoRA reduces to plain base MoE; base weights receive no gradient
# =============================================================================


@pytest.mark.parametrize("concat", [True, False])
def test_none_lora_reduces_to_base(concat):
    torch.manual_seed(2)
    dtype = torch.float64
    E, H, I, T = 3, 8, 6, 13  # noqa: E741
    two_i = 2 * I

    offsets = _rand_offsets(E, T)
    w1 = torch.randn(E, two_i, H, dtype=dtype, requires_grad=True)
    w2 = torch.randn(E, H, I, dtype=dtype, requires_grad=True)
    x = torch.randn(T, H, dtype=dtype, requires_grad=True)

    y = grouped_expert_mlp_lora(
        x,
        offsets,
        w1,
        None,
        w2,
        None,
        None,
        None,
        act="silu",
        backend="torch",
        concat=concat,
        scaling1=1.0,
        scaling2=1.0,
    )

    # Naive base-only MoE.
    expected = x.new_zeros((T, H))
    for e in range(E):
        s, t = int(offsets[e]), int(offsets[e + 1])
        if t <= s:
            continue
        h_e = F.linear(x[s:t], w1[e])
        a_e = _ref_gated_activation(h_e, "silu", concat=concat)
        expected[s:t] = F.linear(a_e, w2[e])

    assert torch.allclose(y, expected, atol=1e-10)

    y.sum().backward()
    # No LoRA Function is invoked here, so the base grouped GEMM is a plain
    # differentiable op and w1/w2 do get grads. We only assert the backward runs.
    assert x.grad is not None


def test_lora_base_weight_no_grad():
    """The LoRA autograd.Functions must NOT produce a gradient for the frozen base."""
    torch.manual_seed(3)
    dtype = torch.float64
    E, H, I, r, T = 2, 6, 4, 2, 7  # noqa: E741
    two_i = 2 * I

    offsets = _rand_offsets(E, T)
    # Base requires_grad=True to prove the Function still returns None for it.
    w1 = torch.randn(E, two_i, H, dtype=dtype, requires_grad=True)
    A1, B1 = _make_lora(E, two_i, H, r, dtype)
    x = torch.randn(T, H, dtype=dtype, requires_grad=True)

    h = GroupedUpProjLoRA.apply(x, w1, offsets, A1, B1, 0.5, "torch", True)
    h.sum().backward()

    assert w1.grad is None
    assert A1.grad is not None
    assert B1.grad is not None
    assert x.grad is not None


def test_down_proj_base_weight_no_grad():
    torch.manual_seed(4)
    dtype = torch.float64
    E, H, I, r, T = 2, 6, 4, 2, 7  # noqa: E741

    offsets = _rand_offsets(E, T)
    w2 = torch.randn(E, H, I, dtype=dtype, requires_grad=True)
    A2, B2 = _make_lora(E, H, I, r, dtype)
    a = torch.randn(T, I, dtype=dtype, requires_grad=True)

    y = GroupedDownProjLoRA.apply(a, w2, offsets, A2, B2, 0.75, "torch")
    y.sum().backward()

    assert w2.grad is None
    assert A2.grad is not None
    assert B2.grad is not None
    assert a.grad is not None
