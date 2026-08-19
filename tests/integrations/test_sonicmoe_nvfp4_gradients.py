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


# =============================================================================
# Merge-aware forward (STE fake-quant of the effective weight)
# =============================================================================


def _make_nvfp4_base(E, n, k):
    """Packed per-expert NVFP4 base [E, n, k] with per-expert pts, orig f32."""
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


def test_merge_aware_forward_matches_snapped_oracle():
    """Merge-aware grouped forward == per-expert oracle on SNAPPED weights
    ``Q(W + s*(B@A))`` through the shared fake-quant."""
    pytest.importorskip("torchao")
    from axolotl.integrations.kernels.libs.sonicmoe.nvfp4_quant import (
        fake_quant_nvfp4,
    )

    torch.manual_seed(5)
    E, H, I, r, T = 3, 32, 16, 2, 11  # noqa: E741
    two_i = 2 * I
    s1, s2 = 0.5, 0.75

    offsets = _rand_offsets(E, T)
    w1 = _make_nvfp4_base(E, two_i, H)
    w2 = _make_nvfp4_base(E, H, I)
    A1, B1 = _make_lora(E, two_i, H, r, torch.float32)
    A2, B2 = _make_lora(E, H, I, r, torch.float32)
    x = torch.randn(T, H)

    w1d, w2d = w1.dequantize(), w2.dequantize()
    y = grouped_expert_mlp_lora(
        x,
        offsets,
        w1d,
        None,
        w2d,
        None,
        (A1, B1),
        (A2, B2),
        act="silu",
        backend="torch",
        concat=True,
        scaling1=s1,
        scaling2=s2,
        merge_aware1=True,
        ma_pts1=w1.per_tensor_scale,
        merge_aware2=True,
        ma_pts2=w2.per_tensor_scale,
    )

    pts1 = w1.per_tensor_scale.reshape(-1)
    pts2 = w2.per_tensor_scale.reshape(-1)
    y_ref = x.new_zeros((T, H))
    for e in range(E):
        st, en = int(offsets[e]), int(offsets[e + 1])
        if en <= st:
            continue
        A1e = A1.reshape(E, r, H)[e]
        B1e = B1.reshape(two_i, r, E).permute(2, 0, 1)[e]
        W1fq = fake_quant_nvfp4(w1d[e] + B1e @ (A1e * s1), pts1[e])
        h_e = F.linear(x[st:en], W1fq)
        a_e = _ref_gated_activation(h_e, "silu", concat=True)
        A2e = A2.reshape(E, r, I)[e]
        B2e = B2.reshape(H, r, E).permute(2, 0, 1)[e]
        W2fq = fake_quant_nvfp4(w2d[e] + B2e @ (A2e * s2), pts2[e])
        y_ref[st:en] = F.linear(a_e, W2fq)

    assert torch.equal(y, y_ref)


def test_merge_aware_ste_gradients():
    """Grads through the merge-aware forward == autograd through the explicit
    STE construction ``W_eff + (Q(W_eff) - W_eff).detach()``; the frozen base
    gets none."""
    pytest.importorskip("torchao")
    from axolotl.integrations.kernels.libs.sonicmoe.nvfp4_quant import (
        fake_quant_nvfp4,
    )

    torch.manual_seed(6)
    E, H, I, r, T = 2, 32, 16, 2, 9  # noqa: E741
    two_i = 2 * I
    s1, s2 = 0.5, 0.75

    offsets = _rand_offsets(E, T)
    w1 = _make_nvfp4_base(E, two_i, H)
    w2 = _make_nvfp4_base(E, H, I)
    w1d, w2d = w1.dequantize(), w2.dequantize()
    pts1 = w1.per_tensor_scale.reshape(-1)
    pts2 = w2.per_tensor_scale.reshape(-1)
    A1, B1 = _make_lora(E, two_i, H, r, torch.float32)
    A2, B2 = _make_lora(E, H, I, r, torch.float32)
    x = torch.randn(T, H, requires_grad=True)
    grad_out = torch.randn(T, H)

    y = grouped_expert_mlp_lora(
        x,
        offsets,
        w1d,
        None,
        w2d,
        None,
        (A1, B1),
        (A2, B2),
        act="silu",
        backend="torch",
        concat=True,
        scaling1=s1,
        scaling2=s2,
        merge_aware1=True,
        ma_pts1=w1.per_tensor_scale,
        merge_aware2=True,
        ma_pts2=w2.per_tensor_scale,
    )
    y.backward(grad_out)

    def ste(w_e, pts_e, A_, B_, e, dims, scaling):
        dim1, dim2 = dims
        A_e = A_.reshape(E, r, dim2)[e]
        B_e = B_.reshape(dim1, r, E).permute(2, 0, 1)[e]
        w_eff = w_e + scaling * (B_e @ A_e)
        return w_eff + (fake_quant_nvfp4(w_eff.detach(), pts_e) - w_eff.detach())

    xo = x.detach().clone().requires_grad_(True)
    A1o, B1o = (
        A1.detach().clone().requires_grad_(True),
        B1.detach().clone().requires_grad_(True),
    )
    A2o, B2o = (
        A2.detach().clone().requires_grad_(True),
        B2.detach().clone().requires_grad_(True),
    )
    y_ref = xo.new_zeros((T, H))
    for e in range(E):
        st, en = int(offsets[e]), int(offsets[e + 1])
        if en <= st:
            continue
        W1 = ste(w1d[e], pts1[e], A1o, B1o, e, (two_i, H), s1)
        a_e = _ref_gated_activation(F.linear(xo[st:en], W1), "silu", concat=True)
        W2 = ste(w2d[e], pts2[e], A2o, B2o, e, (H, I), s2)
        y_ref[st:en] = F.linear(a_e, W2)
    assert torch.allclose(y, y_ref.detach(), atol=1e-6)
    y_ref.backward(grad_out)

    for got, want in (
        (x.grad, xo.grad),
        (A1.grad, A1o.grad),
        (B1.grad, B1o.grad),
        (A2.grad, A2o.grad),
        (B2.grad, B2o.grad),
    ):
        assert torch.allclose(got, want, atol=1e-4, rtol=1e-4)


def test_merge_aware_global_toggle_reference_forward():
    """set_merge_aware_enabled routes packed-NVFP4 LoRA layers through the
    snapped forward (output changes); a dense base is untouched (bitwise
    no-op) since there is no grid to snap to."""
    pytest.importorskip("torchao")
    from axolotl.integrations.kernels.libs.sonicmoe.nvfp4_lora import (
        grouped_moe_reference_forward,
        set_merge_aware_enabled,
    )

    torch.manual_seed(7)
    E, H, I, r, T, K = 3, 32, 16, 2, 8, 2  # noqa: E741
    two_i = 2 * I
    w1 = _make_nvfp4_base(E, two_i, H)
    w2 = _make_nvfp4_base(E, H, I)
    A1, B1 = _make_lora(E, two_i, H, r, torch.float32)
    A2, B2 = _make_lora(E, H, I, r, torch.float32)
    hidden = torch.randn(T, H)
    top_k_index = torch.randint(0, E, (T, K))
    top_k_weights = torch.softmax(torch.randn(T, K), dim=-1)

    def run(w1_, w2_, backend):
        return grouped_moe_reference_forward(
            hidden,
            top_k_index,
            top_k_weights,
            w1_,
            None,
            w2_,
            None,
            (A1, B1),
            (A2, B2),
            E,
            act="silu",
            backend=backend,
            concat=True,
            scaling1=0.5,
            scaling2=0.75,
        )

    try:
        y_off = run(w1, w2, "dequant")
        set_merge_aware_enabled(True)
        y_on = run(w1, w2, "dequant")
        assert not torch.equal(y_on, y_off)

        w1d, w2d = w1.dequantize(), w2.dequantize()
        set_merge_aware_enabled(False)
        y_dense_off = run(w1d, w2d, "torch")
        set_merge_aware_enabled(True)
        y_dense_on = run(w1d, w2d, "torch")
        assert torch.equal(y_dense_on, y_dense_off)
    finally:
        set_merge_aware_enabled(False)


def test_merge_aware_train_then_merge_forward_identity():
    """Capstone: 'train' a few SGD steps with the merge-aware flag on, quantize
    the final effective weights with the shared quantizer (simulating the file
    write + reload), and assert the merged BASE-ONLY forward reproduces the last
    training forward BITWISE on the expert layers. This is the retention
    guarantee by construction."""
    pytest.importorskip("torchao")
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    from axolotl.integrations.kernels.libs.sonicmoe.nvfp4_lora import (
        _merge_aware_wfq,
    )
    from axolotl.integrations.kernels.libs.sonicmoe.nvfp4_quant import (
        quantize_nvfp4_merge,
    )

    torch.manual_seed(8)
    E, H, I, r, T = 2, 32, 16, 2, 13  # noqa: E741
    two_i = 2 * I
    s1, s2 = 0.5, 0.75

    offsets = _rand_offsets(E, T)
    w1 = _make_nvfp4_base(E, two_i, H)
    w2 = _make_nvfp4_base(E, H, I)
    w1d, w2d = w1.dequantize(), w2.dequantize()
    pts1 = w1.per_tensor_scale
    pts2 = w2.per_tensor_scale
    A1, B1 = _make_lora(E, two_i, H, r, torch.float32)
    A2, B2 = _make_lora(E, H, I, r, torch.float32)
    x = torch.randn(T, H)

    def fwd(lora1, lora2, w1_, w2_, ma):
        return grouped_expert_mlp_lora(
            x,
            offsets,
            w1_,
            None,
            w2_,
            None,
            lora1,
            lora2,
            act="silu",
            backend="torch",
            concat=True,
            scaling1=s1,
            scaling2=s2,
            merge_aware1=ma,
            ma_pts1=pts1 if ma else None,
            merge_aware2=ma,
            ma_pts2=pts2 if ma else None,
        )

    opt = torch.optim.SGD([A1, B1, A2, B2], lr=1e-2)
    for _ in range(3):
        y = fwd((A1, B1), (A2, B2), w1d, w2d, ma=True)
        y.square().mean().backward()
        opt.step()
        opt.zero_grad()

    with torch.no_grad():
        y_train = fwd((A1, B1), (A2, B2), w1d, w2d, ma=True)

        # merge: quantize W_eff with the shared quantizer, roundtrip through the
        # stored representation (packed codes + e4m3 scales + pts), reload
        def merge(w_dense, lora_A, lora_B, scaling, pts):
            dim1, dim2 = w_dense.shape[1:]
            A_3d = lora_A.reshape(E, r, dim2)
            B_3d = lora_B.reshape(dim1, r, E).permute(2, 0, 1)
            w_eff = w_dense + torch.bmm(B_3d, A_3d * scaling)
            packed, scale = quantize_nvfp4_merge(
                w_eff, pts.reshape(-1), scale_mode="fresh"
            )
            return NVFP4Tensor(
                packed, scale, 16, torch.bfloat16, per_tensor_scale=pts
            ).dequantize(w_dense.dtype)

        w1_merged = merge(w1d, A1, B1, s1, pts1)
        w2_merged = merge(w2d, A2, B2, s2, pts2)

        # the reloaded merged weight IS the snapped training operand
        assert torch.equal(w1_merged, _merge_aware_wfq(w1d, A1, B1, s1, pts1))
        assert torch.equal(w2_merged, _merge_aware_wfq(w2d, A2, B2, s2, pts2))

        y_merged = fwd(None, None, w1_merged, w2_merged, ma=False)

    assert torch.equal(y_train, y_merged)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA + triton")
def test_fake_quant_triton_bitwise_vs_reference():
    """The fused triton fake-quant must be BITWISE the torchao reference on
    every path (two-level per-expert pts, scalar pts, single-level), including
    boundary values engineered to sit on rounding ties. Pod-run (GPU only)."""
    pytest.importorskip("torchao")
    from axolotl.integrations.kernels.libs.sonicmoe.nvfp4_quant import (
        fake_quant_nvfp4,
    )
    from axolotl.integrations.kernels.libs.sonicmoe.triton_nvfp4 import (
        fake_quant_nvfp4_triton,
        triton_available,
    )

    if not triton_available():
        pytest.skip("triton unavailable")

    torch.manual_seed(9)
    dev = "cuda"
    for dtype in (torch.bfloat16, torch.float32):
        E, N, K = 4, 128, 256
        w = (torch.randn(E, N, K, device=dev) * 0.02).to(dtype)
        # salt in exact tie values (post-scaling ~0.25/0.75/... multiples land often)
        w.view(-1)[:4096] = (
            torch.tensor(
                [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0], device=dev
            ).repeat(512)
            * 1e-3
        ).to(dtype)
        pts_e = (torch.rand(E, device=dev) * 1e-4 + 1e-5).float()

        ref = fake_quant_nvfp4(w, pts_e)
        got = fake_quant_nvfp4_triton(w.clone(), pts_e)
        assert torch.equal(got, ref), f"per-expert pts mismatch ({dtype})"

        got_ip = fake_quant_nvfp4_triton(w.clone(), pts_e, inplace=True)
        assert torch.equal(got_ip, ref), f"inplace mismatch ({dtype})"

        w2 = w[0].contiguous()
        ref2 = fake_quant_nvfp4(w2, pts_e[0])
        got2 = fake_quant_nvfp4_triton(w2.clone(), pts_e[0])
        assert torch.equal(got2, ref2), f"scalar pts mismatch ({dtype})"

        ref3 = fake_quant_nvfp4(w2)
        got3 = fake_quant_nvfp4_triton(w2.clone())
        assert torch.equal(got3, ref3), f"single-level mismatch ({dtype})"
