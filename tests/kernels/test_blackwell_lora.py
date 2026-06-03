# SPDX-License-Identifier: Apache-2.0
"""Correctness + gradient tests for the fused Blackwell (sm_120) LoRA kernels.

Skipped on non-sm_120 hardware. Compares the fused forward/backward against a
plain torch reference (bf16-level tolerance) and checks the dispatch guards fall
back when the config is unsupported.
"""

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12,
    reason="fused LoRA kernels require Blackwell GeForce/RTX (sm_120)",
)

DEV = "cuda"
S = 2.0
TOL = 6e-2


def _bf(o, i):
    return torch.randn(o, i, device=DEV, dtype=torch.bfloat16)


def _leaf(o, i, dtype=torch.float32):  # PEFT keeps adapters in fp32
    return (torch.randn(o, i, device=DEV, dtype=dtype) * 0.1).requires_grad_(True)


def _rel(a, b):
    return ((a.float() - b.float()).abs().max() / (b.float().abs().max() + 1e-6)).item()


def test_lora_dense_fwd_bwd():
    from axolotl.kernels.blackwell.autograd import lora_dense

    M, N, K, r = 512, 512, 256, 16
    X = torch.randn(M, K, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    W = _bf(N, K)
    A, B = _leaf(r, K), _leaf(N, r)
    Xr = X.detach().clone().requires_grad_(True)
    Ar, Br = (
        A.detach().clone().requires_grad_(True),
        B.detach().clone().requires_grad_(True),
    )

    Y = lora_dense(X, W, A, B, S)
    g = torch.randn_like(Y)
    Y.backward(g)
    Yr = Xr.float() @ W.float().t() + S * (Xr.float() @ Ar.float().t()) @ Br.float().t()
    Yr.backward(g)

    assert _rel(Y, Yr) < TOL
    assert _rel(X.grad, Xr.grad) < TOL
    assert _rel(A.grad, Ar.grad) < TOL and A.grad.dtype == torch.float32
    assert _rel(B.grad, Br.grad) < TOL


@pytest.mark.parametrize("act", ["silu", "gelu"])
def test_lora_mlp_glu_fwd_bwd(act):
    from axolotl.kernels.blackwell.mlp import lora_mlp_geglu, lora_mlp_swiglu

    M, H, inter, r = 512, 2048, 4096, 16
    X = torch.randn(M, H, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    gW, uW, dW = _bf(inter, H), _bf(inter, H), _bf(H, inter)
    gA, gB, uA, uB, dA, dB = (
        _leaf(r, H),
        _leaf(inter, r),
        _leaf(r, H),
        _leaf(inter, r),
        _leaf(r, inter),
        _leaf(H, r),
    )
    refs = {
        n: t.detach().clone().requires_grad_(True)
        for n, t in [
            ("gA", gA),
            ("gB", gB),
            ("uA", uA),
            ("uB", uB),
            ("dA", dA),
            ("dB", dB),
        ]
    }
    Xr = X.detach().clone().requires_grad_(True)

    fn = lora_mlp_swiglu if act == "silu" else lora_mlp_geglu
    out = fn(X, gW, gA, gB, uW, uA, uB, dW, dA, dB, S)
    g = torch.randn_like(out)
    out.backward(g)

    act_fn = F.silu if act == "silu" else F.gelu
    gate = (
        Xr.float() @ gW.float().t()
        + S * (Xr.float() @ refs["gA"].float().t()) @ refs["gB"].float().t()
    )
    up = (
        Xr.float() @ uW.float().t()
        + S * (Xr.float() @ refs["uA"].float().t()) @ refs["uB"].float().t()
    )
    h = act_fn(gate) * up
    ref = h @ dW.float().t() + S * (h @ refs["dA"].float().t()) @ refs["dB"].float().t()
    ref.backward(g)

    assert _rel(out, ref) < TOL
    assert _rel(X.grad, Xr.grad) < TOL
    for n in refs:
        assert _rel(locals()[n].grad, refs[n].grad) < TOL


def test_lora_qkv_gqa_fwd_bwd():
    from axolotl.kernels.blackwell.dispatch import maybe_lora_qkv

    M, H, Dq, Dk, r = 512, 2048, 2048, 512, 16  # GQA: K/V smaller than Q
    X = torch.randn(M, H, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    qW, kW, vW = _bf(Dq, H), _bf(Dk, H), _bf(Dk, H)
    qA, qB = _leaf(r, H), _leaf(Dq, r)
    kA, kB = _leaf(r, H), _leaf(Dk, r)
    vA, vB = _leaf(r, H), _leaf(Dk, r)

    out = maybe_lora_qkv(
        X,
        qW,
        qA,
        qB,
        S,
        None,
        None,
        None,
        None,
        kW,
        kA,
        kB,
        S,
        None,
        None,
        None,
        None,
        vW,
        vA,
        vB,
        S,
        None,
        None,
        None,
        None,
    )
    assert out is not None
    Q, K, V = out
    assert Q.shape == (M, Dq) and K.shape == (M, Dk) and V.shape == (M, Dk)

    def ref(W, A, B):
        return (
            X.float() @ W.float().t() + S * (X.float() @ A.float().t()) @ B.float().t()
        )

    assert _rel(Q, ref(qW, qA, qB)) < TOL
    assert _rel(K, ref(kW, kA, kB)) < TOL
    assert _rel(V, ref(vW, vA, vB)) < TOL
    (Q.sum() + K.sum() + V.sum()).backward()
    assert X.grad is not None and qA.grad.dtype == torch.float32 and vB.grad is not None


def test_lora_o_fwd_bwd():
    from axolotl.kernels.blackwell.dispatch import maybe_lora_o

    M, N, H, r = 512, 2048, 2048, 16
    X = torch.randn(M, H, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    W, A, B = _bf(N, H), _leaf(r, H), _leaf(N, r)
    out = maybe_lora_o(X, W, A, B, S, None, None, None, None)
    assert out is not None
    ref = X.float() @ W.float().t() + S * (X.float() @ A.float().t()) @ B.float().t()
    assert _rel(out, ref) < TOL
    out.sum().backward()
    assert X.grad is not None and A.grad.dtype == torch.float32


def test_dispatch_fallbacks():
    from axolotl.kernels.blackwell.dispatch import maybe_lora_o

    M, N, H, r = 512, 2048, 2048, 16
    Xb = torch.randn(M, H, device=DEV, dtype=torch.bfloat16)
    W, A, B = _bf(N, H), _leaf(r, H), _leaf(N, r)
    # fp32 input, quantized base, DoRA, and bias each force fallback (None)
    assert maybe_lora_o(Xb.float(), W, A, B, S, None, None, None, None) is None
    assert maybe_lora_o(Xb, W, A, B, S, object(), None, None, None) is None
    assert (
        maybe_lora_o(Xb, W, A, B, S, None, torch.ones(N, device=DEV), None, None)
        is None
    )
    assert (
        maybe_lora_o(Xb, W, A, B, S, None, None, torch.zeros(N, device=DEV), None)
        is None
    )
    # non-divisible output dim -> no valid tile -> fallback
    assert (
        maybe_lora_o(
            Xb, _bf(N + 8, H), _leaf(r, H), _leaf(N + 8, r), S, None, None, None, None
        )
        is None
    )
