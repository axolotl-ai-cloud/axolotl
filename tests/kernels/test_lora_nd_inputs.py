"""The fused LoRA autograd kernels accept any number of leading input dims, e.g. the 4-D
``[B, 1, S, kv_lora_rank]`` latent Mistral Large 4 feeds ``kv_b_proj``, and match
``x @ W.T + b + s * (x @ A.T) @ B.T`` forward and backward."""

import pytest
import torch

from axolotl.kernels.lora import (
    LoRA_FusedProj,
    LoRA_O,
    LoRA_QK,
    LoRA_QKV,
    matmul_lora,
)

IN, R = 32, 4
SHAPES = [(6,), (2, 3), (2, 1, 3)]
SHAPE_IDS = ["2d", "3d", "4d"]


def _projection(out_features, seed):
    g = torch.Generator().manual_seed(seed)
    return {
        "W": torch.randn(out_features, IN, generator=g) * 0.1,
        "b": torch.randn(out_features, generator=g) * 0.1,
        "A": (torch.randn(R, IN, generator=g) * 0.1).requires_grad_(),
        "B": (torch.randn(out_features, R, generator=g) * 0.1).requires_grad_(),
        "s": 0.5,
    }


def _reference(x, p):
    return x @ p["W"].t() + p["b"] + p["s"] * (x @ p["A"].t()) @ p["B"].t()


def _args(p):
    """weight, bias, quant, A, B, scale, lora_bias, magnitude"""
    return p["W"], p["b"], None, p["A"], p["B"], p["s"], None, None


def _check(lead, fused, projections):
    """Run ``fused(X, X_drop)`` and the reference on ``X [*lead, IN]``, with and without
    dropout, and compare outputs and the input, A and B gradients."""
    for dropout in (False, True):
        torch.manual_seed(0)
        x = torch.randn(*lead, IN, requires_grad=True)
        x_drop = (x * 0.5) if dropout else None
        outs = fused(x, x_drop)
        grads_out = [torch.randn_like(o) for o in outs]
        torch.autograd.backward(outs, grads_out)
        got = [x.grad] + [g for p in projections for g in (p["A"].grad, p["B"].grad)]
        for p in projections:
            p["A"].grad = p["B"].grad = None

        x_ref = x.detach().clone().requires_grad_()
        x_lora = (x_ref * 0.5) if dropout else x_ref
        refs = [
            x_ref @ p["W"].t() + p["b"] + p["s"] * (x_lora @ p["A"].t()) @ p["B"].t()
            for p in projections
        ]
        torch.autograd.backward(refs, grads_out)
        want = [x_ref.grad] + [
            g for p in projections for g in (p["A"].grad, p["B"].grad)
        ]
        for p in projections:
            p["A"].grad = p["B"].grad = None

        for o, r in zip(outs, refs, strict=True):
            assert o.shape == (*lead, r.shape[-1])
            torch.testing.assert_close(o, r)
        for g, w in zip(got, want, strict=True):
            torch.testing.assert_close(g, w)


@pytest.mark.parametrize("lead", SHAPES, ids=SHAPE_IDS)
def test_matmul_lora(lead):
    p = _projection(24, 0)
    x = torch.randn(*lead, IN)
    x_drop = x * 0.5
    with torch.no_grad():
        out = matmul_lora(x, p["W"], p["b"], None, p["A"], p["B"], p["s"])
        out_drop = matmul_lora(
            x, p["W"], p["b"], None, p["A"], p["B"], p["s"], X_drop=x_drop
        )
        ref = _reference(x, p)
        ref_drop = x @ p["W"].t() + p["b"] + p["s"] * (x_drop @ p["A"].t()) @ p["B"].t()
    assert out.shape == (*lead, 24)
    torch.testing.assert_close(out, ref)
    torch.testing.assert_close(out_drop, ref_drop)


@pytest.mark.parametrize("lead", SHAPES, ids=SHAPE_IDS)
def test_lora_o(lead):
    p = _projection(24, 1)
    _check(
        lead,
        lambda x, x_drop: (LoRA_O.apply(x, x_drop, *_args(p)),),
        [p],
    )


@pytest.mark.parametrize("lead", SHAPES, ids=SHAPE_IDS)
def test_lora_qkv(lead):
    q, k, v = _projection(24, 2), _projection(8, 3), _projection(8, 4)
    _check(
        lead,
        lambda x, x_drop: LoRA_QKV.apply(
            x, x_drop, *_args(q), *_args(k), *_args(v), False
        ),
        [q, k, v],
    )


@pytest.mark.parametrize("lead", SHAPES, ids=SHAPE_IDS)
def test_lora_qk(lead):
    q, k = _projection(24, 5), _projection(8, 6)
    _check(
        lead,
        lambda x, x_drop: LoRA_QK.apply(x, x_drop, *_args(q), *_args(k), False),
        [q, k],
    )


@pytest.mark.parametrize("lead", SHAPES, ids=SHAPE_IDS)
def test_lora_fused_proj(lead):
    projections = [_projection(24, 7), _projection(8, 8)]
    tensors = [
        t for p in projections for t in (p["W"], p["b"], p["A"], p["B"], None, None)
    ]
    _check(
        lead,
        lambda x, x_drop: LoRA_FusedProj.apply(
            tuple(p["s"] for p in projections),
            (None, None),
            x,
            x_drop,
            *tensors,
        ),
        projections,
    )
