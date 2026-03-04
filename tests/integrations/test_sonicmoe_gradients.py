"""
Gradient correctness tests for SonicMoE routing functions (CPU-only).

Uses torch.autograd.gradcheck with float32 inputs to match the production
code path where routing happens in float32.
"""

import torch

from axolotl.integrations.kernels.sonicmoe.routing import (
    sigmoid_topk_routing,
    softmax_topk_routing,
)

_GC_EPS = 1e-3
_GC_ATOL = 1e-3
_GC_RTOL = 1e-3


def _make_softmax_moe_block(weight):
    gate = torch.nn.Module()
    gate.weight = weight
    gate.top_k = 2
    gate.norm_topk_prob = True

    moe_block = torch.nn.Module()
    moe_block.gate = gate
    return moe_block


def _make_sigmoid_moe_block(weight, bias):
    gate = torch.nn.Module()
    gate.weight = weight
    gate.e_score_correction_bias = bias

    moe_block = torch.nn.Module()
    moe_block.gate = gate
    moe_block.top_k = 2
    moe_block.n_routed_experts = weight.shape[0]
    moe_block.n_group = 1
    moe_block.norm_topk_prob = True
    moe_block.routed_scaling_factor = 1.0
    return moe_block


class TestSoftmaxTopkRoutingGradcheck:
    """Numerical gradient verification for softmax_topk_routing."""

    def test_gradcheck_wrt_gate_weight(self):
        T, H, E = 4, 8, 4

        hidden = torch.randn(T, H, dtype=torch.float32)

        def fn(weight):
            moe_block = _make_softmax_moe_block(weight)
            scores, _, _, _ = softmax_topk_routing(hidden, moe_block)
            return scores

        weight = torch.randn(E, H, dtype=torch.float32, requires_grad=True)
        torch.autograd.gradcheck(
            fn, (weight,), eps=_GC_EPS, atol=_GC_ATOL, rtol=_GC_RTOL
        )

    def test_gradcheck_wrt_hidden_states(self):
        T, H, E = 4, 8, 4

        weight = torch.randn(E, H, dtype=torch.float32)
        moe_block = _make_softmax_moe_block(weight)

        def fn(hidden):
            scores, _, _, _ = softmax_topk_routing(hidden, moe_block)
            return scores

        hidden = torch.randn(T, H, dtype=torch.float32, requires_grad=True)
        torch.autograd.gradcheck(
            fn, (hidden,), eps=_GC_EPS, atol=_GC_ATOL, rtol=_GC_RTOL
        )

    def test_gradcheck_wrt_router_logits(self):
        T, H, E = 4, 8, 4

        hidden = torch.randn(T, H, dtype=torch.float32)

        def fn(weight):
            moe_block = _make_softmax_moe_block(weight)
            _, _, _, router_logits = softmax_topk_routing(hidden, moe_block)
            return router_logits

        weight = torch.randn(E, H, dtype=torch.float32, requires_grad=True)
        torch.autograd.gradcheck(
            fn, (weight,), eps=_GC_EPS, atol=_GC_ATOL, rtol=_GC_RTOL
        )

    def test_no_norm_variant(self):
        T, H, E = 4, 8, 4

        hidden = torch.randn(T, H, dtype=torch.float32)

        def fn(weight):
            moe_block = _make_softmax_moe_block(weight)
            moe_block.gate.norm_topk_prob = False
            scores, _, _, _ = softmax_topk_routing(hidden, moe_block)
            return scores

        weight = torch.randn(E, H, dtype=torch.float32, requires_grad=True)
        torch.autograd.gradcheck(
            fn, (weight,), eps=_GC_EPS, atol=_GC_ATOL, rtol=_GC_RTOL
        )


class TestSigmoidTopkRoutingGradcheck:
    """Numerical gradient verification for sigmoid_topk_routing."""

    def test_gradcheck_wrt_gate_weight(self):
        T, H, E = 4, 8, 4

        hidden = torch.randn(T, H, dtype=torch.float32)
        bias = torch.zeros(E, dtype=torch.float32)

        def fn(weight):
            moe_block = _make_sigmoid_moe_block(weight, bias)
            scores, _, _, _ = sigmoid_topk_routing(hidden, moe_block)
            return scores

        weight = torch.randn(E, H, dtype=torch.float32, requires_grad=True)
        torch.autograd.gradcheck(
            fn, (weight,), eps=_GC_EPS, atol=_GC_ATOL, rtol=_GC_RTOL
        )

    def test_gradcheck_wrt_hidden_states(self):
        T, H, E = 4, 8, 4

        weight = torch.randn(E, H, dtype=torch.float32)
        bias = torch.zeros(E, dtype=torch.float32)
        moe_block = _make_sigmoid_moe_block(weight, bias)

        def fn(hidden):
            scores, _, _, _ = sigmoid_topk_routing(hidden, moe_block)
            return scores

        hidden = torch.randn(T, H, dtype=torch.float32, requires_grad=True)
        torch.autograd.gradcheck(
            fn, (hidden,), eps=_GC_EPS, atol=_GC_ATOL, rtol=_GC_RTOL
        )

    def test_gradcheck_wrt_bias(self):
        T, H, E = 4, 8, 4

        hidden = torch.randn(T, H, dtype=torch.float32)
        weight = torch.randn(E, H, dtype=torch.float32)

        def fn(bias):
            moe_block = _make_sigmoid_moe_block(weight, bias)
            scores, _, _, _ = sigmoid_topk_routing(hidden, moe_block)
            return scores

        bias = torch.zeros(E, dtype=torch.float32, requires_grad=True)
        torch.autograd.gradcheck(fn, (bias,), eps=_GC_EPS, atol=_GC_ATOL, rtol=_GC_RTOL)
