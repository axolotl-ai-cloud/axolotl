"""
Gradient correctness tests for SonicMoE routing functions (CPU-only).

Uses torch.autograd.gradcheck to verify that the actual routing functions
produce numerically correct gradients (finite-difference comparison).

Usage:
    pytest tests/integrations/test_sonicmoe_gradients.py -v
"""

import torch

from axolotl.integrations.kernels.sonicmoe.routing import (
    sigmoid_topk_routing,
    softmax_topk_routing,
)


def _make_softmax_moe_block(weight):
    """Create a mock moe_block wired to a real weight parameter."""
    gate = torch.nn.Module()
    gate.weight = weight
    gate.top_k = 2
    gate.norm_topk_prob = True

    moe_block = torch.nn.Module()
    moe_block.gate = gate
    return moe_block


def _make_sigmoid_moe_block(weight, bias):
    """Create a mock sigmoid-routing moe_block wired to real parameters."""
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
    """Numerical gradient verification for the real softmax_topk_routing."""

    def test_gradcheck_wrt_gate_weight(self):
        """gradcheck: d(scores)/d(gate.weight) is numerically correct."""
        T, H, E = 4, 8, 4

        hidden = torch.randn(T, H, dtype=torch.float64)

        def fn(weight):
            weight_param = torch.nn.Parameter(weight)
            moe_block = _make_softmax_moe_block(weight_param)
            scores, _, _, _ = softmax_topk_routing(hidden, moe_block)
            return scores

        weight = torch.randn(E, H, dtype=torch.float64, requires_grad=True)
        torch.autograd.gradcheck(fn, (weight,), eps=1e-6, atol=1e-4)

    def test_gradcheck_wrt_hidden_states(self):
        """gradcheck: d(scores)/d(hidden_states) is numerically correct."""
        T, H, E = 4, 8, 4

        weight = torch.nn.Parameter(torch.randn(E, H, dtype=torch.float64))
        moe_block = _make_softmax_moe_block(weight)

        def fn(hidden):
            scores, _, _, _ = softmax_topk_routing(hidden, moe_block)
            return scores

        hidden = torch.randn(T, H, dtype=torch.float64, requires_grad=True)
        torch.autograd.gradcheck(fn, (hidden,), eps=1e-6, atol=1e-4)

    def test_gradcheck_wrt_router_logits(self):
        """gradcheck: d(router_logits)/d(gate.weight) is numerically correct."""
        T, H, E = 4, 8, 4

        hidden = torch.randn(T, H, dtype=torch.float64)

        def fn(weight):
            weight_param = torch.nn.Parameter(weight)
            moe_block = _make_softmax_moe_block(weight_param)
            _, _, _, router_logits = softmax_topk_routing(hidden, moe_block)
            return router_logits

        weight = torch.randn(E, H, dtype=torch.float64, requires_grad=True)
        torch.autograd.gradcheck(fn, (weight,), eps=1e-6, atol=1e-4)

    def test_no_norm_variant(self):
        """gradcheck: routing without renormalization (norm_topk_prob=False)."""
        T, H, E = 4, 8, 4

        hidden = torch.randn(T, H, dtype=torch.float64)

        def fn(weight):
            weight_param = torch.nn.Parameter(weight)
            moe_block = _make_softmax_moe_block(weight_param)
            moe_block.gate.norm_topk_prob = False
            scores, _, _, _ = softmax_topk_routing(hidden, moe_block)
            return scores

        weight = torch.randn(E, H, dtype=torch.float64, requires_grad=True)
        torch.autograd.gradcheck(fn, (weight,), eps=1e-6, atol=1e-4)


class TestSigmoidTopkRoutingGradcheck:
    """Numerical gradient verification for the real sigmoid_topk_routing."""

    def test_gradcheck_wrt_gate_weight(self):
        """gradcheck: d(scores)/d(gate.weight) is numerically correct."""
        T, H, E = 4, 8, 4

        hidden = torch.randn(T, H, dtype=torch.float64)
        bias = torch.nn.Parameter(torch.zeros(E, dtype=torch.float64))

        def fn(weight):
            weight_param = torch.nn.Parameter(weight)
            moe_block = _make_sigmoid_moe_block(weight_param, bias)
            scores, _, _, _ = sigmoid_topk_routing(hidden, moe_block)
            return scores

        weight = torch.randn(E, H, dtype=torch.float64, requires_grad=True)
        torch.autograd.gradcheck(fn, (weight,), eps=1e-6, atol=1e-4)

    def test_gradcheck_wrt_hidden_states(self):
        """gradcheck: d(scores)/d(hidden_states) is numerically correct."""
        T, H, E = 4, 8, 4

        weight = torch.nn.Parameter(torch.randn(E, H, dtype=torch.float64))
        bias = torch.nn.Parameter(torch.zeros(E, dtype=torch.float64))
        moe_block = _make_sigmoid_moe_block(weight, bias)

        def fn(hidden):
            scores, _, _, _ = sigmoid_topk_routing(hidden, moe_block)
            return scores

        hidden = torch.randn(T, H, dtype=torch.float64, requires_grad=True)
        torch.autograd.gradcheck(fn, (hidden,), eps=1e-6, atol=1e-4)

    def test_gradcheck_wrt_bias(self):
        """gradcheck: d(scores)/d(e_score_correction_bias) is numerically correct."""
        T, H, E = 4, 8, 4

        hidden = torch.randn(T, H, dtype=torch.float64)
        weight = torch.nn.Parameter(torch.randn(E, H, dtype=torch.float64))

        def fn(bias):
            bias_param = torch.nn.Parameter(bias)
            moe_block = _make_sigmoid_moe_block(weight, bias_param)
            scores, _, _, _ = sigmoid_topk_routing(hidden, moe_block)
            return scores

        bias = torch.zeros(E, dtype=torch.float64, requires_grad=True)
        torch.autograd.gradcheck(fn, (bias,), eps=1e-6, atol=1e-4)
