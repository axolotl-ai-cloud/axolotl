# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""
Parity tests between scattermoe-lora and sonicmoe routing implementations.

These tests verify that both implementations produce numerically identical
results for the same inputs, ensuring safe centralization of the routing code.

ScatterMoE returns 2D tensors [T, K]; SonicMoE returns flattened 1D [T*K].
The core algorithm should be identical — only the output format differs.
"""

from types import SimpleNamespace

import pytest
import torch


def _require_triton():
    pytest.importorskip("triton")


# ============================================================================
# Fixtures / helpers
# ============================================================================


def _make_softmax_block(T=8, H=16, E=4, K=2):
    """Qwen/OLMoE-style block usable by both implementations."""
    gate = SimpleNamespace(
        weight=torch.randn(E, H),
        top_k=K,
        num_experts=E,
        norm_topk_prob=True,
    )
    moe_block = SimpleNamespace(gate=gate)
    hidden = torch.randn(T, H)
    return moe_block, gate, hidden, T, H, E, K


def _make_sigmoid_block(
    T=8, H=16, E=16, K=4, n_group=2, topk_group=1, bias_on_gate=True
):
    """GLM/DeepSeek-style block usable by both implementations."""
    if bias_on_gate:
        gate = SimpleNamespace(
            weight=torch.randn(E, H),
            e_score_correction_bias=torch.zeros(E),
        )
        moe_block = SimpleNamespace(
            gate=gate,
            top_k=K,
            n_routed_experts=E,
            n_group=n_group,
            topk_group=topk_group,
            norm_topk_prob=True,
            routed_scaling_factor=1.0,
        )
    else:
        # minimax_m2 style: bias on block
        gate = SimpleNamespace(
            weight=torch.randn(E, H),
            top_k=K,
        )
        moe_block = SimpleNamespace(
            gate=gate,
            top_k=K,
            e_score_correction_bias=torch.zeros(E),
        )
    return moe_block, gate, hidden_states(T, H), T, H, E, K


def hidden_states(T, H):
    return torch.randn(T, H)


# ============================================================================
# 1. Softmax routing parity
# ============================================================================


class TestSoftmaxRoutingParity:
    """Verify scattermoe and sonicmoe softmax routing produce identical results."""

    @pytest.fixture(autouse=True)
    def _require(self):
        _require_triton()

    def test_weights_match(self):
        """2D weights from scattermoe == reshaped 1D weights from sonicmoe."""
        from axolotl.integrations.kernels.libs.scattermoe_lora.layers import (
            _softmax_topk_route,
        )
        from axolotl.integrations.kernels.sonicmoe.routing import softmax_topk_routing

        moe_block, gate, hidden, T, H, E, K = _make_softmax_block()

        # ScatterMoE path (no LoRA delta)
        sm_weights, sm_experts, sm_topk, sm_E = _softmax_topk_route(
            moe_block, gate, hidden, gate.weight, None
        )

        # SonicMoE path
        sonic_scores, sonic_tok_idx, sonic_exp_idx, sonic_logits = softmax_topk_routing(
            hidden, moe_block
        )

        # ScatterMoE returns [T, K], SonicMoE returns [T*K] flattened
        sonic_weights_2d = sonic_scores.reshape(T, K)
        sonic_experts_2d = sonic_exp_idx.reshape(T, K)

        assert sm_topk == K
        assert sm_E == E

        # Both should select the same experts and produce the same weights
        assert torch.equal(sm_experts, sonic_experts_2d.to(sm_experts.dtype))
        assert torch.allclose(sm_weights, sonic_weights_2d, atol=1e-6)

    def test_logits_not_returned_by_scattermoe(self):
        """ScatterMoE doesn't return logits; SonicMoE does — verify SonicMoE logits shape."""
        from axolotl.integrations.kernels.sonicmoe.routing import softmax_topk_routing

        moe_block, gate, hidden, T, H, E, K = _make_softmax_block()
        _, _, _, logits = softmax_topk_routing(hidden, moe_block)
        assert logits.shape == (T, E)

    def test_no_renorm(self):
        """With norm_topk_prob=False, both should skip renormalization."""
        from axolotl.integrations.kernels.libs.scattermoe_lora.layers import (
            _softmax_topk_route,
        )
        from axolotl.integrations.kernels.sonicmoe.routing import softmax_topk_routing

        moe_block, gate, hidden, T, H, E, K = _make_softmax_block()
        gate.norm_topk_prob = False

        sm_weights, sm_experts, _, _ = _softmax_topk_route(
            moe_block, gate, hidden, gate.weight, None
        )
        sonic_scores, _, sonic_exp_idx, _ = softmax_topk_routing(hidden, moe_block)

        sonic_weights_2d = sonic_scores.reshape(T, K)
        sonic_experts_2d = sonic_exp_idx.reshape(T, K)

        assert torch.equal(sm_experts, sonic_experts_2d.to(sm_experts.dtype))
        assert torch.allclose(sm_weights, sonic_weights_2d, atol=1e-6)

    def test_various_expert_counts(self):
        """Parity across different E and K values."""
        from axolotl.integrations.kernels.libs.scattermoe_lora.layers import (
            _softmax_topk_route,
        )
        from axolotl.integrations.kernels.sonicmoe.routing import softmax_topk_routing

        for E, K in [(2, 1), (8, 2), (16, 4), (32, 8)]:
            moe_block, gate, hidden, T, H, _, _ = _make_softmax_block(E=E, K=K)

            sm_weights, sm_experts, _, _ = _softmax_topk_route(
                moe_block, gate, hidden, gate.weight, None
            )
            sonic_scores, _, sonic_exp_idx, _ = softmax_topk_routing(hidden, moe_block)

            sonic_weights_2d = sonic_scores.reshape(T, K)
            sonic_experts_2d = sonic_exp_idx.reshape(T, K)

            assert torch.equal(sm_experts, sonic_experts_2d.to(sm_experts.dtype)), (
                f"Expert mismatch for E={E}, K={K}"
            )
            assert torch.allclose(sm_weights, sonic_weights_2d, atol=1e-6), (
                f"Weight mismatch for E={E}, K={K}"
            )


# ============================================================================
# 2. Sigmoid routing parity
# ============================================================================


class TestSigmoidRoutingParity:
    """Verify scattermoe and sonicmoe sigmoid routing produce identical results."""

    @pytest.fixture(autouse=True)
    def _require(self):
        _require_triton()

    def test_weights_match_with_groups(self):
        """Both implementations should produce identical weights with group selection."""
        from axolotl.integrations.kernels.libs.scattermoe_lora.layers import (
            _sigmoid_topk_route,
        )
        from axolotl.integrations.kernels.sonicmoe.routing import sigmoid_topk_routing

        moe_block, gate, hidden, T, H, E, K = _make_sigmoid_block(
            E=16, K=4, n_group=2, topk_group=1, bias_on_gate=True
        )

        sm_weights, sm_experts, sm_topk, sm_E = _sigmoid_topk_route(
            moe_block, gate, hidden, gate.weight, None
        )

        sonic_scores, sonic_tok_idx, sonic_exp_idx, sonic_logits = sigmoid_topk_routing(
            hidden, moe_block
        )

        sonic_weights_2d = sonic_scores.reshape(T, K)
        sonic_experts_2d = sonic_exp_idx.reshape(T, K)

        assert sm_topk == K
        assert sm_E == E

        # Sort experts within each token to handle different topk orderings
        sm_sorted, sm_order = sm_experts.sort(dim=-1)
        sonic_sorted, sonic_order = sonic_experts_2d.to(sm_experts.dtype).sort(dim=-1)

        assert torch.equal(sm_sorted, sonic_sorted)

        # Gather weights in sorted order for comparison
        sm_weights_sorted = sm_weights.gather(1, sm_order)
        sonic_weights_sorted = sonic_weights_2d.gather(1, sonic_order)
        assert torch.allclose(sm_weights_sorted, sonic_weights_sorted, atol=1e-6)

    def test_weights_match_no_groups(self):
        """Both implementations match without group selection (n_group=1)."""
        from axolotl.integrations.kernels.libs.scattermoe_lora.layers import (
            _sigmoid_topk_route,
        )
        from axolotl.integrations.kernels.sonicmoe.routing import sigmoid_topk_routing

        moe_block, gate, hidden, T, H, E, K = _make_sigmoid_block(
            E=16, K=4, n_group=1, topk_group=1, bias_on_gate=True
        )

        sm_weights, sm_experts, _, _ = _sigmoid_topk_route(
            moe_block, gate, hidden, gate.weight, None
        )
        sonic_scores, _, sonic_exp_idx, _ = sigmoid_topk_routing(hidden, moe_block)

        sonic_weights_2d = sonic_scores.reshape(T, K)
        sonic_experts_2d = sonic_exp_idx.reshape(T, K)

        # Sort for comparison (topk with sorted=False may differ in order)
        sm_sorted, sm_order = sm_experts.sort(dim=-1)
        sonic_sorted, sonic_order = sonic_experts_2d.to(sm_experts.dtype).sort(dim=-1)

        assert torch.equal(sm_sorted, sonic_sorted)
        sm_weights_sorted = sm_weights.gather(1, sm_order)
        sonic_weights_sorted = sonic_weights_2d.gather(1, sonic_order)
        assert torch.allclose(sm_weights_sorted, sonic_weights_sorted, atol=1e-6)

    def test_bias_on_block_parity(self):
        """minimax_m2 style: bias on block, not gate."""
        from axolotl.integrations.kernels.libs.scattermoe_lora.layers import (
            _sigmoid_topk_route,
        )
        from axolotl.integrations.kernels.sonicmoe.routing import sigmoid_topk_routing

        moe_block, gate, hidden, T, H, E, K = _make_sigmoid_block(
            E=16, K=4, n_group=1, bias_on_gate=False
        )

        sm_weights, sm_experts, _, _ = _sigmoid_topk_route(
            moe_block, gate, hidden, gate.weight, None
        )
        sonic_scores, _, sonic_exp_idx, _ = sigmoid_topk_routing(hidden, moe_block)

        sonic_weights_2d = sonic_scores.reshape(T, K)
        sonic_experts_2d = sonic_exp_idx.reshape(T, K)

        sm_sorted, sm_order = sm_experts.sort(dim=-1)
        sonic_sorted, sonic_order = sonic_experts_2d.to(sm_experts.dtype).sort(dim=-1)

        assert torch.equal(sm_sorted, sonic_sorted)
        sm_weights_sorted = sm_weights.gather(1, sm_order)
        sonic_weights_sorted = sonic_weights_2d.gather(1, sonic_order)
        assert torch.allclose(sm_weights_sorted, sonic_weights_sorted, atol=1e-6)

    def test_scaling_factor_parity(self):
        """routed_scaling_factor applied identically by both."""
        from axolotl.integrations.kernels.libs.scattermoe_lora.layers import (
            _sigmoid_topk_route,
        )
        from axolotl.integrations.kernels.sonicmoe.routing import sigmoid_topk_routing

        moe_block, gate, hidden, T, H, E, K = _make_sigmoid_block(
            n_group=1, bias_on_gate=True
        )
        moe_block.routed_scaling_factor = 2.5

        sm_weights, sm_experts, _, _ = _sigmoid_topk_route(
            moe_block, gate, hidden, gate.weight, None
        )
        sonic_scores, _, sonic_exp_idx, _ = sigmoid_topk_routing(hidden, moe_block)

        sonic_weights_2d = sonic_scores.reshape(T, K)
        sonic_experts_2d = sonic_exp_idx.reshape(T, K)

        sm_sorted, sm_order = sm_experts.sort(dim=-1)
        sonic_sorted, sonic_order = sonic_experts_2d.to(sm_experts.dtype).sort(dim=-1)

        assert torch.equal(sm_sorted, sonic_sorted)
        sm_weights_sorted = sm_weights.gather(1, sm_order)
        sonic_weights_sorted = sonic_weights_2d.gather(1, sonic_order)
        assert torch.allclose(sm_weights_sorted, sonic_weights_sorted, atol=1e-6)

    def test_no_renorm_parity(self):
        """norm_topk_prob=False produces same results in both."""
        from axolotl.integrations.kernels.libs.scattermoe_lora.layers import (
            _sigmoid_topk_route,
        )
        from axolotl.integrations.kernels.sonicmoe.routing import sigmoid_topk_routing

        moe_block, gate, hidden, T, H, E, K = _make_sigmoid_block(
            n_group=1, bias_on_gate=True
        )
        moe_block.norm_topk_prob = False

        sm_weights, sm_experts, _, _ = _sigmoid_topk_route(
            moe_block, gate, hidden, gate.weight, None
        )
        sonic_scores, _, sonic_exp_idx, _ = sigmoid_topk_routing(hidden, moe_block)

        sonic_weights_2d = sonic_scores.reshape(T, K)
        sonic_experts_2d = sonic_exp_idx.reshape(T, K)

        sm_sorted, sm_order = sm_experts.sort(dim=-1)
        sonic_sorted, sonic_order = sonic_experts_2d.to(sm_experts.dtype).sort(dim=-1)

        assert torch.equal(sm_sorted, sonic_sorted)
        sm_weights_sorted = sm_weights.gather(1, sm_order)
        sonic_weights_sorted = sonic_weights_2d.gather(1, sonic_order)
        assert torch.allclose(sm_weights_sorted, sonic_weights_sorted, atol=1e-6)


# ============================================================================
# 3. Shared expert parity
# ============================================================================


class TestSharedExpertParity:
    """Verify both _compute_shared_expert implementations behave identically."""

    @pytest.fixture(autouse=True)
    def _require(self):
        _require_triton()

    def _get_both_fns(self):
        from axolotl.integrations.kernels.libs.scattermoe_lora.layers import (
            _compute_shared_expert as scatter_compute,
        )
        from axolotl.integrations.kernels.sonicmoe.patch import (
            _compute_shared_expert as sonic_compute,
        )

        return scatter_compute, sonic_compute

    def test_shared_expert_singular(self):
        scatter_fn, sonic_fn = self._get_both_fns()
        out = torch.randn(4, 8)
        block = SimpleNamespace(shared_expert=lambda x: out)
        hidden = torch.randn(4, 8)

        assert torch.equal(scatter_fn(block, hidden), sonic_fn(block, hidden))

    def test_shared_experts_plural(self):
        scatter_fn, sonic_fn = self._get_both_fns()
        out = torch.randn(4, 8)
        block = SimpleNamespace(shared_experts=lambda x: out)
        hidden = torch.randn(4, 8)

        assert torch.equal(scatter_fn(block, hidden), sonic_fn(block, hidden))

    def test_shared_mlp(self):
        scatter_fn, sonic_fn = self._get_both_fns()
        out = torch.randn(4, 8)
        block = SimpleNamespace(shared_mlp=lambda x: out)
        hidden = torch.randn(4, 8)

        assert torch.equal(scatter_fn(block, hidden), sonic_fn(block, hidden))

    def test_no_shared_expert(self):
        scatter_fn, sonic_fn = self._get_both_fns()
        block = SimpleNamespace()
        hidden = torch.randn(4, 8)

        assert scatter_fn(block, hidden) is None
        assert sonic_fn(block, hidden) is None

    def test_shared_expert_gate_only_in_scattermoe(self):
        """ScatterMoE's _compute_shared_expert handles shared_expert_gate;
        SonicMoE's patch.py handles it externally in the forward function.

        This documents the known divergence: the scattermoe version applies
        sigmoid gating inline, while sonicmoe applies it in the forward.
        """
        scatter_fn, sonic_fn = self._get_both_fns()

        H = 8
        expert_out = torch.ones(4, H)
        gate_fn = lambda x: torch.zeros(4, H)  # noqa: E731  # sigmoid(0) = 0.5

        block = SimpleNamespace(
            shared_expert=lambda x: expert_out,
            shared_expert_gate=gate_fn,
        )
        hidden = torch.randn(4, H)

        scatter_result = scatter_fn(block, hidden)
        sonic_result = sonic_fn(block, hidden)

        # ScatterMoE applies the gate: expert_out * sigmoid(0) = 0.5
        expected_gated = expert_out * 0.5
        assert torch.allclose(scatter_result, expected_gated, atol=1e-6)

        # SonicMoE does NOT apply the gate here (it does it in the forward)
        assert torch.equal(sonic_result, expert_out)


# ============================================================================
# 4. Route dispatcher parity
# ============================================================================


class TestRouteDispatcherParity:
    """Verify _route in scattermoe dispatches correctly and matches individual fns."""

    @pytest.fixture(autouse=True)
    def _require(self):
        _require_triton()

    def test_route_dispatches_softmax(self):
        """_route should use softmax when no e_score_correction_bias."""
        from axolotl.integrations.kernels.libs.scattermoe_lora.layers import (
            _route,
            _softmax_topk_route,
        )

        moe_block, gate, hidden, T, H, E, K = _make_softmax_block()

        route_w, route_e, route_k, route_E = _route(
            moe_block, gate, hidden, gate.weight, None
        )
        direct_w, direct_e, direct_k, direct_E = _softmax_topk_route(
            moe_block, gate, hidden, gate.weight, None
        )

        assert torch.equal(route_w, direct_w)
        assert torch.equal(route_e, direct_e)
        assert route_k == direct_k
        assert route_E == direct_E

    def test_route_dispatches_sigmoid(self):
        """_route should use sigmoid when e_score_correction_bias is present."""
        from axolotl.integrations.kernels.libs.scattermoe_lora.layers import (
            _route,
            _sigmoid_topk_route,
        )

        moe_block, gate, hidden, T, H, E, K = _make_sigmoid_block(
            n_group=1, bias_on_gate=True
        )

        route_w, route_e, route_k, route_E = _route(
            moe_block, gate, hidden, gate.weight, None
        )
        direct_w, direct_e, direct_k, direct_E = _sigmoid_topk_route(
            moe_block, gate, hidden, gate.weight, None
        )

        assert torch.equal(route_w, direct_w)
        assert torch.equal(route_e, direct_e)
        assert route_k == direct_k
        assert route_E == direct_E
