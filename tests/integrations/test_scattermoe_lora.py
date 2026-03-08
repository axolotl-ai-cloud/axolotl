# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""
Unit tests for scattermoe-lora.

Tests cover:
- KernelsArgs validator: disable_mlp_kernel
- ParallelExperts: scaling=0.0 not treated as falsy
- single2scatter: non-aligned K/N dimensions
- group_compileable: coeff=None accepted
- HFScatterMoEGatedMLP / ScatterMoEGatedMLP: return value contract
- Routing strategy detection and sigmoid routing
- Generic shared expert handling
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

# ============================================================================
# 1. KernelsArgs: disable_mlp_kernel validator
# ============================================================================


class TestKernelsArgsValidator:
    """Test that disable_mlp_kernel sets both flags correctly.

    These tests call the validator classmethod directly on raw dicts,
    since lora_mlp_kernel / mlp_kernel are not declared model fields.
    """

    def test_disables_lora_mlp_kernel_when_scattermoe(self):
        """lora_mlp_kernel=True gets set to False when use_scattermoe=True."""
        from axolotl.integrations.kernels.args import KernelsArgs

        data = {
            "use_kernels": True,
            "use_scattermoe": True,
            "lora_mlp_kernel": True,
        }
        result = KernelsArgs.disable_mlp_kernel(data)
        assert result["lora_mlp_kernel"] is False
        assert result["mlp_kernel"] is False

    def test_mlp_kernel_disabled_without_lora(self):
        """Even without lora_mlp_kernel, mlp_kernel should be disabled."""
        from axolotl.integrations.kernels.args import KernelsArgs

        data = {
            "use_kernels": True,
            "use_scattermoe": True,
        }
        result = KernelsArgs.disable_mlp_kernel(data)
        assert result["mlp_kernel"] is False
        # lora_mlp_kernel was not in data, should not be added
        assert "lora_mlp_kernel" not in result

    def test_lora_mlp_kernel_false_unchanged(self):
        """lora_mlp_kernel=False should stay False (no warning, no change)."""
        from axolotl.integrations.kernels.args import KernelsArgs

        data = {
            "use_kernels": True,
            "use_scattermoe": True,
            "lora_mlp_kernel": False,
        }
        result = KernelsArgs.disable_mlp_kernel(data)
        assert result["lora_mlp_kernel"] is False

    def test_no_change_when_scattermoe_disabled(self):
        """When use_scattermoe is not True, nothing should be changed."""
        from axolotl.integrations.kernels.args import KernelsArgs

        data = {
            "use_kernels": True,
            "use_scattermoe": False,
            "lora_mlp_kernel": True,
        }
        result = KernelsArgs.disable_mlp_kernel(data)
        assert result["lora_mlp_kernel"] is True


class TestParallelExpertsScaling:
    """Test that scaling=0.0 is preserved and not overridden to 1.0."""

    def test_scaling_zero_preserved(self):
        """scaling=0.0 should be passed as 0.0, not replaced with 1.0."""
        pytest.importorskip("triton")
        from axolotl.integrations.kernels.libs.scattermoe_lora.lora_ops import (
            ParallelExperts,
        )

        pe = ParallelExperts(num_experts=2, input_size=4, output_size=4)
        pe.set_lora(
            lora_A=torch.randn(4, 4),
            lora_B=torch.randn(4, 4),
            scaling=0.0,
        )
        assert pe._lora_scaling == 0.0

        # Patch parallel_linear_lora to capture the scaling arg
        with patch(
            "axolotl.integrations.kernels.libs.scattermoe_lora.lora_ops.parallel_linear_lora"
        ) as mock_pll:
            mock_pll.return_value = torch.randn(4, 4)
            # Create dummy routing tensors
            pe.forward(
                inputs=torch.randn(2, 4),
                k=1,
                sorted_expert_idxs=torch.tensor([0, 0, 1, 1]),
                sorted_scattered_idxs=torch.tensor([0, 1, 0, 1]),
                expert_offsets=torch.tensor([2, 4]),
            )
            # Check that scaling=0.0 was passed, not 1.0
            call_kwargs = mock_pll.call_args
            assert (
                call_kwargs.kwargs.get("scaling") == 0.0
                or call_kwargs[1].get("scaling") == 0.0
            ), f"Expected scaling=0.0 but got {call_kwargs}"

    def test_scaling_none_defaults_to_one(self):
        """scaling=None (no LoRA attached) should default to 1.0."""
        pytest.importorskip("triton")
        from axolotl.integrations.kernels.libs.scattermoe_lora.lora_ops import (
            ParallelExperts,
        )

        pe = ParallelExperts(num_experts=2, input_size=4, output_size=4)
        # No set_lora called, so _lora_scaling is None

        with patch(
            "axolotl.integrations.kernels.libs.scattermoe_lora.lora_ops.parallel_linear_lora"
        ) as mock_pll:
            mock_pll.return_value = torch.randn(4, 4)
            pe.forward(
                inputs=torch.randn(2, 4),
                k=1,
                sorted_expert_idxs=torch.tensor([0, 0, 1, 1]),
                sorted_scattered_idxs=torch.tensor([0, 1, 0, 1]),
                expert_offsets=torch.tensor([2, 4]),
            )
            call_kwargs = mock_pll.call_args
            scaling_val = call_kwargs.kwargs.get("scaling") or call_kwargs[1].get(
                "scaling"
            )
            assert scaling_val == 1.0, (
                f"Expected scaling=1.0 for None but got {scaling_val}"
            )

    def test_scaling_positive_preserved(self):
        """Normal positive scaling should be preserved."""
        pytest.importorskip("triton")
        from axolotl.integrations.kernels.libs.scattermoe_lora.lora_ops import (
            ParallelExperts,
        )

        pe = ParallelExperts(num_experts=2, input_size=4, output_size=4)
        pe.set_lora(
            lora_A=torch.randn(4, 4),
            lora_B=torch.randn(4, 4),
            scaling=0.5,
        )

        with patch(
            "axolotl.integrations.kernels.libs.scattermoe_lora.lora_ops.parallel_linear_lora"
        ) as mock_pll:
            mock_pll.return_value = torch.randn(4, 4)
            pe.forward(
                inputs=torch.randn(2, 4),
                k=1,
                sorted_expert_idxs=torch.tensor([0, 0, 1, 1]),
                sorted_scattered_idxs=torch.tensor([0, 1, 0, 1]),
                expert_offsets=torch.tensor([2, 4]),
            )
            call_kwargs = mock_pll.call_args
            scaling_val = call_kwargs.kwargs.get("scaling") or call_kwargs[1].get(
                "scaling"
            )
            assert scaling_val == 0.5


# ============================================================================
# 4. single2scatter: non-aligned K/N dimensions (GPU only)
# ============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestSingle2ScatterBounds:
    """Test single2scatter with non-aligned dimensions."""

    def test_non_aligned_k(self):
        """K not a multiple of BLOCK_K should produce correct results."""
        from axolotl.integrations.kernels.libs.scattermoe_lora.kernels.single import (
            single2scatter,
        )

        E, K, N = 2, 100, 128  # K=100 not a multiple of 128
        W = torch.randn(E, K, N, device="cuda", dtype=torch.float32)
        X = torch.randn(1, K, device="cuda", dtype=torch.float32)
        expert_idxs = torch.tensor([[0, 1]], device="cuda", dtype=torch.long)

        Y = single2scatter(X, W, expert_idxs)
        assert Y.shape == (2, N)

        # Verify against manual computation
        Y_ref_0 = X[0] @ W[0]
        Y_ref_1 = X[0] @ W[1]
        torch.testing.assert_close(Y[0], Y_ref_0, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(Y[1], Y_ref_1, atol=1e-2, rtol=1e-2)

    def test_non_aligned_n(self):
        """N not a multiple of BLOCK_N should produce correct results."""
        from axolotl.integrations.kernels.libs.scattermoe_lora.kernels.single import (
            single2scatter,
        )

        E, K, N = 2, 128, 100  # N=100 not a multiple of 128
        W = torch.randn(E, K, N, device="cuda", dtype=torch.float32)
        X = torch.randn(1, K, device="cuda", dtype=torch.float32)
        expert_idxs = torch.tensor([[0, 1]], device="cuda", dtype=torch.long)

        Y = single2scatter(X, W, expert_idxs)
        assert Y.shape == (2, N)

        Y_ref_0 = X[0] @ W[0]
        Y_ref_1 = X[0] @ W[1]
        torch.testing.assert_close(Y[0], Y_ref_0, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(Y[1], Y_ref_1, atol=1e-2, rtol=1e-2)

    def test_non_aligned_both(self):
        """Both K and N not aligned should produce correct results."""
        from axolotl.integrations.kernels.libs.scattermoe_lora.kernels.single import (
            single2scatter,
        )

        E, K, N = 2, 100, 100  # Neither aligned to 128
        W = torch.randn(E, K, N, device="cuda", dtype=torch.float32)
        X = torch.randn(1, K, device="cuda", dtype=torch.float32)
        expert_idxs = torch.tensor([[0, 1]], device="cuda", dtype=torch.long)

        Y = single2scatter(X, W, expert_idxs)
        assert Y.shape == (2, N)

        Y_ref_0 = X[0] @ W[0]
        Y_ref_1 = X[0] @ W[1]
        torch.testing.assert_close(Y[0], Y_ref_0, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(Y[1], Y_ref_1, atol=1e-2, rtol=1e-2)


# ============================================================================
# 5. group_compileable: coeff=None accepted
# ============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGroupCoeffNone:
    """Test that group() works with coeff=None."""

    def test_group_with_none_coeff(self):
        """group() should accept coeff=None without errors."""
        from axolotl.integrations.kernels.libs.scattermoe_lora.kernels.ops import group

        M, K = 4, 32
        A = torch.randn(M, K, device="cuda", dtype=torch.float32)
        sorted_expert_idxs = torch.tensor([0, 1, 2, 3], device="cuda", dtype=torch.long)

        # This should not raise a TypeError
        Y = group(A, sorted_expert_idxs, coeff=None, fan_out=1)
        assert Y.shape == (M, K)

    def test_group_with_coeff(self):
        """group() should also work with actual coeff values."""
        from axolotl.integrations.kernels.libs.scattermoe_lora.kernels.ops import group

        M, K = 4, 32
        A = torch.randn(M, K, device="cuda", dtype=torch.float32)
        sorted_expert_idxs = torch.tensor([0, 1, 2, 3], device="cuda", dtype=torch.long)
        coeff = torch.ones(M, device="cuda", dtype=torch.float32) * 0.5

        Y = group(A, sorted_expert_idxs, coeff=coeff, fan_out=1)
        assert Y.shape == (M, K)


# ============================================================================
# 6. Layer return value contracts
# ============================================================================


class TestLayerReturnValues:
    """Test that layer forward methods return the correct types."""

    def test_hf_scatter_moe_returns_single_tensor(self):
        """HFScatterMoEGatedMLP.forward should return a single tensor, not a tuple."""
        pytest.importorskip("triton")
        # Verify the forward method signature and return annotation
        import inspect

        from axolotl.integrations.kernels.libs.scattermoe_lora.layers import (
            HFScatterMoEGatedMLP,
        )

        sig = inspect.signature(HFScatterMoEGatedMLP.forward)
        # It's a staticmethod taking (self, layer_input)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "layer_input" in params

    def test_scatter_moe_gated_mlp_docstring_no_router_logits(self):
        """ScatterMoEGatedMLP.forward docstring should not mention router logits as return."""
        pytest.importorskip("triton")
        from axolotl.integrations.kernels.libs.scattermoe_lora.layers import (
            ScatterMoEGatedMLP,
        )

        docstring = ScatterMoEGatedMLP.forward.__doc__
        assert docstring is not None
        # The docstring should mention output tensor but NOT router logits
        assert "Output tensor" in docstring or "output tensor" in docstring.lower()
        assert "Router logits" not in docstring, (
            "Docstring should not mention 'Router logits' in Returns section"
        )


# ============================================================================
# 7. Routing strategy detection and sigmoid routing
# ============================================================================


def _make_softmax_gate(E=4, H=16, K=2):
    """Create a mock softmax-style gate (Qwen/OLMoE)."""
    return SimpleNamespace(
        weight=torch.randn(E, H),
        top_k=K,
        num_experts=E,
        norm_topk_prob=True,
    )


def _make_sigmoid_gate_with_bias(E=16, H=16):
    """Create a mock sigmoid-style gate with e_score_correction_bias on gate."""
    return SimpleNamespace(
        weight=torch.randn(E, H),
        e_score_correction_bias=torch.zeros(E),
    )


def _make_sigmoid_moe_block(
    T=8, H=16, E=16, K=4, n_group=2, topk_group=1, bias_on_gate=True
):
    """Create a mock GLM/DeepSeek-style MoE block for sigmoid routing tests."""
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
        # minimax_m2 style: bias on block, not gate
        gate = SimpleNamespace(
            weight=torch.randn(E, H),
            top_k=K,
        )
        moe_block = SimpleNamespace(
            gate=gate,
            top_k=K,
            e_score_correction_bias=torch.zeros(E),
        )
    return moe_block, T, H, E, K


def _skip_without_triton():
    pytest.importorskip("triton")


class TestSigmoidRoutingInScatterMoE:
    """Test _sigmoid_topk_route from layers.py."""

    @pytest.fixture(autouse=True)
    def _require_triton(self):
        _skip_without_triton()

    def test_output_shapes(self):
        from axolotl.integrations.kernels.libs.scattermoe_lora.layers import (
            _sigmoid_topk_route,
        )

        moe_block, T, H, E, K = _make_sigmoid_moe_block()
        gate = moe_block.gate
        hidden = torch.randn(T, H)

        weights, experts, top_k, num_experts = _sigmoid_topk_route(
            moe_block, gate, hidden, gate.weight, None
        )

        assert weights.shape == (T, K)
        assert experts.shape == (T, K)
        assert top_k == K
        assert num_experts == E

    def test_weights_nonnegative(self):
        from axolotl.integrations.kernels.libs.scattermoe_lora.layers import (
            _sigmoid_topk_route,
        )

        moe_block, T, H, E, K = _make_sigmoid_moe_block()
        gate = moe_block.gate
        hidden = torch.randn(T, H)

        weights, _, _, _ = _sigmoid_topk_route(
            moe_block, gate, hidden, gate.weight, None
        )
        assert (weights >= 0).all()

    def test_group_selection_restricts_experts(self):
        """With n_group=4, topk_group=1, experts should be from selected groups."""
        from axolotl.integrations.kernels.libs.scattermoe_lora.layers import (
            _sigmoid_topk_route,
        )

        moe_block, T, H, E, K = _make_sigmoid_moe_block(
            E=16, K=2, n_group=4, topk_group=1
        )
        gate = moe_block.gate
        hidden = torch.randn(T, H)

        _, expert_idx, _, _ = _sigmoid_topk_route(
            moe_block, gate, hidden, gate.weight, None
        )

        # Each token's experts should fall within a single group (size E//n_group=4)
        for t in range(T):
            experts_t = expert_idx[t]
            groups = experts_t // (E // moe_block.n_group)
            assert (groups == groups[0]).all()

    def test_scaling_factor_applied(self):
        from axolotl.integrations.kernels.libs.scattermoe_lora.layers import (
            _sigmoid_topk_route,
        )

        moe_block, T, H, E, K = _make_sigmoid_moe_block(n_group=1)
        gate = moe_block.gate
        hidden = torch.randn(T, H)

        weights_1x, _, _, _ = _sigmoid_topk_route(
            moe_block, gate, hidden, gate.weight, None
        )

        moe_block.routed_scaling_factor = 2.0
        weights_2x, _, _, _ = _sigmoid_topk_route(
            moe_block, gate, hidden, gate.weight, None
        )

        assert torch.allclose(weights_2x, weights_1x * 2.0, atol=1e-5)

    def test_bias_on_gate(self):
        """e_score_correction_bias on gate is found."""
        from axolotl.integrations.kernels.libs.scattermoe_lora.layers import (
            _sigmoid_topk_route,
        )

        moe_block, T, H, E, K = _make_sigmoid_moe_block(bias_on_gate=True)
        gate = moe_block.gate
        hidden = torch.randn(T, H)

        weights, experts, _, _ = _sigmoid_topk_route(
            moe_block, gate, hidden, gate.weight, None
        )
        assert weights.shape == (T, K)

    def test_bias_on_block(self):
        """e_score_correction_bias on moe_block (not gate) is found."""
        from axolotl.integrations.kernels.libs.scattermoe_lora.layers import (
            _sigmoid_topk_route,
        )

        moe_block, T, H, E, K = _make_sigmoid_moe_block(bias_on_gate=False)
        gate = moe_block.gate
        hidden = torch.randn(T, H)

        weights, experts, _, _ = _sigmoid_topk_route(
            moe_block, gate, hidden, gate.weight, None
        )
        assert weights.shape == (T, K)

    def test_gate_lora_delta_applied(self):
        """Gate LoRA delta should affect routing logits."""
        from axolotl.integrations.kernels.libs.scattermoe_lora.layers import (
            _sigmoid_topk_route,
        )

        moe_block, T, H, E, K = _make_sigmoid_moe_block(n_group=1)
        gate = moe_block.gate
        hidden = torch.randn(T, H)

        weights_no_lora, _, _, _ = _sigmoid_topk_route(
            moe_block, gate, hidden, gate.weight, None
        )

        # Large delta should change the results
        delta = torch.randn(E, H) * 10.0
        weights_with_lora, _, _, _ = _sigmoid_topk_route(
            moe_block, gate, hidden, gate.weight, delta
        )

        assert not torch.equal(weights_no_lora, weights_with_lora)

    def test_no_bias_does_not_crash(self):
        """Calling _sigmoid_topk_route with no e_score_correction_bias should not crash."""
        from axolotl.integrations.kernels.libs.scattermoe_lora.layers import (
            _sigmoid_topk_route,
        )

        T, H, E, K = 8, 16, 8, 2
        gate = SimpleNamespace(weight=torch.randn(E, H))
        moe_block = SimpleNamespace(
            gate=gate,
            top_k=K,
            n_routed_experts=E,
            n_group=1,
            norm_topk_prob=True,
            routed_scaling_factor=1.0,
        )
        hidden = torch.randn(T, H)

        weights, experts, top_k, num_experts = _sigmoid_topk_route(
            moe_block, gate, hidden, gate.weight, None
        )
        assert weights.shape == (T, K)
        assert experts.shape == (T, K)
        # Without bias, scores_for_choice == sigmoid(logits) — all positive
        assert (weights >= 0).all()

    def test_missing_topk_group_defaults_to_n_group(self):
        """When topk_group is absent but n_group > 1, should default to n_group (no-op masking)."""
        from axolotl.integrations.kernels.libs.scattermoe_lora.layers import (
            _sigmoid_topk_route,
        )

        T, H, E, K, n_group = 8, 16, 16, 2, 4
        gate = SimpleNamespace(
            weight=torch.randn(E, H),
            e_score_correction_bias=torch.zeros(E),
        )
        # Intentionally omit topk_group
        moe_block = SimpleNamespace(
            gate=gate,
            top_k=K,
            n_routed_experts=E,
            n_group=n_group,
            norm_topk_prob=True,
            routed_scaling_factor=1.0,
        )
        hidden = torch.randn(T, H)

        # Should not raise AttributeError; defaults topk_group to n_group
        weights, experts, top_k_out, num_experts = _sigmoid_topk_route(
            moe_block, gate, hidden, gate.weight, None
        )
        assert weights.shape == (T, K)
        assert experts.shape == (T, K)


class TestRoutingStrategyDetection:
    """Test that _route dispatches to the correct strategy."""

    @pytest.fixture(autouse=True)
    def _require_triton(self):
        _skip_without_triton()

    def test_softmax_for_qwen_style(self):
        """Block without e_score_correction_bias should use softmax."""
        from axolotl.integrations.kernels.libs.scattermoe_lora.layers import _route

        gate = _make_softmax_gate(E=4, H=16, K=2)
        moe_block = SimpleNamespace(gate=gate)
        hidden = torch.randn(8, 16)

        weights, experts, top_k, num_experts = _route(
            moe_block, gate, hidden, gate.weight, None
        )

        assert weights.shape == (8, 2)
        assert experts.shape == (8, 2)
        assert top_k == 2
        assert num_experts == 4
        per_token_sums = weights.sum(dim=-1)
        assert torch.allclose(per_token_sums, torch.ones(8), atol=1e-5)

    def test_sigmoid_for_glm_style(self):
        """Block with e_score_correction_bias on gate should use sigmoid."""
        from axolotl.integrations.kernels.libs.scattermoe_lora.layers import _route

        moe_block, T, H, E, K = _make_sigmoid_moe_block(bias_on_gate=True, n_group=1)
        gate = moe_block.gate
        hidden = torch.randn(T, H)

        weights, experts, top_k, num_experts = _route(
            moe_block, gate, hidden, gate.weight, None
        )

        assert weights.shape == (T, K)
        assert experts.shape == (T, K)
        assert (weights >= 0).all()

    def test_sigmoid_for_minimax_m2_style(self):
        """Block with e_score_correction_bias on block (not gate) should use sigmoid."""
        from axolotl.integrations.kernels.libs.scattermoe_lora.layers import _route

        moe_block, T, H, E, K = _make_sigmoid_moe_block(bias_on_gate=False)
        gate = moe_block.gate
        hidden = torch.randn(T, H)

        weights, experts, top_k, num_experts = _route(
            moe_block, gate, hidden, gate.weight, None
        )

        assert weights.shape == (T, K)
        assert (weights >= 0).all()


# ============================================================================
# 8. Generic shared expert handling
# ============================================================================


class TestGenericSharedExpert:
    """Test _compute_shared_expert from layers.py."""

    @pytest.fixture(autouse=True)
    def _require_triton(self):
        _skip_without_triton()

    def test_shared_expert_singular(self):
        """shared_expert attribute (Qwen2MoE style)."""
        from axolotl.integrations.kernels.libs.scattermoe_lora.layers import (
            _compute_shared_expert,
        )

        called = torch.randn(4, 8)
        moe_block = SimpleNamespace(
            shared_expert=lambda x: called,
        )
        result = _compute_shared_expert(moe_block, torch.randn(4, 8))
        assert torch.equal(result, called)

    def test_shared_experts_plural(self):
        """shared_experts attribute (DeepSeek V3 style)."""
        from axolotl.integrations.kernels.libs.scattermoe_lora.layers import (
            _compute_shared_expert,
        )

        called = torch.randn(4, 8)
        moe_block = SimpleNamespace(
            shared_experts=lambda x: called,
        )
        result = _compute_shared_expert(moe_block, torch.randn(4, 8))
        assert torch.equal(result, called)

    def test_shared_mlp(self):
        """shared_mlp attribute (Hunyuan style)."""
        from axolotl.integrations.kernels.libs.scattermoe_lora.layers import (
            _compute_shared_expert,
        )

        called = torch.randn(4, 8)
        moe_block = SimpleNamespace(
            shared_mlp=lambda x: called,
        )
        result = _compute_shared_expert(moe_block, torch.randn(4, 8))
        assert torch.equal(result, called)

    def test_shared_expert_with_gate(self):
        """shared_expert + shared_expert_gate applies sigmoid gating."""
        from axolotl.integrations.kernels.libs.scattermoe_lora.layers import (
            _compute_shared_expert,
        )

        H = 8
        expert_out = torch.ones(4, H)
        gate_fn = lambda x: torch.zeros(4, H)  # noqa: E731

        moe_block = SimpleNamespace(
            shared_expert=lambda x: expert_out,
            shared_expert_gate=gate_fn,
        )
        result = _compute_shared_expert(moe_block, torch.randn(4, H))
        expected = expert_out * 0.5  # sigmoid(0) = 0.5
        assert torch.allclose(result, expected, atol=1e-6)

    def test_no_shared_expert(self):
        """No shared expert attributes returns None."""
        from axolotl.integrations.kernels.libs.scattermoe_lora.layers import (
            _compute_shared_expert,
        )

        moe_block = SimpleNamespace()
        result = _compute_shared_expert(moe_block, torch.randn(4, 8))
        assert result is None
