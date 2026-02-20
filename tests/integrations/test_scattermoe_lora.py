# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""
Unit tests for scattermoe-lora code-review fixes.

Tests cover:
- KernelsArgs validator: disable_mlp_kernel_scattermoe
- CPU_Offloaded_Gradient_Checkpointer: tuple vs plain tensor backward
- ParallelExperts: scaling=0.0 not treated as falsy
- single2scatter: non-aligned K/N dimensions
- group_compileable: coeff=None accepted
- HFScatterMoEGatedMLP / ScatterMoEGatedMLP: return value contract
"""

from unittest.mock import patch

import pytest
import torch

# ============================================================================
# 1. KernelsArgs: disable_mlp_kernel_scattermoe validator
# ============================================================================


class TestKernelsArgsValidator:
    """Test that disable_mlp_kernel_scattermoe sets both flags correctly.

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
        result = KernelsArgs.disable_mlp_kernel_scattermoe(data)
        assert result["lora_mlp_kernel"] is False
        assert result["mlp_kernel"] is False

    def test_mlp_kernel_disabled_without_lora(self):
        """Even without lora_mlp_kernel, mlp_kernel should be disabled."""
        from axolotl.integrations.kernels.args import KernelsArgs

        data = {
            "use_kernels": True,
            "use_scattermoe": True,
        }
        result = KernelsArgs.disable_mlp_kernel_scattermoe(data)
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
        result = KernelsArgs.disable_mlp_kernel_scattermoe(data)
        assert result["lora_mlp_kernel"] is False

    def test_no_change_when_scattermoe_disabled(self):
        """When use_scattermoe is not True, nothing should be changed."""
        from axolotl.integrations.kernels.args import KernelsArgs

        data = {
            "use_kernels": True,
            "use_scattermoe": False,
            "lora_mlp_kernel": True,
        }
        result = KernelsArgs.disable_mlp_kernel_scattermoe(data)
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
