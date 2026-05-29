"""Unit tests for the SonicMoE ExpertsInterface registration."""

from types import SimpleNamespace

import pytest
import torch

from axolotl.integrations.kernels.args import KernelsArgs


class TestKernelsArgs:
    def test_mutual_exclusivity_raises(self):
        with pytest.raises(ValueError, match="Cannot use both"):
            KernelsArgs.model_validate({"use_scattermoe": True, "use_sonicmoe": True})

    def test_sonicmoe_only(self):
        result = KernelsArgs.model_validate({"use_sonicmoe": True})
        assert result.use_sonicmoe is True
        assert result.use_scattermoe is None

    def test_scattermoe_only(self):
        result = KernelsArgs.model_validate({"use_scattermoe": True})
        assert result.use_scattermoe is True
        assert result.use_sonicmoe is None

    def test_neither_set(self):
        result = KernelsArgs.model_validate({})
        assert result.use_scattermoe is None
        assert result.use_sonicmoe is None

    def test_disables_mlp_kernel_when_sonicmoe(self):
        data = {"use_sonicmoe": True, "lora_mlp_kernel": True}
        result = KernelsArgs.disable_mlp_kernel(data)
        assert result["lora_mlp_kernel"] is False
        assert result["mlp_kernel"] is False

    def test_experts_implementation_auto_sonicmoe(self):
        out = KernelsArgs.check_experts_implementation({"use_sonicmoe": True})
        assert out["experts_implementation"] == "sonicmoe"

    def test_experts_implementation_auto_scattermoe(self):
        out = KernelsArgs.check_experts_implementation({"use_scattermoe": True})
        assert out["experts_implementation"] == "scattermoe"

    def test_experts_implementation_default_eager(self):
        out = KernelsArgs.check_experts_implementation({})
        assert out["experts_implementation"] == "eager"

    def test_sonicmoe_impl_requires_flag(self):
        out = KernelsArgs.check_experts_implementation(
            {"experts_implementation": "sonicmoe"}
        )
        assert out["experts_implementation"] == "eager"

    def test_scattermoe_impl_requires_flag(self):
        out = KernelsArgs.check_experts_implementation(
            {"experts_implementation": "scattermoe"}
        )
        assert out["experts_implementation"] == "eager"

    def test_unknown_impl_falls_back_to_eager(self):
        out = KernelsArgs.check_experts_implementation(
            {"experts_implementation": "not-a-real-impl"}
        )
        assert out["experts_implementation"] == "eager"

    def test_builtin_impls_pass_through(self):
        for impl in ("eager", "batched_mm", "grouped_mm"):
            out = KernelsArgs.check_experts_implementation(
                {"experts_implementation": impl}
            )
            assert out["experts_implementation"] == impl


class TestSonicMoERegistration:
    """Test that register_sonicmoe_experts plugs into ALL_EXPERTS_FUNCTIONS."""

    def test_register_adds_entry(self):
        from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS

        from axolotl.integrations.kernels.libs.sonicmoe.experts import (
            register_sonicmoe_experts,
            sonicmoe_experts_forward_with_lora,
        )

        register_sonicmoe_experts()
        assert "sonicmoe" in ALL_EXPERTS_FUNCTIONS
        assert ALL_EXPERTS_FUNCTIONS["sonicmoe"] is sonicmoe_experts_forward_with_lora

    def test_register_is_idempotent(self):
        from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS

        from axolotl.integrations.kernels.libs.sonicmoe.experts import (
            register_sonicmoe_experts,
        )

        register_sonicmoe_experts()
        register_sonicmoe_experts()
        # Just one entry, no error
        assert "sonicmoe" in ALL_EXPERTS_FUNCTIONS

    def test_register_overrides_upstream(self):
        """Axolotl's LoRA-aware variant replaces upstream's plain forward."""
        from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS
        from transformers.integrations.sonicmoe import sonicmoe_experts_forward

        from axolotl.integrations.kernels.libs.sonicmoe.experts import (
            register_sonicmoe_experts,
            sonicmoe_experts_forward_with_lora,
        )

        register_sonicmoe_experts()
        assert ALL_EXPERTS_FUNCTIONS["sonicmoe"] is sonicmoe_experts_forward_with_lora
        assert ALL_EXPERTS_FUNCTIONS["sonicmoe"] is not sonicmoe_experts_forward


class TestMoELoRAMaterialize:
    """Verify the LoRA materialization autograd Function used by the registered forward."""

    def test_forward_shape_and_identity_with_zero_lora(self):
        """W_eff == base when LoRA tensors are zero, regardless of layout convention."""
        from axolotl.integrations.kernels.libs.sonicmoe.lora import MoELoRAMaterialize

        E, dim1, dim2, r = 4, 8, 6, 2
        base = torch.randn(E, dim1, dim2)
        lora_A = torch.zeros(r * E, dim2)
        lora_B = torch.zeros(dim1, r * E)
        scaling = 0.5

        W_eff = MoELoRAMaterialize.apply(base, lora_A, lora_B, scaling)
        assert W_eff.shape == base.shape
        torch.testing.assert_close(W_eff, base, atol=1e-6, rtol=1e-6)

    def test_forward_scaling_linearity(self):
        """Doubling scaling should double the LoRA delta."""
        from axolotl.integrations.kernels.libs.sonicmoe.lora import MoELoRAMaterialize

        E, dim1, dim2, r = 4, 8, 6, 2
        base = torch.randn(E, dim1, dim2)
        lora_A = torch.randn(r * E, dim2)
        lora_B = torch.randn(dim1, r * E)

        W_1 = MoELoRAMaterialize.apply(base, lora_A, lora_B, 1.0)
        W_2 = MoELoRAMaterialize.apply(base, lora_A, lora_B, 2.0)
        torch.testing.assert_close(W_2 - base, 2 * (W_1 - base), atol=1e-5, rtol=1e-5)

    def test_forward_matches_peft_einsum(self):
        """Delta matches PEFT's ParamWrapper.get_delta_weight einsum convention.

        Reference: ``peft.tuners.lora.layer.ParamWrapper.get_delta_weight``
        on PEFT 0.19.x — ``einsum("o r e, e r i -> e o i", B_3d, A_3d)`` where
        ``B_3d = lora_B.reshape(dim1, r, E)`` and ``A_3d = lora_A.reshape(E, r, dim2)``.
        """
        from axolotl.integrations.kernels.libs.sonicmoe.lora import MoELoRAMaterialize

        E, dim1, dim2, r = 3, 5, 4, 2
        base = torch.zeros(E, dim1, dim2)
        lora_A = torch.randn(r * E, dim2)
        lora_B = torch.randn(dim1, r * E)
        scaling = 0.7

        W_eff = MoELoRAMaterialize.apply(base, lora_A, lora_B, scaling)

        # PEFT's reference computation
        A_3d = lora_A.reshape(E, r, dim2)
        B_3d = lora_B.reshape(dim1, r, E)
        peft_delta = torch.einsum("o r e, e r i -> e o i", B_3d, A_3d) * scaling

        torch.testing.assert_close(W_eff, peft_delta, atol=1e-5, rtol=1e-5)

    def test_gradient_flows_to_lora(self):
        from axolotl.integrations.kernels.libs.sonicmoe.lora import MoELoRAMaterialize

        E, dim1, dim2, r = 4, 8, 6, 2
        base = torch.randn(E, dim1, dim2, requires_grad=False)
        lora_A = torch.randn(r * E, dim2, requires_grad=True)
        lora_B = torch.randn(dim1, r * E, requires_grad=True)
        scaling = 0.5

        W_eff = MoELoRAMaterialize.apply(base, lora_A, lora_B, scaling)
        loss = W_eff.sum()
        loss.backward()

        assert lora_A.grad is not None
        assert lora_B.grad is not None
        assert lora_A.grad.abs().max() > 0
        assert lora_B.grad.abs().max() > 0
        # Base weight is frozen — no grad expected.
        assert base.grad is None

    def test_no_lora_returns_base_unchanged(self):
        from axolotl.integrations.kernels.libs.sonicmoe.lora import (
            materialize_expert_lora,
        )

        base = torch.randn(4, 8, 6)
        result = materialize_expert_lora(base, None)
        assert result is base


class TestExpertsClassMetadata:
    """The forward reads `has_gate`/`has_bias`/`is_transposed`/`is_concatenated`
    that are set by transformers' @use_experts_implementation decorator.
    Verify our forward respects these without an actual CUDA kernel call.
    """

    def test_rejects_non_gated(self):
        from axolotl.integrations.kernels.libs.sonicmoe.experts import (
            sonicmoe_experts_forward_with_lora,
        )

        fake_self = SimpleNamespace(has_gate=False)
        hidden = torch.zeros(2, 4)
        top_k_index = torch.zeros(2, 1, dtype=torch.long)
        top_k_weights = torch.ones(2, 1)

        with pytest.raises(ValueError, match="has_gate"):
            sonicmoe_experts_forward_with_lora(
                fake_self, hidden, top_k_index, top_k_weights
            )

    def test_rejects_non_cuda(self):
        from axolotl.integrations.kernels.libs.sonicmoe.experts import (
            sonicmoe_experts_forward_with_lora,
        )

        fake_self = SimpleNamespace(has_gate=True)
        hidden = torch.zeros(2, 4)  # CPU tensor
        top_k_index = torch.zeros(2, 1, dtype=torch.long)
        top_k_weights = torch.ones(2, 1)

        with pytest.raises(ValueError, match="CUDA"):
            sonicmoe_experts_forward_with_lora(
                fake_self, hidden, top_k_index, top_k_weights
            )
