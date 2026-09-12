"""Integration tests for Qwen3 Next modeling patches."""

import pytest
import torch

# Skip entire module if qwen3_next not available
qwen3_next = pytest.importorskip("transformers.models.qwen3_next.modeling_qwen3_next")


class TestQwen3NextModelingPatchIntegration:
    """Test Qwen3 Next modeling patch integration."""

    @pytest.mark.integration
    def test_qwen3_next_decoder_layer_patch(self):
        """Test that Qwen3Next decoder layer patch can be applied."""
        from axolotl.monkeypatch.models.qwen3_next.modeling import (
            patch_qwen3_next_decoder_layer,
        )

        # Store original method
        original_forward = qwen3_next.Qwen3NextDecoderLayer.forward

        # Apply patch and get unpatch function
        unpatch_fn = patch_qwen3_next_decoder_layer()

        # Verify patch was applied
        assert qwen3_next.Qwen3NextDecoderLayer.forward != original_forward, (
            "decoder layer forward method was not patched"
        )

        # Verify the method is still callable
        assert callable(qwen3_next.Qwen3NextDecoderLayer.forward), (
            "Patched method is not callable"
        )

        # Test unpatch function
        if unpatch_fn:
            unpatch_fn()
            assert qwen3_next.Qwen3NextDecoderLayer.forward == original_forward, (
                "unpatch function did not restore original method"
            )

    @pytest.mark.integration
    def test_qwen3_next_gateddelta_layer_patch(self):
        """Test that Qwen3Next GatedDeltaNet patch can be applied."""
        from axolotl.monkeypatch.models.qwen3_next.modeling import (
            patch_qwen3_next_gateddelta_layer,
        )

        # Store original method
        original_forward = qwen3_next.Qwen3NextGatedDeltaNet.forward

        # Apply patch and get unpatch function
        unpatch_fn = patch_qwen3_next_gateddelta_layer()

        # Verify patch was applied
        assert qwen3_next.Qwen3NextGatedDeltaNet.forward != original_forward, (
            "GatedDeltaNet forward method was not patched"
        )

        # Verify the method is still callable
        assert callable(qwen3_next.Qwen3NextGatedDeltaNet.forward), (
            "Patched method is not callable"
        )

        # Test unpatch function
        if unpatch_fn:
            unpatch_fn()
            assert qwen3_next.Qwen3NextGatedDeltaNet.forward == original_forward, (
                "unpatch function did not restore original method"
            )

    @pytest.mark.integration
    def test_qwen3_next_modeling_packing_patch(self):
        """Test that all Qwen3Next modeling patches can be applied together."""
        from axolotl.monkeypatch.models.qwen3_next.modeling import (
            patch_qwen3_next_modeling_packing,
        )

        # This should not raise any exceptions
        patch_qwen3_next_modeling_packing()

    @pytest.mark.integration
    def test_linear_attention_receives_position_ids(self):
        """The patched decoder layer must route linear-attention layers with position_ids."""
        from torch import nn
        from transformers.models.qwen3_next.configuration_qwen3_next import (
            Qwen3NextConfig,
        )

        from axolotl.monkeypatch.models.qwen3_next.modeling import (
            patch_qwen3_next_decoder_layer,
        )

        class Recorder(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = []

            def forward(self, **kwargs):
                self.calls.append(kwargs)
                return torch.ones_like(kwargs["hidden_states"])

        class StubMLP(nn.Module):
            def forward(self, hidden_states):
                return torch.ones_like(hidden_states)

        cfg = Qwen3NextConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            linear_num_value_heads=4,
            linear_num_key_heads=2,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            linear_conv_kernel_dim=4,
            layer_types=["linear_attention"],
            num_experts=2,
            num_experts_per_tok=1,
            moe_intermediate_size=32,
            shared_expert_intermediate_size=32,
        )
        torch.manual_seed(0)
        layer = qwen3_next.Qwen3NextDecoderLayer(cfg, 0).eval()
        mixer = Recorder()
        layer.linear_attn = mixer
        layer.mlp = StubMLP()

        unpatch_fn = patch_qwen3_next_decoder_layer()
        try:
            hidden_states = torch.randn(2, 6, cfg.hidden_size)
            position_ids = torch.arange(6).expand(2, 6).contiguous()
            cos = torch.ones(2, 6, 8)
            out = layer(
                hidden_states,
                position_embeddings=(cos, cos.clone()),
                attention_mask=None,
                position_ids=position_ids,
            )
        finally:
            if unpatch_fn:
                unpatch_fn()

        assert len(mixer.calls) == 1
        assert torch.equal(mixer.calls[0]["position_ids"], position_ids)
        assert out.shape == hidden_states.shape


@pytest.mark.integration
def test_get_cu_seqlens_utility():
    """Test the get_cu_seqlens utility function."""
    from axolotl.monkeypatch.models.qwen3_next.modeling import get_cu_seqlens

    # Test with simple position_ids
    position_ids = torch.tensor([[0, 1, 2, 0, 1]])
    cu_seqlens = get_cu_seqlens(position_ids)
    assert cu_seqlens.dtype == torch.int32, "Should be int32 dtype"

    # Should return tensor with start positions and total length
    expected = torch.tensor([0, 3, 5], dtype=torch.int32)
    assert torch.equal(cu_seqlens, expected), f"Expected {expected}, got {cu_seqlens}"
