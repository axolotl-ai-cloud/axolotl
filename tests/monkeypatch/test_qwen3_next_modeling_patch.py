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
    def test_qwen3_next_imports_patch(self):
        """Test that Qwen3Next imports patch can be applied without errors."""
        from axolotl.monkeypatch.models.qwen3_next.modeling import (
            patch_qwen3_next_imports,
        )

        # Apply patch - should not raise any exceptions even if modules unavailable
        unpatch_fn = patch_qwen3_next_imports()

        # Test that unpatch function is returned (or None if skipped)
        assert unpatch_fn is None or callable(unpatch_fn), (
            "patch_qwen3_next_imports should return None or callable unpatch function"
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
