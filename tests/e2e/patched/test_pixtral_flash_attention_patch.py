"""Integration tests for Pixtral Flash Attention patches."""

import pytest
import torch


class TestPixtralFlashAttentionPatchIntegration:
    """Test Pixtral Flash Attention patch integration."""

    @pytest.mark.integration
    def test_pixtral_flash_attention_patch(self):
        """Test that Pixtral Flash Attention patch can be applied and works correctly."""
        try:
            from transformers import modeling_flash_attention_utils
        except ImportError:
            pytest.skip("Flash Attention utils not available")

        from axolotl.monkeypatch.models.pixtral.modeling_flash_attention_utils import (
            apply_patch_is_packed_sequence,
        )

        # Store original method
        original_is_packed_sequence = modeling_flash_attention_utils._is_packed_sequence

        # Apply patch and get unpatch function
        unpatch_fn = apply_patch_is_packed_sequence()

        # Verify patch was applied
        assert (
            modeling_flash_attention_utils._is_packed_sequence
            != original_is_packed_sequence
        ), "_is_packed_sequence was not patched"

        # Test the patched function with 1D position_ids
        patched_fn = modeling_flash_attention_utils._is_packed_sequence

        # Test with 1D position_ids (should work after patch)
        position_ids_1d = torch.tensor([0, 1, 2, 3])
        result = patched_fn(position_ids_1d, batch_size=1)
        assert isinstance(result, bool), "Function should return a boolean"

        # Test unpatch function
        unpatch_fn()
        assert (
            modeling_flash_attention_utils._is_packed_sequence
            == original_is_packed_sequence
        ), "unpatch function did not restore original method"
