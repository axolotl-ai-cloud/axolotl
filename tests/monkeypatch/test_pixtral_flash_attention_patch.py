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

        # Test 1D position_ids 1 sequence
        position_ids_1d = torch.tensor([0, 1, 2, 3])
        result = patched_fn(position_ids_1d, batch_size=1)
        assert isinstance(result, bool), "Function should return a boolean"
        assert result is False, "1D sequential position_ids should not be packed"

        # Test 1D packed 2 sequences
        position_ids_1d_packed = torch.tensor([0, 1, 2, 0, 1, 2])
        result = patched_fn(position_ids_1d_packed, batch_size=1)
        assert isinstance(result, bool), "Function should return a boolean"
        assert result is True, "1D packed position_ids should be detected as packed"

        # Test 2D packed 2 sequences
        position_ids_2d_packed = torch.tensor([[0, 1, 2, 3, 0, 1]])
        result = patched_fn(position_ids_2d_packed, batch_size=1)
        assert isinstance(result, bool), "Function should return a boolean"
        assert result is True, "2D packed position_ids should be detected as packed"

        # Test 2D 1 sequence
        position_ids_2d_normal = torch.tensor([[0, 1, 2, 3, 4, 5]])
        result = patched_fn(position_ids_2d_normal, batch_size=1)
        assert isinstance(result, bool), "Function should return a boolean"
        assert result is False, "2D sequential position_ids should not be packed"

        # Test 2D batch size 2
        position_ids_2d_normal = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8]])
        result = patched_fn(position_ids_2d_normal, batch_size=2)
        assert isinstance(result, bool), "Function should return a boolean"
        assert result is False, "2D position_ids batch 2 should not be packed"

        # Test None case
        result = patched_fn(None, batch_size=1)
        assert isinstance(result, bool), "Function should return a boolean"
        assert result is False, "None position_ids should return False"

        # Test unpatch function
        unpatch_fn()
        assert (
            modeling_flash_attention_utils._is_packed_sequence
            == original_is_packed_sequence
        ), "unpatch function did not restore original method"
