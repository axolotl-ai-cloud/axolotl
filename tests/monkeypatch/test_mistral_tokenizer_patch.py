"""Integration tests for MistralCommonTokenizer patches."""

import pytest


class TestMistralTokenizerPatchIntegration:
    """Test MistralCommonTokenizer patch integration."""

    @pytest.mark.integration
    def test_mistral_tokenizer_image_patch(self):
        """Test that MistralCommonTokenizer image patch can be applied."""
        try:
            from transformers.tokenization_mistral_common import MistralCommonTokenizer
        except ImportError:
            pytest.skip("MistralCommonTokenizer not available")

        from axolotl.monkeypatch.models.mistral3.mistral_common_tokenizer import (
            apply_mistral_tokenizer_image_patch,
        )

        # Store original method
        original_apply_chat_template = MistralCommonTokenizer.apply_chat_template

        # Apply patch
        apply_mistral_tokenizer_image_patch()

        # Verify patch was applied
        assert (
            MistralCommonTokenizer.apply_chat_template != original_apply_chat_template
        ), "apply_chat_template was not patched"

        # Verify the method is still callable
        assert callable(MistralCommonTokenizer.apply_chat_template), (
            "Patched method is not callable"
        )
