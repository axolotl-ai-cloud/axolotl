"""Integration tests for Voxtral modeling patches."""

import pytest


class TestVoxtralModelingPatchIntegration:
    """Test Voxtral modeling patch integration."""

    @pytest.mark.integration
    def test_voxtral_conditional_generation_patch(self):
        """Test that Voxtral conditional generation patch can be applied."""
        try:
            from transformers.models.voxtral.modeling_voxtral import (
                VoxtralForConditionalGeneration,
            )
        except ImportError:
            pytest.skip("VoxtralForConditionalGeneration not available")

        from axolotl.monkeypatch.models.voxtral.modeling import (
            patch_voxtral_conditional_generation_forward,
        )

        # Store original method
        original_forward = VoxtralForConditionalGeneration.forward

        # Apply patch and get unpatch function
        unpatch_fn = patch_voxtral_conditional_generation_forward()

        # Verify patch was applied
        assert VoxtralForConditionalGeneration.forward != original_forward, (
            "forward method was not patched"
        )

        # Verify the method is still callable
        assert callable(VoxtralForConditionalGeneration.forward), (
            "Patched method is not callable"
        )

        # Test unpatch function
        unpatch_fn()
        assert VoxtralForConditionalGeneration.forward == original_forward, (
            "unpatch function did not restore original method"
        )
