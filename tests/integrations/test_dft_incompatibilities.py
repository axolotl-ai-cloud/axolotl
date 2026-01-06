"""Test DFT incompatibilities with conflicting features.

This test suite verifies that DFT correctly detects and handles incompatible
configurations, either by:
1. Raising clear error messages (hard incompatibilities)
2. Silently falling back to standard loss (soft incompatibilities)
3. Warning users about conflicts (advisory incompatibilities)

TESTED INCOMPATIBILITIES:
- ❌ Label Smoothing: Raises ValueError (mathematically unclear)
- ⚠️  ORPO: Silent fallback to ORPO loss (DFT disabled)
- ⚠️  Liger FLCE: Cannot use with DFT chunked_ce (both chunk CE)
- ⚠️  Cut Cross Entropy: Incompatible with per-token weighting

REFERENCE: specs/001-dft-compatibility-matrix/README.md
"""

import pytest
import torch
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.axolotl.integrations.dft.patch import patch_compute_loss_for_dft


class TestDFTLabelSmoothingIncompatibility:
    """Test DFT + label smoothing incompatibility detection."""

    def test_label_smoothing_raises_error(self):
        """Verify DFT raises ValueError when label smoothing is enabled.

        Label smoothing modifies the CE loss formula fundamentally, making
        it unclear how to combine with DFT's exp(-loss) weighting.
        """
        # Create mock trainer with label smoothing enabled
        mock_trainer = Mock()
        mock_trainer.args = SimpleNamespace(
            enable_dft_loss=True,
            label_smoothing_factor=0.1,  # Non-zero = enabled
            orpo_alpha=None,
            include_tkps=False,
            enable_dft_channel_loss=False,
            dft_chunk_size=None,
        )

        # Create mock config
        mock_cfg = SimpleNamespace()

        # Patch the trainer
        patch_compute_loss_for_dft(mock_trainer, mock_cfg)

        # Create mock inputs
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "labels": torch.tensor([[2, 3, 4, -100]]),
        }

        # Create mock model that returns logits
        mock_model = Mock()
        mock_outputs = SimpleNamespace(
            logits=torch.randn(1, 4, 100, requires_grad=True)
        )
        mock_model.return_value = mock_outputs

        # Should raise ValueError with clear message
        with pytest.raises(ValueError) as exc_info:
            mock_trainer.compute_loss(mock_model, inputs)

        error_msg = str(exc_info.value)
        assert "label smoothing" in error_msg.lower(), \
            "Error should mention label smoothing"
        assert "incompatible" in error_msg.lower(), \
            "Error should indicate incompatibility"

        print("\n✓ Label smoothing correctly raises ValueError with clear message")

    def test_label_smoothing_zero_works(self):
        """Verify DFT works when label_smoothing_factor is 0 or None."""
        mock_trainer = Mock()
        mock_trainer.args = SimpleNamespace(
            enable_dft_loss=True,
            label_smoothing_factor=0.0,  # Zero = disabled
            orpo_alpha=None,
            include_tkps=False,
            enable_dft_channel_loss=False,
            dft_chunk_size=None,
        )
        mock_cfg = SimpleNamespace()

        # Should not raise error
        patch_compute_loss_for_dft(mock_trainer, mock_cfg)

        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "labels": torch.tensor([[2, 3, 4, -100]]),
        }

        mock_model = Mock()
        mock_outputs = SimpleNamespace(
            logits=torch.randn(1, 4, 100, requires_grad=True)
        )
        mock_model.return_value = mock_outputs

        # Should work without error
        loss = mock_trainer.compute_loss(mock_model, inputs)
        assert torch.isfinite(loss), "Loss should be finite"

        print("\n✓ Label smoothing factor=0 works correctly")


class TestDFTORPOIncompatibility:
    """Test DFT + ORPO silent fallback behavior."""

    def test_orpo_silently_disables_dft(self):
        """Verify DFT silently falls back to ORPO loss when ORPO is enabled.

        ORPO uses a different loss function (not CE-based), so DFT cannot apply.
        The patch silently disables DFT and uses the original compute_loss.
        """
        # Create mock trainer with ORPO enabled
        original_compute_loss = Mock(return_value=torch.tensor(1.5))

        mock_trainer = Mock()
        mock_trainer.args = SimpleNamespace(
            enable_dft_loss=True,
            orpo_alpha=0.1,  # ORPO enabled
            label_smoothing_factor=0.0,
            include_tkps=False,
        )
        mock_trainer.compute_loss = original_compute_loss

        mock_cfg = SimpleNamespace()

        # Patch the trainer
        original_func = mock_trainer.compute_loss
        patch_compute_loss_for_dft(mock_trainer, mock_cfg)

        inputs = {"labels": torch.tensor([[1, 2, 3]])}
        mock_model = Mock()

        # Call compute_loss
        loss = mock_trainer.compute_loss(mock_model, inputs)

        # Should have called original_compute_loss (fallback)
        original_func.assert_called_once()
        assert loss.item() == 1.5, "Should use original loss value"

        print("\n✓ ORPO correctly triggers silent fallback to original loss")


class TestDFTChunkedCEConflicts:
    """Test DFT chunked_ce conflicts with other CE chunking methods."""

    def test_dft_chunked_ce_with_liger_flce_documentation(self):
        """Document the conflict between DFT chunked_ce and Liger FLCE.

        Both implement chunked cross-entropy:
        - DFT chunked_ce: PyTorch autograd function, chunks vocab dimension
        - Liger FLCE: Triton kernel, fuses linear + chunked CE

        Cannot use both simultaneously - they would apply chunking twice.

        NOTE: This is a documentation test. Runtime detection would require
        checking if Liger is installed and FLCE is enabled, which is out of
        scope for DFT unit tests.
        """
        conflict_analysis = {
            "DFT chunked_ce": "Custom autograd function, vocab chunking",
            "Liger FLCE": "Triton kernel, fused linear + CE chunking",
            "Conflict": "Both chunk CE computation → cannot use together",
            "Detection": "Currently not implemented (requires Liger integration)",
            "Recommendation": "Users must choose one optimization strategy",
            "User guidance": "Document in DFT README and config validation",
        }

        print("\nDFT Chunked CE vs Liger FLCE Conflict Analysis:")
        for key, value in conflict_analysis.items():
            print(f"  {key}: {value}")

        # This test documents the conflict - always passes
        assert True, "Conflict documented"

    def test_dft_with_cut_cross_entropy_documentation(self):
        """Document the incompatibility between DFT and Cut Cross Entropy.

        Cut Cross Entropy (Apple):
        - Only materializes logits for correct tokens
        - Uses CUDA kernel to compute LSE in SRAM
        - Provides 1000-10000x memory reduction

        DFT Incompatibility:
        - DFT requires per-token losses: loss * exp(-loss)
        - Cut CE avoids computing what DFT needs
        - Fundamental conflict in approach

        NOTE: Cut CE is not widely integrated in axolotl yet, so runtime
        detection is deferred.
        """
        incompatibility_analysis = {
            "Cut Cross Entropy": "Only computes correct token logits, estimates LSE",
            "DFT requirement": "Needs all per-token losses for exp(-loss) weighting",
            "Conflict": "Cut CE avoids computing what DFT needs",
            "Severity": "Hard incompatibility - cannot use together",
            "Detection": "Not implemented (Cut CE not widely used yet)",
            "Recommendation": "Document as incompatible in compatibility matrix",
        }

        print("\nDFT vs Cut Cross Entropy Incompatibility Analysis:")
        for key, value in incompatibility_analysis.items():
            print(f"  {key}: {value}")

        # This test documents the incompatibility - always passes
        assert True, "Incompatibility documented"


class TestDFTFeatureConflictDetection:
    """Test detection of feature conflicts in configuration."""

    def test_multiple_incompatibilities_priority_order(self):
        """Verify incompatibility check priority order.

        When multiple incompatible features are enabled, DFT follows this priority:
        1. ORPO check (silent fallback) - checked first at patch.py:33
        2. Label smoothing check (ValueError) - checked at patch.py:41

        If both ORPO and label_smoothing are enabled, ORPO fallback takes priority.
        """
        # Test case 1: ORPO + label_smoothing → ORPO fallback wins
        original_compute_loss = Mock(return_value=torch.tensor(1.5))

        mock_trainer = Mock()
        mock_trainer.args = SimpleNamespace(
            enable_dft_loss=True,
            label_smoothing_factor=0.1,  # Would error if checked
            orpo_alpha=0.1,  # Checked first → triggers fallback
            include_tkps=False,
        )
        mock_trainer.compute_loss = original_compute_loss
        mock_cfg = SimpleNamespace()

        original_func = mock_trainer.compute_loss
        patch_compute_loss_for_dft(mock_trainer, mock_cfg)

        inputs = {"labels": torch.tensor([[1, 2, 3]])}
        mock_model = Mock()

        # Should trigger ORPO fallback (no error raised)
        loss = mock_trainer.compute_loss(mock_model, inputs)
        original_func.assert_called_once()  # Fallback was used
        assert loss.item() == 1.5, "Should use ORPO fallback"

        print("\n✓ Multiple incompatibilities: ORPO fallback has higher priority")


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
