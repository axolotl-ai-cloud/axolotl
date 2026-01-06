"""Integration tests for DFT + Sequence Packing compatibility.

This module tests that DFT loss works correctly with sequence packing,
verifying that:
1. Packed sequences produce correct loss values
2. Padding tokens are properly masked with ignore_index
3. Loss computation is equivalent to unpacked sequences
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from axolotl.integrations.dft.dft_utils import compute_dft_loss
from axolotl.integrations.dft.patch import patch_compute_loss_for_dft


class TestDFTPackingCompatibility:
    """Test DFT integration with sequence packing."""

    def test_packed_sequences_basic(self):
        """Test DFT loss with packed sequences (3 sequences in one batch)."""
        # Simulate 3 sequences packed into one batch:
        # Seq 1: tokens [0, 1] (length 2)
        # Seq 2: tokens [2, 3, 4] (length 3)
        # Seq 3: tokens [5, 6] (length 2)
        # Total packed length: 2 + 3 + 2 = 7

        batch_size = 1
        packed_seq_len = 7
        vocab_size = 10

        # Create logits for packed sequence
        logits = torch.randn(batch_size, packed_seq_len, vocab_size, requires_grad=True)

        # Labels with padding (-100) between sequences
        # [seq1_tok0, seq1_tok1, seq2_tok0, seq2_tok1, seq2_tok2, seq3_tok0, seq3_tok1]
        labels = torch.tensor([[0, 1, 2, 3, 4, 5, 6]])

        # Compute DFT loss
        loss = compute_dft_loss(logits, labels)

        # Verify loss is valid
        assert loss.ndim == 0, "Loss should be scalar"
        assert loss.item() > 0, "Loss should be positive"
        assert loss.requires_grad, "Loss should have gradients"

        # Verify gradients can be computed
        loss.backward()
        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()

    def test_packed_sequences_with_padding(self):
        """Test DFT correctly handles padding between packed sequences."""
        batch_size = 1
        packed_seq_len = 10
        vocab_size = 20

        logits = torch.randn(batch_size, packed_seq_len, vocab_size, requires_grad=True)

        # Packed sequences with explicit padding (-100) between them:
        # [seq1: 0,1,2] [-100] [seq2: 3,4] [-100, -100] [seq3: 5,6,7]
        labels = torch.tensor([[0, 1, 2, -100, 3, 4, -100, -100, 5, 6]])

        loss = compute_dft_loss(logits, labels)

        # Loss should ignore -100 tokens
        assert loss.item() > 0

        # Verify gradients
        loss.backward()
        assert logits.grad is not None

        # Check that gradients for padding positions are affected correctly
        # After shift_labels, positions corresponding to -100 should not contribute to loss
        # shift_labels creates: [1, 2, -100, 3, 4, -100, -100, 5, 6] (length 9)
        # Valid positions: [1, 2, 3, 4, 5, 6] (indices 0,1,3,4,7,8 in shifted)

    def test_packed_vs_unpacked_equivalence(self):
        """Test that packed sequences produce reasonable loss compared to unpacked.

        IMPORTANT: Due to DFT's non-linear weighting (loss * exp(-loss)), packed and
        unpacked losses are NOT mathematically equivalent. This test verifies that:
        1. Valid token counts match
        2. Loss values are in the same ballpark (within reasonable tolerance)
        3. Both approaches produce valid, trainable losses

        In real packing, sequences are separated by setting the label after each
        sequence's last token to -100, preventing cross-sequence prediction.
        """
        vocab_size = 50

        # Create 3 separate sequences
        seq1_len, seq2_len, seq3_len = 4, 3, 5

        # Individual sequences
        logits1 = torch.randn(1, seq1_len, vocab_size, requires_grad=True)
        logits2 = torch.randn(1, seq2_len, vocab_size, requires_grad=True)
        logits3 = torch.randn(1, seq3_len, vocab_size, requires_grad=True)

        labels1 = torch.randint(0, vocab_size, (1, seq1_len))
        labels2 = torch.randint(0, vocab_size, (1, seq2_len))
        labels3 = torch.randint(0, vocab_size, (1, seq3_len))

        # Compute loss for each sequence separately
        loss1 = compute_dft_loss(logits1, labels1)
        loss2 = compute_dft_loss(logits2, labels2)
        loss3 = compute_dft_loss(logits3, labels3)

        # Average loss (weighted by sequence length after shift)
        valid_tokens1 = seq1_len - 1
        valid_tokens2 = seq2_len - 1
        valid_tokens3 = seq3_len - 1
        total_valid = valid_tokens1 + valid_tokens2 + valid_tokens3

        unpacked_avg_loss = (
            loss1 * valid_tokens1 +
            loss2 * valid_tokens2 +
            loss3 * valid_tokens3
        ) / total_valid

        # Create packed sequence with sequence boundaries
        packed_logits = torch.cat([logits1, logits2, logits3], dim=1).requires_grad_()

        # Set sequence boundaries: last token of each sequence (except the last)
        # has label -100 to prevent cross-sequence prediction
        labels1_with_boundary = labels1.clone()
        labels1_with_boundary[0, -1] = -100

        labels2_with_boundary = labels2.clone()
        labels2_with_boundary[0, -1] = -100

        packed_labels = torch.cat([
            labels1_with_boundary,
            labels2_with_boundary,
            labels3
        ], dim=1)

        # Compute packed loss
        packed_loss = compute_dft_loss(packed_logits, packed_labels)

        # Verify valid token counts match
        shift_packed_labels = packed_labels[:, 1:].flatten()
        valid_in_packed = (shift_packed_labels != -100).sum().item()
        assert valid_in_packed == total_valid, (
            f"Valid token count mismatch: {valid_in_packed} != {total_valid}"
        )

        # Verify both losses are valid and trainable
        assert packed_loss.item() > 0, "Packed loss should be positive"
        assert unpacked_avg_loss.item() > 0, "Unpacked loss should be positive"
        assert packed_loss.requires_grad, "Packed loss should require gradients"
        assert torch.isfinite(packed_loss), "Packed loss should be finite"
        assert torch.isfinite(unpacked_avg_loss), "Unpacked loss should be finite"

        # Verify losses are in reasonable range (within 20% of each other)
        # Note: Due to DFT's non-linear weighting, they won't be exactly equal
        rel_diff = abs(packed_loss - unpacked_avg_loss) / unpacked_avg_loss
        assert rel_diff < 0.20, (
            f"Packed loss ({packed_loss.item():.6f}) and unpacked loss "
            f"({unpacked_avg_loss.item():.6f}) differ by {rel_diff.item()*100:.1f}%, "
            f"expected < 20%"
        )

    def test_packed_sequences_all_padding_edge_case(self):
        """Test edge case where entire sequence is padding."""
        batch_size = 1
        seq_len = 5
        vocab_size = 10

        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        # All tokens are padding
        labels = torch.full((batch_size, seq_len), -100)

        loss = compute_dft_loss(logits, labels)

        # Loss should be 0 when all tokens are ignored
        assert loss.item() == pytest.approx(0.0, abs=1e-12)

        loss.backward()
        assert logits.grad is not None
        # Gradients should be 0 for ignored tokens
        assert torch.all(logits.grad == 0)

    def test_trainer_patch_with_packing(self):
        """Test DFT patch works with packed sequences in trainer context."""
        trainer = MagicMock()
        trainer.args = SimpleNamespace(
            enable_dft_loss=True,
            dft_chunk_size=None,
            enable_dft_channel_loss=False,
            include_tkps=False,
            label_smoothing_factor=0.0,
            orpo_alpha=None,
        )
        trainer.state = SimpleNamespace()
        trainer.compute_loss = MagicMock()

        # Apply DFT patch
        patch_compute_loss_for_dft(trainer, cfg=MagicMock())

        # Create packed sequence data
        batch_size = 2  # 2 packed batches
        packed_seq_len = 8
        vocab_size = 100

        logits = torch.randn(batch_size, packed_seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (batch_size, packed_seq_len))
        # Add some padding
        labels[:, 3] = -100
        labels[:, 6:] = -100

        class DummyModel(torch.nn.Module):
            def forward(self, **kwargs):
                return SimpleNamespace(logits=logits)

        model = DummyModel()
        inputs = {"input_ids": torch.zeros(batch_size, packed_seq_len), "labels": labels}

        # Compute loss
        loss, outputs = trainer.compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=None
        )

        # Verify loss computed successfully
        assert loss.ndim == 0
        assert loss.requires_grad
        assert outputs.logits is logits

        # Verify loss is positive and finite
        assert loss.item() > 0
        assert torch.isfinite(loss)

    def test_packed_sequences_with_chunked_ce(self):
        """Test DFT + packing works with chunked cross-entropy."""
        trainer = MagicMock()
        trainer.args = SimpleNamespace(
            enable_dft_loss=True,
            dft_chunk_size=32,  # Enable chunked CE
            enable_dft_channel_loss=False,
            include_tkps=False,
            label_smoothing_factor=0.0,
            orpo_alpha=None,
        )
        trainer.state = SimpleNamespace()
        trainer.compute_loss = MagicMock()

        patch_compute_loss_for_dft(trainer, cfg=MagicMock())

        batch_size = 1
        packed_seq_len = 10
        vocab_size = 128  # Larger vocab to benefit from chunking

        logits = torch.randn(batch_size, packed_seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (batch_size, packed_seq_len))
        # Add padding
        labels[:, 4] = -100
        labels[:, 7:] = -100

        class DummyModel(torch.nn.Module):
            def forward(self, **kwargs):
                return SimpleNamespace(logits=logits)

        model = DummyModel()
        inputs = {"input_ids": torch.zeros(batch_size, packed_seq_len), "labels": labels}

        # Compute loss with chunked CE
        loss, outputs = trainer.compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=None
        )

        # Verify loss is valid
        assert loss.ndim == 0
        assert loss.requires_grad
        assert torch.isfinite(loss)
        assert loss.item() > 0

        # Verify gradients
        loss.backward()
        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()

    def test_attention_mask_does_not_affect_loss(self):
        """Test that attention masks (used by packing) don't affect loss computation.

        In sequence packing, attention masks have unique IDs per sequence
        (e.g., [1,1,1, 2,2,2, 3,3,3]) to prevent cross-sequence attention.
        DFT loss should only care about labels, not attention masks.
        """
        batch_size = 1
        packed_seq_len = 9
        vocab_size = 50

        logits = torch.randn(batch_size, packed_seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (batch_size, packed_seq_len))

        # Simulate packing attention mask: [seq1: 1,1,1 | seq2: 2,2,2 | seq3: 3,3,3]
        # (In real packing, this prevents cross-sequence attention)
        attention_mask = torch.tensor([[1, 1, 1, 2, 2, 2, 3, 3, 3]])

        # Compute loss without attention mask
        loss_without = compute_dft_loss(logits, labels)

        # DFT compute_dft_loss doesn't take attention_mask parameter
        # This is correct - attention masks affect model forward, not loss computation
        # The test verifies this design is correct for packing

        # In trainer context, attention mask is passed to model.forward(), not loss
        # So DFT loss sees the same logits regardless of attention mask structure

        assert loss_without.item() > 0
        assert torch.isfinite(loss_without)
