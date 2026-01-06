"""Unit tests for Chunked Cross-Entropy implementation."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from axolotl.integrations.dft.chunked_ce import (
    chunked_cross_entropy,
)
from axolotl.integrations.dft.dft_utils import compute_per_token_cross_entropy


class TestChunkedCrossEntropy:
    """Test suite for ChunkedCrossEntropy autograd function."""

    def test_mathematical_equivalence_small_vocab(self):
        """Test that chunked CE produces identical results to standard CE."""
        torch.manual_seed(42)
        batch_size, seq_len, vocab_size = 4, 32, 1000
        chunk_size = 16

        # Create random logits and labels
        logits = torch.randn(batch_size * seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (batch_size * seq_len,))

        # Compute standard cross-entropy
        loss_standard = F.cross_entropy(
            logits.detach(),
            labels,
            reduction="none",
            ignore_index=-100,
        )

        # Compute chunked cross-entropy
        loss_chunked = chunked_cross_entropy(
            logits.detach(),
            labels,
            chunk_size=chunk_size,
            ignore_index=-100,
        )

        # Assert mathematical equivalence
        assert torch.allclose(loss_standard, loss_chunked, atol=1e-6), (
            f"Max diff: {(loss_standard - loss_chunked).abs().max().item()}"
        )

    def test_gradient_correctness(self):
        """Test that gradients computed by chunked CE match standard CE."""
        torch.manual_seed(123)
        batch_size, seq_len, vocab_size = 2, 16, 500
        chunk_size = 8

        # Create test data
        logits_std = torch.randn(batch_size * seq_len, vocab_size, requires_grad=True)
        logits_chunk = logits_std.detach().clone().requires_grad_(True)
        labels = torch.randint(0, vocab_size, (batch_size * seq_len,))

        # Standard CE
        loss_std = F.cross_entropy(logits_std, labels, reduction="none")
        loss_std.sum().backward()

        # Chunked CE
        loss_chunk = chunked_cross_entropy(logits_chunk, labels, chunk_size=chunk_size)
        loss_chunk.sum().backward()

        # Compare gradients
        assert torch.allclose(logits_std.grad, logits_chunk.grad, atol=1e-5), (
            f"Max gradient diff: {(logits_std.grad - logits_chunk.grad).abs().max().item()}"
        )

    def test_ignore_index_handling(self):
        """Test that ignore_index is properly handled."""
        torch.manual_seed(456)
        seq_len, vocab_size = 32, 100
        chunk_size = 16
        ignore_index = -100

        logits = torch.randn(seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (seq_len,))

        # Set some labels to ignore_index
        labels[::3] = ignore_index  # Every 3rd token

        # Compute losses
        loss_std = F.cross_entropy(
            logits.detach(),
            labels,
            reduction="none",
            ignore_index=ignore_index,
        )
        loss_chunk = chunked_cross_entropy(
            logits.detach(),
            labels,
            chunk_size=chunk_size,
            ignore_index=ignore_index,
        )

        # Check that ignored positions have zero loss
        assert torch.all(loss_std[labels == ignore_index] == 0.0)
        assert torch.all(loss_chunk[labels == ignore_index] == 0.0)

        # Check equivalence for non-ignored positions
        valid_mask = labels != ignore_index
        assert torch.allclose(
            loss_std[valid_mask],
            loss_chunk[valid_mask],
            atol=1e-6,
        )

    def test_chunk_size_larger_than_sequence(self):
        """Test behavior when chunk_size > sequence length."""
        torch.manual_seed(789)
        seq_len, vocab_size = 16, 200
        chunk_size = 100  # Larger than seq_len

        logits = torch.randn(seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (seq_len,))

        # Should still work correctly
        loss_std = F.cross_entropy(logits.detach(), labels, reduction="none")
        loss_chunk = chunked_cross_entropy(
            logits.detach(),
            labels,
            chunk_size=chunk_size,
        )

        assert torch.allclose(loss_std, loss_chunk, atol=1e-6)

    def test_single_chunk(self):
        """Test edge case with only one chunk."""
        torch.manual_seed(101112)
        seq_len, vocab_size = 5, 50
        chunk_size = 10  # Larger than seq_len, so only 1 chunk

        logits = torch.randn(seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (seq_len,))

        loss_std = F.cross_entropy(logits.detach(), labels, reduction="none")
        loss_chunk = chunked_cross_entropy(
            logits.detach(), labels, chunk_size=chunk_size
        )

        assert torch.allclose(loss_std, loss_chunk, atol=1e-6)

    def test_exact_chunk_boundaries(self):
        """Test when sequence length is exactly divisible by chunk_size."""
        torch.manual_seed(131415)
        seq_len, vocab_size = 64, 300
        chunk_size = 16  # 64 / 16 = 4 chunks exactly

        logits = torch.randn(seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (seq_len,))

        loss_std = F.cross_entropy(logits.detach(), labels, reduction="none")
        loss_chunk = chunked_cross_entropy(
            logits.detach(), labels, chunk_size=chunk_size
        )

        assert torch.allclose(loss_std, loss_chunk, atol=1e-6)

    def test_gradient_flow_with_dft_weighting(self):
        """Test gradient flow when combined with DFT weighting."""
        torch.manual_seed(161718)
        seq_len, vocab_size = 32, 100
        chunk_size = 16

        logits = torch.randn(seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (seq_len,))

        # Compute chunked CE
        loss = chunked_cross_entropy(logits, labels, chunk_size=chunk_size)

        # Apply DFT-like weighting
        with torch.no_grad():
            weights = torch.exp(-loss)

        weighted_loss = (loss * weights).sum()
        weighted_loss.backward()

        # Check that gradients exist and are finite
        assert logits.grad is not None
        assert torch.all(torch.isfinite(logits.grad))

    @pytest.mark.parametrize("chunk_size", [4, 8, 16, 32])
    def test_different_chunk_sizes(self, chunk_size):
        """Test that different chunk sizes all produce correct results."""
        torch.manual_seed(192021)
        seq_len, vocab_size = 64, 200

        logits = torch.randn(seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (seq_len,))

        loss_std = F.cross_entropy(logits, labels, reduction="none")
        loss_chunk = chunked_cross_entropy(logits, labels, chunk_size=chunk_size)

        assert torch.allclose(loss_std, loss_chunk, atol=1e-6)


class TestIntegrationWithDFTUtils:
    """Test integration with compute_per_token_cross_entropy."""

    def test_compute_per_token_cross_entropy_with_chunking(self):
        """Test that compute_per_token_cross_entropy works with chunk_size."""
        torch.manual_seed(222324)
        batch_size, seq_len, vocab_size = 2, 32, 500
        chunk_size = 16

        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Without chunking
        loss_std, mask_std = compute_per_token_cross_entropy(
            logits,
            labels,
            shift_labels=True,
            chunk_size=None,
        )

        # With chunking
        loss_chunk, mask_chunk = compute_per_token_cross_entropy(
            logits,
            labels,
            shift_labels=True,
            chunk_size=chunk_size,
        )

        # Masks should be identical
        assert torch.all(mask_std == mask_chunk)

        # Losses should be close
        assert torch.allclose(loss_std, loss_chunk, atol=1e-6)

    def test_large_vocabulary_scenario(self):
        """Test with vocabulary size similar to Qwen (152K tokens)."""
        torch.manual_seed(252627)
        batch_size, seq_len, vocab_size = 1, 16, 50000  # Reduced for test speed
        chunk_size = 8

        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # This should not OOM even with large vocab
        loss_chunk, mask = compute_per_token_cross_entropy(
            logits,
            labels,
            shift_labels=True,
            chunk_size=chunk_size,
        )

        assert loss_chunk.shape[0] == batch_size * (seq_len - 1)  # -1 from shift
        assert mask.sum() > 0  # Some valid tokens

    def test_with_ignore_index_and_chunking(self):
        """Test chunking with ignore_index in realistic scenario."""
        torch.manual_seed(282930)
        batch_size, seq_len, vocab_size = 4, 32, 1000
        chunk_size = 16
        ignore_index = -100

        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Add some padding (ignore_index)
        labels[:, -5:] = ignore_index  # Last 5 tokens are padding

        loss_chunk, mask = compute_per_token_cross_entropy(
            logits,
            labels,
            shift_labels=True,
            chunk_size=chunk_size,
            ignore_index=ignore_index,
        )

        # Check that padding tokens are masked
        assert mask.sum() < batch_size * (seq_len - 1)  # Some tokens masked

        # Check that loss for masked tokens is well-behaved
        assert torch.all(torch.isfinite(loss_chunk))
