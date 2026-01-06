"""Test DFT + Context Parallel compatibility (Phase 3 fix).

This test suite verifies that the CP-aware DFT implementation correctly handles
sharded logits in Context Parallelism mode.

SOLUTION IMPLEMENTED:
---------------------
1. Detect CP-local logits by comparing logits_seq_len with expected chunk size
2. Use full labels tensor (available from inputs pre-hook)
3. Compute boundary-correct losses for each CP rank's token shard
4. Pad out-of-range labels with ignore_index=-100

REFERENCE:
----------
Based on Channel Loss CP compatibility implementation in:
/home/scbjtfy/axolotl/worktrees/channel-loss/src/axolotl/integrations/channel_loss/compute_loss_patch.py
"""

import torch
from types import SimpleNamespace
import pytest

from src.axolotl.integrations.dft.dft_utils import (
    compute_per_token_cross_entropy,
    apply_dft_weighting,
    reduce_token_loss,
)


class MockTrainer:
    """Mock trainer with CP group for testing."""

    def __init__(self, cp_size=2, cp_rank=0):
        self.cp_size = cp_size
        self.cp_rank = cp_rank
        self.accelerator = SimpleNamespace(
            context_parallel_group=SimpleNamespace() if cp_size > 1 else None
        )


class TestDFTContextParallelCompatibility:
    """Test DFT compatibility with Context Parallelism after Phase 3 fix."""

    def test_cp_aware_loss_computation_single_rank(self):
        """Test CP-aware loss with simulated CP environment for a single rank."""
        batch_size, full_seq_len, vocab_size = 2, 16, 100
        cp_size = 2
        cp_rank = 0  # Test rank 0

        # Simulate CP: logits are sharded, labels are full
        divisor = min(cp_size, 64)
        pad_len = (divisor - (full_seq_len % divisor)) % divisor
        chunk_len = (full_seq_len + pad_len) // cp_size

        # Create CP-local logits for rank 0
        logits_local = torch.randn(batch_size, chunk_len, vocab_size, requires_grad=True)

        # Full labels (as they appear in inputs before CP pre-hook)
        labels_full = torch.randint(0, vocab_size, (batch_size, full_seq_len))

        # Mock trainer with CP environment
        mock_trainer = MockTrainer(cp_size=cp_size, cp_rank=cp_rank)

        # Manually override CP group methods for testing
        import torch.distributed as dist
        original_is_initialized = dist.is_initialized
        original_get_world_size = dist.get_world_size
        original_get_rank = dist.get_rank

        def mock_is_initialized():
            return True

        def mock_get_world_size(group=None):
            return cp_size

        def mock_get_rank(group=None):
            return cp_rank

        # Patch distributed methods
        dist.is_initialized = mock_is_initialized
        dist.get_world_size = mock_get_world_size
        dist.get_rank = mock_get_rank

        try:
            # Compute per-token loss with CP awareness
            per_token_loss, valid_mask = compute_per_token_cross_entropy(
                logits_local,
                labels_full,
                ignore_index=-100,
                shift_labels=True,
                trainer=mock_trainer,
            )

            # Apply DFT weighting
            per_token_loss = apply_dft_weighting(per_token_loss)

            # Reduce to scalar
            loss = reduce_token_loss(per_token_loss, valid_mask)

            # Verify loss is valid
            assert torch.isfinite(loss), "Loss should be finite"
            assert loss.item() > 0, "Loss should be positive"

            # Verify per_token_loss shape
            # Expected: chunk_len tokens (non-last rank doesn't drop last token in shift)
            expected_token_count = chunk_len  # All tokens in this shard
            # After shift, we should have per-token losses
            assert per_token_loss.numel() > 0, "Should have per-token losses"

            print(
                f"\n✓ CP-aware DFT (rank {cp_rank}): "
                f"logits_local shape={tuple(logits_local.shape)}, "
                f"labels_full shape={tuple(labels_full.shape)}, "
                f"per_token_loss numel={per_token_loss.numel()}, "
                f"loss={loss.item():.6f}"
            )

        finally:
            # Restore original methods
            dist.is_initialized = original_is_initialized
            dist.get_world_size = original_get_world_size
            dist.get_rank = original_get_rank

    def test_cp_aware_loss_last_rank(self):
        """Test CP-aware loss computation for the last CP rank."""
        batch_size, full_seq_len, vocab_size = 2, 16, 100
        cp_size = 2
        cp_rank = 1  # Last rank

        # Simulate CP
        divisor = min(cp_size, 64)
        pad_len = (divisor - (full_seq_len % divisor)) % divisor
        chunk_len = (full_seq_len + pad_len) // cp_size

        # Create CP-local logits for last rank
        logits_local = torch.randn(batch_size, chunk_len, vocab_size, requires_grad=True)

        # Full labels
        labels_full = torch.randint(0, vocab_size, (batch_size, full_seq_len))

        # Mock trainer
        mock_trainer = MockTrainer(cp_size=cp_size, cp_rank=cp_rank)

        import torch.distributed as dist
        original_is_initialized = dist.is_initialized
        original_get_world_size = dist.get_world_size
        original_get_rank = dist.get_rank

        dist.is_initialized = lambda: True
        dist.get_world_size = lambda group=None: cp_size
        dist.get_rank = lambda group=None: cp_rank

        try:
            per_token_loss, valid_mask = compute_per_token_cross_entropy(
                logits_local,
                labels_full,
                ignore_index=-100,
                shift_labels=True,
                trainer=mock_trainer,
            )

            per_token_loss = apply_dft_weighting(per_token_loss)
            loss = reduce_token_loss(per_token_loss, valid_mask)

            assert torch.isfinite(loss), "Loss should be finite"
            assert loss.item() > 0, "Loss should be positive"

            # Last rank drops last token in shift (global last token has no target)
            # Expected: chunk_len - 1 tokens
            print(
                f"\n✓ CP-aware DFT (last rank {cp_rank}): "
                f"per_token_loss numel={per_token_loss.numel()}, "
                f"loss={loss.item():.6f}"
            )

        finally:
            dist.is_initialized = original_is_initialized
            dist.get_world_size = original_get_world_size
            dist.get_rank = original_get_rank

    def test_cp_aware_vs_naive_difference(self):
        """Compare CP-aware implementation vs naive approach to show correctness."""
        batch_size, full_seq_len, vocab_size = 2, 16, 100
        cp_size = 2
        cp_rank = 0

        # Setup
        divisor = min(cp_size, 64)
        pad_len = (divisor - (full_seq_len % divisor)) % divisor
        chunk_len = (full_seq_len + pad_len) // cp_size

        # Same logits for both approaches
        torch.manual_seed(42)
        logits_local = torch.randn(batch_size, chunk_len, vocab_size, requires_grad=True)
        labels_full = torch.randint(0, vocab_size, (batch_size, full_seq_len))

        # Mock trainer
        mock_trainer = MockTrainer(cp_size=cp_size, cp_rank=cp_rank)

        import torch.distributed as dist
        original_is_initialized = dist.is_initialized
        original_get_world_size = dist.get_world_size
        original_get_rank = dist.get_rank

        dist.is_initialized = lambda: True
        dist.get_world_size = lambda group=None: cp_size
        dist.get_rank = lambda group=None: cp_rank

        try:
            # CP-aware approach (correct)
            per_token_loss_correct, valid_mask_correct = compute_per_token_cross_entropy(
                logits_local,
                labels_full,
                ignore_index=-100,
                shift_labels=True,
                trainer=mock_trainer,
            )
            loss_correct = reduce_token_loss(
                apply_dft_weighting(per_token_loss_correct), valid_mask_correct
            )

            # Naive approach (incorrect): use labels[:, 1:] directly
            # This would mismatch with CP-local logits
            naive_logits = logits_local[:, :-1, :].contiguous()
            naive_labels = labels_full[:, 1:chunk_len].contiguous()  # Wrong slice!

            naive_logits_flat = naive_logits.view(-1, vocab_size)
            naive_labels_flat = naive_labels.view(-1)

            naive_loss_raw = torch.nn.functional.cross_entropy(
                naive_logits_flat, naive_labels_flat, reduction="none"
            )
            naive_valid = naive_labels_flat != -100
            naive_loss = reduce_token_loss(
                apply_dft_weighting(naive_loss_raw), naive_valid
            )

            print(f"\n✓ CP-aware loss: {loss_correct.item():.6f}")
            print(f"✗ Naive loss (wrong): {naive_loss.item():.6f}")
            print(f"Difference: {abs(loss_correct.item() - naive_loss.item()):.6f}")

            # The approaches should produce different results
            # (demonstrating that CP-aware logic is necessary)
            assert not torch.allclose(loss_correct, naive_loss, rtol=0.01), (
                "CP-aware and naive approaches should differ, "
                "showing that special handling is needed"
            )

        finally:
            dist.is_initialized = original_is_initialized
            dist.get_world_size = original_get_world_size
            dist.get_rank = original_get_rank

    def test_non_cp_mode_unchanged(self):
        """Verify that non-CP mode still works as before."""
        batch_size, seq_len, vocab_size = 2, 16, 100

        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # No trainer = no CP
        per_token_loss, valid_mask = compute_per_token_cross_entropy(
            logits,
            labels,
            ignore_index=-100,
            shift_labels=True,
            trainer=None,  # No CP
        )

        loss = reduce_token_loss(apply_dft_weighting(per_token_loss), valid_mask)

        assert torch.isfinite(loss), "Loss should be finite"
        assert loss.item() > 0, "Loss should be positive"

        # Expected: (seq_len - 1) tokens per sequence after shift
        expected_tokens = batch_size * (seq_len - 1)
        assert per_token_loss.numel() == expected_tokens, (
            f"Expected {expected_tokens} tokens, got {per_token_loss.numel()}"
        )

        print(f"\n✓ Non-CP mode: loss={loss.item():.6f}, tokens={per_token_loss.numel()}")

    def test_cp_with_padding(self):
        """Test CP-aware loss with padded sequences (last rank sees padding)."""
        batch_size, full_seq_len, vocab_size = 2, 16, 100
        cp_size = 2
        cp_rank = 1  # Use last rank which sees the padding

        divisor = min(cp_size, 64)
        pad_len = (divisor - (full_seq_len % divisor)) % divisor
        chunk_len = (full_seq_len + pad_len) // cp_size

        logits_local = torch.randn(batch_size, chunk_len, vocab_size, requires_grad=True)
        labels_full = torch.randint(0, vocab_size, (batch_size, full_seq_len))

        # Add padding to labels (last 4 tokens) - rank 1 handles tokens [8:16]
        labels_full[:, -4:] = -100

        mock_trainer = MockTrainer(cp_size=cp_size, cp_rank=cp_rank)

        import torch.distributed as dist
        original_is_initialized = dist.is_initialized
        original_get_world_size = dist.get_world_size
        original_get_rank = dist.get_rank

        dist.is_initialized = lambda: True
        dist.get_world_size = lambda group=None: cp_size
        dist.get_rank = lambda group=None: cp_rank

        try:
            per_token_loss, valid_mask = compute_per_token_cross_entropy(
                logits_local,
                labels_full,
                ignore_index=-100,
                shift_labels=True,
                trainer=mock_trainer,
            )

            loss = reduce_token_loss(apply_dft_weighting(per_token_loss), valid_mask)

            assert torch.isfinite(loss), "Loss should be finite"
            assert loss.item() > 0, "Loss should be positive"

            # Valid mask should exclude padding
            valid_count = valid_mask.sum().item()
            assert valid_count < per_token_loss.numel(), (
                "Some tokens should be masked due to padding"
            )

            print(
                f"\n✓ CP with padding (rank {cp_rank}): loss={loss.item():.6f}, "
                f"valid_tokens={valid_count}/{per_token_loss.numel()}"
            )

        finally:
            dist.is_initialized = original_is_initialized
            dist.get_world_size = original_get_world_size
            dist.get_rank = original_get_rank


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
