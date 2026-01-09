"""Test to demonstrate DFT + Context Parallel incompatibility.

This test documents the incompatibility between DFT loss and Context Parallelism
in standard SFT training mode (non-GRPO).

PROBLEM:
--------
1. In CP mode (non-GRPO), gather_outputs=False (see train.py:203)
2. Model forward returns SHARDED logits (shape: [batch, seq/cp_size, vocab])
3. DFT patch computes CE loss on SHARDED logits -> INCORRECT LOSS
4. Each rank computes loss only on its sequence shard

IMPACT:
-------
- DFT + CP (SFT mode) produces incorrect training signal
- Loss values are wrong
- Model does not train correctly

WORKAROUND:
-----------
- Use DFT without CP, or
- Use CP without DFT, or
- Use GRPO mode (where gather_outputs=True)

FIX REQUIRED:
-------------
Phase 3 of spec (DFT-aware CP optimization) addresses this by:
- Forcing gather of logits before DFT loss computation, or
- Implementing ms-swift's approach: compute per-rank DFT loss, then gather

NOTE: This test is marked as EXPECTED TO DEMONSTRATE INCOMPATIBILITY.
"""

import pytest
import torch


class TestDFTContextParallelIncompatibility:
    """Demonstrate incompatibility between DFT and Context Parallel."""

    def test_dft_receives_sharded_logits_in_cp_mode(self):
        """
        Demonstrate that DFT patch receives sharded logits when CP is enabled.

        This is a DOCUMENTATION test showing the incompatibility, not a passing test.
        """
        # Skip if torch.distributed not available
        pytest.skip(
            "Documentation test: DFT + CP (SFT mode) is INCOMPATIBLE. "
            "Logits are sharded but DFT assumes full logits. "
            "See spec Phase 3 for fix."
        )

    def test_sharded_ce_loss_is_incorrect(self):
        """
        Show that computing CE loss on sharded logits gives wrong results.

        This demonstrates why DFT + CP is broken.
        """
        batch_size, seq_len, vocab_size = 2, 8, 100
        cp_size = 2  # Context parallel size

        # Full sequence logits and labels
        full_logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Compute correct CE loss on full logits
        shift_logits = full_logits[:, :-1, :].contiguous().view(-1, vocab_size)
        shift_labels = labels[:, 1:].contiguous().view(-1)
        correct_loss = torch.nn.functional.cross_entropy(
            shift_logits, shift_labels, reduction="mean"
        )

        # Simulate CP: shard logits and labels
        shard_size = seq_len // cp_size
        logits_shard_rank0 = full_logits[:, :shard_size, :]  # First half
        logits_shard_rank1 = full_logits[:, shard_size:, :]  # Second half
        labels_shard_rank0 = labels[:, :shard_size]
        labels_shard_rank1 = labels[:, shard_size:]

        # Compute CE loss on each shard (what DFT patch does in CP mode)
        # Rank 0
        shift_logits_r0 = (
            logits_shard_rank0[:, :-1, :].contiguous().view(-1, vocab_size)
        )
        shift_labels_r0 = labels_shard_rank0[:, 1:].contiguous().view(-1)
        loss_rank0 = torch.nn.functional.cross_entropy(
            shift_logits_r0, shift_labels_r0, reduction="mean"
        )

        # Rank 1
        shift_logits_r1 = (
            logits_shard_rank1[:, :-1, :].contiguous().view(-1, vocab_size)
        )
        shift_labels_r1 = labels_shard_rank1[:, 1:].contiguous().view(-1)
        loss_rank1 = torch.nn.functional.cross_entropy(
            shift_logits_r1, shift_labels_r1, reduction="mean"
        )

        # Average losses from both ranks (what happens in distributed training)
        sharded_loss = (loss_rank0 + loss_rank1) / 2.0

        # The losses should be different!
        # This shows why DFT + CP is broken
        print(f"\nCorrect loss (full logits): {correct_loss.item():.6f}")
        print(f"Sharded loss (DFT + CP): {sharded_loss.item():.6f}")
        print(f"Difference: {abs(correct_loss.item() - sharded_loss.item()):.6f}")

        # This test documents the bug - losses ARE different
        # NOTE: In practice they might be close but not identical
        assert not torch.allclose(correct_loss, sharded_loss, atol=1e-5), (
            "Expected sharded loss to differ from correct loss, "
            "demonstrating DFT + CP incompatibility"
        )

    def test_solution_approach_gather_before_dft(self):
        """
        Show the correct approach: gather logits before computing DFT loss.

        This is what Phase 3 implementation should do.
        """
        batch_size, seq_len, vocab_size = 2, 8, 100
        cp_size = 2

        # Simulate sharded logits from CP
        shard_size = seq_len // cp_size
        logits_shard_rank0 = torch.randn(batch_size, shard_size, vocab_size)
        logits_shard_rank1 = torch.randn(batch_size, shard_size, vocab_size)

        # SOLUTION: Gather logits across ranks before DFT
        full_logits = torch.cat([logits_shard_rank0, logits_shard_rank1], dim=1)

        # Now DFT can compute on full logits
        assert full_logits.shape == (batch_size, seq_len, vocab_size)

        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # DFT loss computation on full logits (correct)
        shift_logits = full_logits[:, :-1, :].contiguous().view(-1, vocab_size)
        shift_labels = labels[:, 1:].contiguous().view(-1)
        loss = torch.nn.functional.cross_entropy(
            shift_logits, shift_labels, reduction="mean"
        )

        # This works correctly
        assert loss.item() > 0
        print(f"\n✓ Correct approach: gather then compute DFT loss = {loss.item():.6f}")
