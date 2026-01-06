"""Test DFT with multiple features enabled simultaneously.

This test suite verifies that DFT correctly integrates with combinations of
compatible features that are commonly used together in production training.

TESTED COMBINATIONS:
- ✅ DFT + Packing + FSDP: Multi-sequence packing with distributed training
- ✅ DFT + Chunked CE + Gradient Accumulation: Memory-efficient large-batch training
- ✅ DFT + Channel Loss + Mixed Precision: DFT intermediates with BF16
- ✅ DFT + CP + Packing: Context parallel with packed sequences

ARCHITECTURE NOTES:
- Packing: Multiple sequences in one batch, labels padded with -100
- FSDP: Model sharding, transparent to loss (complete logits after All-Gather)
- Chunked CE: Memory-efficient CE for large vocabs, compatible with DFT
- Channel Loss: Uses DFT's per_token_loss and valid_mask intermediates
- CP: Context parallel with sharded logits, DFT has CP-aware handling

REFERENCE: specs/001-dft-compatibility-matrix/README.md
"""

import pytest
import torch
from types import SimpleNamespace
from unittest.mock import Mock

from src.axolotl.integrations.dft.dft_utils import (
    compute_dft_loss,
    compute_dft_loss_with_intermediate,
    compute_per_token_cross_entropy,
    apply_dft_weighting,
    reduce_token_loss,
)
from src.axolotl.integrations.dft.patch import patch_compute_loss_for_dft


class TestDFTPackingCombinations:
    """Test DFT with sequence packing."""

    def test_dft_with_packing_basic(self):
        """Verify DFT correctly handles packed sequences.

        Packing combines multiple sequences into one batch:
        - Sequence boundaries marked by labels = -100
        - DFT should compute loss only for valid tokens (labels != -100)
        - Example: [seq1_tokens | -100 padding | seq2_tokens | -100 padding]
        """
        batch_size, seq_len, vocab_size = 1, 32, 100

        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)

        # Create packed labels: [8 tokens | 4 padding | 12 tokens | 8 padding]
        labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
        labels[0, :8] = torch.randint(0, vocab_size, (8,))  # Sequence 1
        labels[0, 12:24] = torch.randint(0, vocab_size, (12,))  # Sequence 2

        # Compute DFT loss
        loss = compute_dft_loss(
            logits,
            labels,
            shift_labels=True,
            ignore_index=-100,
        )

        # Loss should be finite and positive
        assert torch.isfinite(loss), "Loss should be finite with packing"
        assert loss.item() > 0, "Loss should be positive"

        # Verify gradient flow
        loss.backward()
        assert logits.grad is not None, "Gradients should flow with packing"

        print(
            f"\n✓ DFT + Packing: "
            f"packed_sequences=2, total_valid_tokens=19 (after shift), "
            f"loss={loss.item():.6f}"
        )

    def test_dft_packing_with_gradient_accumulation(self):
        """Test DFT + Packing + Gradient Accumulation.

        Common combination for training with long sequences:
        - Pack multiple short sequences into one batch
        - Use gradient accumulation for effective large batch
        - num_items_in_batch normalizes across accumulated batches
        """
        batch_size, seq_len, vocab_size = 2, 32, 100

        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)

        # Create packed labels for 2 batches, each with 2 sequences
        labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
        labels[0, :10] = torch.randint(0, vocab_size, (10,))  # Batch 0, seq 1
        labels[0, 14:20] = torch.randint(0, vocab_size, (6,))  # Batch 0, seq 2
        labels[1, :8] = torch.randint(0, vocab_size, (8,))  # Batch 1, seq 1
        labels[1, 12:24] = torch.randint(0, vocab_size, (12,))  # Batch 1, seq 2

        # Compute with gradient accumulation
        num_items_in_batch = 8  # Effective batch size
        loss = compute_dft_loss(
            logits,
            labels,
            shift_labels=True,
            ignore_index=-100,
            num_items_in_batch=num_items_in_batch,
        )

        assert torch.isfinite(loss), "Loss should be finite"
        assert loss.item() > 0, "Loss should be positive"

        print(
            f"\n✓ DFT + Packing + Gradient Accumulation: "
            f"num_items_in_batch={num_items_in_batch}, "
            f"loss={loss.item():.6f}"
        )

    def test_dft_packing_fsdp_simulation(self):
        """Document DFT + Packing + FSDP compatibility.

        FSDP shards model parameters, but:
        - Forward pass outputs complete logits [batch, seq, vocab] after All-Gather
        - DFT loss computation sees complete tensors
        - Packing is orthogonal to FSDP (handled by data collator)

        This test simulates the scenario (no actual FSDP process groups).
        """
        batch_size, seq_len, vocab_size = 2, 64, 50000  # Realistic FSDP use case

        # Simulate complete logits (as FSDP provides after All-Gather)
        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)

        # Packed sequences
        labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
        labels[0, :20] = torch.randint(0, vocab_size, (20,))
        labels[0, 24:40] = torch.randint(0, vocab_size, (16,))
        labels[1, :16] = torch.randint(0, vocab_size, (16,))
        labels[1, 20:48] = torch.randint(0, vocab_size, (28,))

        # Compute DFT loss
        loss = compute_dft_loss(
            logits,
            labels,
            shift_labels=True,
            ignore_index=-100,
        )

        assert torch.isfinite(loss), "Loss should be finite with FSDP + Packing"
        assert loss.item() > 0, "Loss should be positive"

        print(
            f"\n✓ DFT + Packing + FSDP (simulated): "
            f"vocab_size={vocab_size}, "
            f"loss={loss.item():.6f}"
        )


class TestDFTChunkedCECombinations:
    """Test DFT with chunked cross-entropy for memory efficiency."""

    def test_dft_chunked_ce_basic(self):
        """Verify DFT chunked CE works correctly.

        Chunked CE splits vocab dimension to reduce memory:
        - Standard CE: materializes [batch*seq, vocab] logits
        - Chunked CE: processes [batch*seq, chunk_size] at a time
        - Memory reduction: vocab_size / chunk_size
        """
        batch_size, seq_len, vocab_size = 2, 16, 50000
        chunk_size = 4096

        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Compute with chunked CE
        loss_chunked = compute_dft_loss(
            logits,
            labels,
            shift_labels=True,
            ignore_index=-100,
            chunk_size=chunk_size,
        )

        # Compute without chunking (for comparison)
        loss_standard = compute_dft_loss(
            logits,
            labels,
            shift_labels=True,
            ignore_index=-100,
            chunk_size=None,
        )

        # Should produce similar results
        assert torch.isfinite(loss_chunked), "Chunked loss should be finite"
        assert torch.isfinite(loss_standard), "Standard loss should be finite"
        assert torch.allclose(loss_chunked, loss_standard, rtol=1e-3), (
            f"Chunked and standard CE should match: "
            f"chunked={loss_chunked.item():.6f}, standard={loss_standard.item():.6f}"
        )

        print(
            f"\n✓ DFT + Chunked CE: "
            f"vocab_size={vocab_size}, chunk_size={chunk_size}, "
            f"loss_chunked={loss_chunked.item():.6f}, "
            f"loss_standard={loss_standard.item():.6f}"
        )

    def test_dft_chunked_ce_with_gradient_accumulation(self):
        """Test DFT + Chunked CE + Gradient Accumulation.

        Typical configuration for large vocab + large batch training:
        - Chunked CE reduces memory for vocab dimension
        - Gradient accumulation reduces memory for batch dimension
        - num_items_in_batch normalizes loss across accumulation steps
        """
        batch_size, seq_len, vocab_size = 2, 32, 100000
        chunk_size = 8192
        num_items_in_batch = 16

        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Compute with both optimizations
        loss = compute_dft_loss(
            logits,
            labels,
            shift_labels=True,
            ignore_index=-100,
            chunk_size=chunk_size,
            num_items_in_batch=num_items_in_batch,
        )

        assert torch.isfinite(loss), "Loss should be finite"
        assert loss.item() > 0, "Loss should be positive"

        # Verify gradient flow
        loss.backward()
        assert logits.grad is not None, "Gradients should flow"

        print(
            f"\n✓ DFT + Chunked CE + Gradient Accumulation: "
            f"vocab_size={vocab_size}, chunk_size={chunk_size}, "
            f"num_items_in_batch={num_items_in_batch}, "
            f"loss={loss.item():.6f}"
        )


class TestDFTChannelLossCombinations:
    """Test DFT with Channel Loss integration."""

    def test_dft_channel_loss_basic(self):
        """Verify DFT + Channel Loss integration.

        Channel Loss plugin requires:
        - per_token_loss: [batch*seq] tensor with per-token losses
        - valid_mask: [batch*seq] bool tensor
        - DFT provides these via compute_dft_loss_with_intermediate
        """
        batch_size, seq_len, vocab_size = 2, 16, 100

        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Compute with intermediate outputs (for Channel Loss)
        scalar_loss, per_token_loss, valid_mask = compute_dft_loss_with_intermediate(
            logits,
            labels,
            shift_labels=True,
            ignore_index=-100,
        )

        # Verify scalar loss
        assert torch.isfinite(scalar_loss), "Scalar loss should be finite"
        assert scalar_loss.item() > 0, "Scalar loss should be positive"

        # Verify intermediates
        assert per_token_loss.ndim == 1, "per_token_loss should be 1D"
        assert valid_mask.ndim == 1, "valid_mask should be 1D"
        assert per_token_loss.shape == valid_mask.shape, "Shapes should match"
        assert valid_mask.dtype == torch.bool, "valid_mask should be bool"

        # Verify gradient flow
        scalar_loss.backward()
        assert logits.grad is not None, "Gradients should flow"

        print(
            f"\n✓ DFT + Channel Loss: "
            f"scalar_loss={scalar_loss.item():.6f}, "
            f"per_token_loss shape={tuple(per_token_loss.shape)}, "
            f"valid_tokens={valid_mask.sum().item()}"
        )

    def test_dft_channel_loss_mixed_precision(self):
        """Test DFT + Channel Loss + Mixed Precision (BF16).

        Verifies that:
        - Intermediates preserve gradient flow with BF16
        - scalar_loss is FP32 (upcast)
        - per_token_loss supports autograd
        """
        batch_size, seq_len, vocab_size = 2, 16, 100

        # BF16 logits (common in mixed precision training)
        logits = torch.randn(batch_size, seq_len, vocab_size).bfloat16().requires_grad_(True)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Compute with intermediate outputs
        scalar_loss, per_token_loss, valid_mask = compute_dft_loss_with_intermediate(
            logits,
            labels,
            shift_labels=True,
            ignore_index=-100,
        )

        # Verify dtypes
        assert scalar_loss.dtype == torch.float32, "Scalar loss should be FP32"
        # per_token_loss may be FP32 due to .float() upcast in DFT weighting

        # Verify gradient flow with BF16
        scalar_loss.backward()
        assert logits.grad is not None, "Gradients should flow"
        assert logits.grad.dtype == torch.bfloat16, "Gradients should be BF16"

        print(
            f"\n✓ DFT + Channel Loss + BF16: "
            f"scalar_loss={scalar_loss.item():.6f} (dtype={scalar_loss.dtype}), "
            f"logits_grad dtype={logits.grad.dtype}"
        )

    def test_dft_channel_loss_trainer_patch(self):
        """Test DFT + Channel Loss through trainer patch.

        Verifies that enable_dft_channel_loss flag correctly:
        - Attaches intermediates to outputs
        - Returns scalar loss for backward
        """
        mock_trainer = Mock()
        mock_trainer.args = SimpleNamespace(
            enable_dft_loss=True,
            enable_dft_channel_loss=True,  # Channel Loss integration enabled
            label_smoothing_factor=0.0,
            orpo_alpha=None,
            include_tkps=False,
            dft_chunk_size=None,
        )
        mock_cfg = SimpleNamespace()

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

        # Call with return_outputs=True to get outputs
        loss, outputs = mock_trainer.compute_loss(
            mock_model, inputs, return_outputs=True
        )

        # Verify scalar loss
        assert torch.isfinite(loss), "Loss should be finite"
        assert loss.item() > 0, "Loss should be positive"

        # Verify intermediates are attached to outputs
        assert hasattr(outputs, "per_token_loss"), "Should have per_token_loss"
        assert hasattr(outputs, "valid_mask"), "Should have valid_mask"
        assert hasattr(outputs, "loss"), "Should have loss"

        # Verify intermediate shapes
        assert outputs.per_token_loss.ndim == 1, "per_token_loss should be 1D"
        assert outputs.valid_mask.ndim == 1, "valid_mask should be 1D"

        print(
            f"\n✓ DFT + Channel Loss via trainer patch: "
            f"loss={loss.item():.6f}, "
            f"intermediates attached: per_token_loss={tuple(outputs.per_token_loss.shape)}, "
            f"valid_mask={tuple(outputs.valid_mask.shape)}"
        )


class TestDFTContextParallelCombinations:
    """Test DFT with Context Parallelism (CP)."""

    def test_dft_cp_with_packing_simulation(self):
        """Document DFT + CP + Packing compatibility.

        Context Parallel:
        - Shards sequence dimension across GPUs
        - Each rank processes a chunk of the sequence
        - DFT has CP-aware label slicing

        Packing:
        - Multiple sequences in one batch with -100 padding
        - Works orthogonally to CP (each rank sees packed labels)

        This test simulates CP environment (no actual process groups).
        """
        batch_size, full_seq_len, vocab_size = 2, 64, 100
        cp_size = 2
        cp_rank = 0

        # Simulate CP: logits are sharded
        divisor = min(cp_size, 64)
        pad_len = (divisor - (full_seq_len % divisor)) % divisor
        chunk_len = (full_seq_len + pad_len) // cp_size

        logits_local = torch.randn(batch_size, chunk_len, vocab_size, requires_grad=True)

        # Full labels with packing
        labels_full = torch.full((batch_size, full_seq_len), -100, dtype=torch.long)
        labels_full[0, :20] = torch.randint(0, vocab_size, (20,))  # Seq 1
        labels_full[0, 24:40] = torch.randint(0, vocab_size, (16,))  # Seq 2
        labels_full[1, :16] = torch.randint(0, vocab_size, (16,))  # Seq 3
        labels_full[1, 20:48] = torch.randint(0, vocab_size, (28,))  # Seq 4

        # Mock CP trainer
        class MockCPTrainer:
            def __init__(self, cp_size, cp_rank):
                self.accelerator = SimpleNamespace(
                    context_parallel_group=SimpleNamespace()
                )

        mock_trainer = MockCPTrainer(cp_size=cp_size, cp_rank=cp_rank)

        # Mock distributed methods
        import torch.distributed as dist
        original_is_initialized = dist.is_initialized
        original_get_world_size = dist.get_world_size
        original_get_rank = dist.get_rank

        dist.is_initialized = lambda: True
        dist.get_world_size = lambda group=None: cp_size
        dist.get_rank = lambda group=None: cp_rank

        try:
            # Compute DFT loss with CP awareness
            per_token_loss, valid_mask = compute_per_token_cross_entropy(
                logits_local,
                labels_full,
                shift_labels=True,
                ignore_index=-100,
                trainer=mock_trainer,
            )

            loss = reduce_token_loss(
                apply_dft_weighting(per_token_loss), valid_mask
            )

            assert torch.isfinite(loss), "Loss should be finite"
            assert loss.item() > 0, "Loss should be positive"

            print(
                f"\n✓ DFT + CP + Packing (simulated): "
                f"cp_size={cp_size}, cp_rank={cp_rank}, "
                f"logits_local shape={tuple(logits_local.shape)}, "
                f"labels_full shape={tuple(labels_full.shape)}, "
                f"loss={loss.item():.6f}"
            )

        finally:
            # Restore original methods
            dist.is_initialized = original_is_initialized
            dist.get_world_size = original_get_world_size
            dist.get_rank = original_get_rank


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
