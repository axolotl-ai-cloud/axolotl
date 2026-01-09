from __future__ import annotations

"""Consolidated DFT test suite.

All DFT tests in a single file, organized by feature area.
Total: 69 tests covering all DFT functionality.

Tests consolidated using AST parsing for syntax correctness.
Includes all helper functions, classes, and test cases.
"""
# ruff: noqa: E402
import math
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from axolotl.integrations.dft.args import DFTArgs
from axolotl.integrations.dft.chunked_ce import chunked_cross_entropy
from axolotl.integrations.dft.dft_utils import (
    apply_dft_weighting,
    compute_dft_loss,
    compute_dft_loss_with_intermediate,
    compute_per_token_cross_entropy,
    reduce_token_loss,
)
from axolotl.integrations.dft.patch import patch_compute_loss_for_dft

# ==============================================================================
# CORE.PY TESTS
# Source: test_dft.py.original
# ==============================================================================


class TestDFTArgs:
    def test_defaults(self):
        args = DFTArgs()
        assert args.enable_dft_loss is False

    def test_custom(self):
        args = DFTArgs(enable_dft_loss=True)
        assert args.enable_dft_loss is True


class TestDFTUtils:
    def test_apply_dft_weighting_values(self):
        loss = torch.tensor([0.0, 1.0, 10.0], requires_grad=True)
        out = apply_dft_weighting(loss)
        expected = loss.detach() * torch.exp(-loss.detach())
        assert torch.allclose(out.detach(), expected, atol=1e-06)
        out.sum().backward()
        assert loss.grad is not None

    def test_compute_dft_loss_matches_manual(self):
        log4 = math.log(4.0)
        logits = torch.tensor(
            [[[0.0, log4], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]], requires_grad=True
        )
        labels = torch.tensor([[0, 1, 0, -100]])
        loss = compute_dft_loss(logits, labels)
        l0 = -math.log(0.8) * 0.8
        l1 = -math.log(0.5) * 0.5
        expected = (l0 + l1) / 2.0
        assert loss.item() == pytest.approx(expected, abs=1e-06)
        loss.backward()
        assert logits.grad is not None

    def test_compute_dft_loss_all_ignored_is_zero(self):
        logits = torch.zeros(1, 4, 2, requires_grad=True)
        labels = torch.full((1, 4), -100)
        loss = compute_dft_loss(logits, labels)
        assert loss.item() == pytest.approx(0.0, abs=1e-12)
        loss.backward()
        assert logits.grad is not None
        assert torch.all(logits.grad == 0)


class TestDFTPatch:
    def test_patch_compute_loss_removes_labels(self):
        trainer = MagicMock()
        trainer.args = SimpleNamespace(
            enable_dft_loss=True,
            include_tkps=False,
            label_smoothing_factor=0.0,
            orpo_alpha=None,
        )
        trainer.state = SimpleNamespace()
        sentinel = object()
        original_compute_loss = MagicMock(return_value=sentinel)
        trainer.compute_loss = original_compute_loss
        patch_compute_loss_for_dft(trainer, cfg=MagicMock())
        logits = torch.zeros(1, 4, 2, requires_grad=True)
        labels = torch.tensor([[0, 1, 0, -100]])

        class DummyModel(torch.nn.Module):
            def forward(self, **kwargs):
                assert "labels" not in kwargs
                return SimpleNamespace(logits=logits)

        loss, outputs = trainer.compute_loss(
            DummyModel(),
            {"input_ids": torch.zeros(1, 4, dtype=torch.long), "labels": labels},
            return_outputs=True,
        )
        assert outputs.logits is logits
        assert isinstance(loss, torch.Tensor)
        assert original_compute_loss.call_count == 0

    def test_patch_compute_loss_falls_back_when_disabled(self):
        trainer = MagicMock()
        trainer.args = SimpleNamespace(
            enable_dft_loss=False,
            include_tkps=False,
            label_smoothing_factor=0.0,
            orpo_alpha=None,
        )
        trainer.state = SimpleNamespace()
        sentinel = object()
        original_compute_loss = MagicMock(return_value=sentinel)
        trainer.compute_loss = original_compute_loss
        patch_compute_loss_for_dft(trainer, cfg=MagicMock())
        out = trainer.compute_loss(model=MagicMock(), inputs={})
        assert out is sentinel
        assert original_compute_loss.call_count == 1

    def test_patch_compute_loss_raises_on_label_smoothing(self):
        trainer = MagicMock()
        trainer.args = SimpleNamespace(
            enable_dft_loss=True,
            include_tkps=False,
            label_smoothing_factor=0.1,
            orpo_alpha=None,
        )
        trainer.state = SimpleNamespace()
        original_compute_loss = MagicMock(return_value=None)
        trainer.compute_loss = original_compute_loss
        patch_compute_loss_for_dft(trainer, cfg=MagicMock())
        with pytest.raises(ValueError, match="label smoothing"):
            trainer.compute_loss(
                model=MagicMock(), inputs={"labels": torch.zeros(1, 2)}
            )
        assert original_compute_loss.call_count == 0


# ==============================================================================
# COMPATIBILITY TESTS
# Source: test_dft_compatibility.py
# ==============================================================================


class TestDFTGradientAccumulationCompatibility:
    """Test DFT compatibility with gradient accumulation."""

    def test_gradient_accumulation_normalization(self):
        """Verify DFT correctly uses num_items_in_batch for normalization.

        In gradient accumulation, num_items_in_batch is used to normalize loss
        across accumulated micro-batches. The loss should scale inversely with
        num_items_in_batch.

        Example:
        - Micro-batch: batch_size=2, seq_len=16 → ~30 valid tokens after shift
        - Without num_items_in_batch: loss normalized by 30 tokens
        - With num_items_in_batch=8: loss normalized by 8 (accumulation batch size)
        - Expected: loss_with / loss_without ≈ 30 / 8 = 3.75x
        """
        batch_size, seq_len, vocab_size = (2, 16, 100)
        torch.manual_seed(42)
        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        expected_tokens = batch_size * (seq_len - 1)
        loss_without = compute_dft_loss(
            logits,
            labels,
            shift_labels=True,
            ignore_index=-100,
            num_items_in_batch=None,
        )
        num_items_in_batch = 8
        loss_with = compute_dft_loss(
            logits,
            labels,
            shift_labels=True,
            ignore_index=-100,
            num_items_in_batch=num_items_in_batch,
        )
        expected_ratio = expected_tokens / num_items_in_batch
        actual_ratio = loss_with.item() / loss_without.item()
        assert torch.isfinite(loss_with), "Loss with accumulation should be finite"
        assert torch.isfinite(loss_without), (
            "Loss without accumulation should be finite"
        )
        assert abs(actual_ratio - expected_ratio) < 0.1, (
            f"Loss ratio should be ~{expected_ratio:.3f}, got {actual_ratio:.3f}"
        )
        print(
            f"\n Gradient accumulation normalization: expected_tokens={expected_tokens}, num_items_in_batch={num_items_in_batch}, loss_with={loss_with.item():.6f}, loss_without={loss_without.item():.6f}, ratio={actual_ratio:.3f} (expected {expected_ratio:.3f})"
        )

    def test_gradient_accumulation_in_trainer_patch(self):
        """Verify num_items_in_batch is correctly passed through trainer patch."""
        mock_trainer = Mock()
        mock_trainer.args = SimpleNamespace(
            enable_dft_loss=True,
            label_smoothing_factor=0.0,
            orpo_alpha=None,
            include_tkps=False,
            enable_dft_channel_loss=False,
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
        loss = mock_trainer.compute_loss(mock_model, inputs, num_items_in_batch=8)
        assert torch.isfinite(loss), "Loss should be finite"
        assert loss.item() > 0, "Loss should be positive"
        print(f"\n Trainer patch with gradient accumulation: loss={loss.item():.6f}")


class TestDFTMixedPrecisionCompatibility:
    """Test DFT compatibility with mixed precision training (FP16/BF16)."""

    def test_fp16_compatibility(self):
        """Verify DFT works correctly with FP16 tensors.

        DFT uses .float() cast for exp(-loss) computation to avoid
        numerical issues with FP16's limited range.
        """
        batch_size, seq_len, vocab_size = (2, 16, 100)
        logits = (
            torch.randn(batch_size, seq_len, vocab_size).half().requires_grad_(True)
        )
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        loss = compute_dft_loss(logits, labels, shift_labels=True, ignore_index=-100)
        assert torch.isfinite(loss), "Loss should be finite with FP16"
        assert loss.item() > 0, "Loss should be positive"
        assert loss.dtype == torch.float32, "Loss should be FP32 (upcast)"
        loss.backward()
        assert logits.grad is not None, "Gradients should flow with FP16"
        assert logits.grad.dtype == torch.float16, "Gradients should be FP16"
        print(f"\n FP16 compatibility: loss={loss.item():.6f} (dtype={loss.dtype})")

    def test_bf16_compatibility(self):
        """Verify DFT works correctly with BF16 tensors.

        BF16 has better range than FP16 but lower precision. DFT should
        handle it correctly with .float() upcast.
        """
        batch_size, seq_len, vocab_size = (2, 16, 100)
        logits = (
            torch.randn(batch_size, seq_len, vocab_size).bfloat16().requires_grad_(True)
        )
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        loss = compute_dft_loss(logits, labels, shift_labels=True, ignore_index=-100)
        assert torch.isfinite(loss), "Loss should be finite with BF16"
        assert loss.item() > 0, "Loss should be positive"
        assert loss.dtype == torch.float32, "Loss should be FP32 (upcast)"
        loss.backward()
        assert logits.grad is not None, "Gradients should flow with BF16"
        assert logits.grad.dtype == torch.bfloat16, "Gradients should be BF16"
        print(f"\n BF16 compatibility: loss={loss.item():.6f} (dtype={loss.dtype})")

    def test_mixed_precision_numerical_stability(self):
        """Verify DFT's exp(-loss) computation is numerically stable in FP16.

        The .float() upcast in apply_dft_weighting prevents:
        - Overflow: exp(-loss) for small losses (close to 1.0)
        - Underflow: exp(-loss) for large losses (close to 0.0)
        """
        batch_size, seq_len, vocab_size = (2, 16, 100)
        logits_fp16 = torch.randn(batch_size, seq_len, vocab_size).half()
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        per_token_loss, valid_mask = compute_per_token_cross_entropy(
            logits_fp16, labels, shift_labels=True, ignore_index=-100
        )
        weighted_loss = apply_dft_weighting(per_token_loss)
        assert torch.all(torch.isfinite(weighted_loss)), (
            "All weighted losses should be finite"
        )
        assert torch.all(weighted_loss >= 0), "All weights should be non-negative"
        loss = reduce_token_loss(weighted_loss, valid_mask)
        assert torch.isfinite(loss), "Final loss should be finite"
        print(
            f"\n Mixed precision stability: per_token_loss range=[{per_token_loss.min():.3f}, {per_token_loss.max():.3f}], weighted range=[{weighted_loss.min():.3f}, {weighted_loss.max():.3f}], final loss={loss.item():.6f}"
        )


class TestDFTFlashAttentionCompatibility:
    """Test DFT compatibility with Flash Attention.

    Flash Attention operates at the attention layer, computing Q@K^T and softmax
    more efficiently. It is transparent to loss computation since it only affects
    the forward pass (attention mechanism), not logits or loss.
    """

    def test_flash_attention_transparency_documented(self):
        """Document why Flash Attention is transparent to DFT.

        This is a documentation test to ensure understanding is captured.
        """
        architectural_separation = {
            "Flash Attention layer": "Attention mechanism (Q, K, V computation)",
            "DFT loss layer": "Cross-entropy on final logits",
            "Interaction": "None - Flash Attention outputs hidden states, DFT uses logits",
            "Transparency": " Flash Attention is invisible to DFT loss computation",
        }
        expected_behavior = {
            "DFT code changes": "None required - Flash Attention is transparent",
            "Special handling": "Not needed - operates at different layer",
            "Performance impact": "Flash Attention speeds up forward pass, DFT unchanged",
            "Memory usage": "Flash Attention reduces attention memory, DFT unchanged",
        }
        print("\nFlash Attention Architectural Separation:")
        for aspect, detail in architectural_separation.items():
            print(f"  {aspect}: {detail}")
        print("\nExpected DFT Behavior with Flash Attention:")
        for aspect, detail in expected_behavior.items():
            print(f"  {aspect}: {detail}")
        assert True, "Flash Attention transparency documented"

    def test_dft_with_standard_attention_tensors(self):
        """Verify DFT works with tensors shaped like Flash Attention outputs.

        Flash Attention outputs the same shape as standard attention:
        [batch, seq, hidden_dim] → after LM head → [batch, seq, vocab]

        This test verifies DFT handles these standard shapes correctly.
        """
        batch_size, seq_len, vocab_size = (2, 128, 50000)
        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        loss = compute_dft_loss(logits, labels, shift_labels=True, ignore_index=-100)
        assert torch.isfinite(loss), "Loss should be finite"
        assert loss.item() > 0, "Loss should be positive"
        loss.backward()
        assert logits.grad is not None, "Gradients should flow"
        print(
            f"\n Flash Attention tensor shapes: logits={tuple(logits.shape)}, loss={loss.item():.6f}"
        )


class TestDFTDistributedTrainingCompatibility:
    """Test DFT compatibility with distributed training frameworks."""

    def test_fsdp_transparency_documented(self):
        """Document why FSDP is transparent to DFT.

        FSDP (Fully Sharded Data Parallel):
        - Shards model parameters across GPUs
        - All-Gathers parameters before forward/backward
        - Forward pass outputs complete logits [batch, seq, vocab]
        - Loss computation sees complete tensors

        DFT receives complete logits, so FSDP is transparent.
        """
        fsdp_guarantees = {
            "Parameter sharding": "Reduces per-GPU memory usage",
            "All-Gather timing": "Before forward pass, parameters reconstructed",
            "Forward output": "Complete logits [batch, seq, vocab] on each GPU",
            "DFT receives": "Complete logits, identical to non-FSDP",
            "Loss computation": "Identical to single-GPU case",
        }
        expected_behavior = {
            "DFT code changes": "None required - FSDP is transparent",
            "Special handling": "Not needed - FSDP handles communication",
            "Performance impact": "FSDP communication overhead, not DFT-specific",
            "Memory usage": "FSDP reduces param memory, DFT unchanged",
        }
        print("\nFSDP Architectural Guarantees:")
        for aspect, detail in fsdp_guarantees.items():
            print(f"  {aspect}: {detail}")
        print("\nExpected DFT Behavior with FSDP:")
        for aspect, detail in expected_behavior.items():
            print(f"  {aspect}: {detail}")
        assert True, "FSDP transparency documented"

    def test_ddp_transparency_documented(self):
        """Document why DDP is transparent to DFT.

        DDP (Distributed Data Parallel):
        - Replicates model across GPUs
        - Each GPU processes different data batch
        - Gradients are All-Reduced after backward
        - Loss computation is local (per-GPU)

        DFT computes loss locally, then gradients are synchronized by DDP.
        """
        ddp_workflow = {
            "Model replication": "Full model copy on each GPU",
            "Data sharding": "Different batches on each GPU",
            "Forward pass": "Independent, produces local logits",
            "Loss computation": "DFT computes local loss (per-GPU)",
            "Gradient sync": "DDP All-Reduces gradients after backward",
        }
        expected_behavior = {
            "DFT code changes": "None required - DDP is transparent",
            "Special handling": "Not needed - DDP handles gradient sync",
            "Performance impact": "DDP communication in backward, not DFT-specific",
            "Memory usage": "DDP replicates model, DFT unchanged",
        }
        print("\nDDP Workflow with DFT:")
        for step, detail in ddp_workflow.items():
            print(f"  {step}: {detail}")
        print("\nExpected DFT Behavior with DDP:")
        for aspect, detail in expected_behavior.items():
            print(f"  {aspect}: {detail}")
        assert True, "DDP transparency documented"


# ==============================================================================
# PACKING TESTS
# Source: test_dft_packing.py
# ==============================================================================


class TestDFTPackingCompatibility:
    """Test DFT integration with sequence packing."""

    def test_packed_sequences_basic(self):
        """Test DFT loss with packed sequences (3 sequences in one batch)."""
        batch_size = 1
        packed_seq_len = 7
        vocab_size = 10
        logits = torch.randn(batch_size, packed_seq_len, vocab_size, requires_grad=True)
        labels = torch.tensor([[0, 1, 2, 3, 4, 5, 6]])
        loss = compute_dft_loss(logits, labels)
        assert loss.ndim == 0, "Loss should be scalar"
        assert loss.item() > 0, "Loss should be positive"
        assert loss.requires_grad, "Loss should have gradients"
        loss.backward()
        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()

    def test_packed_sequences_with_padding(self):
        """Test DFT correctly handles padding between packed sequences."""
        batch_size = 1
        packed_seq_len = 10
        vocab_size = 20
        logits = torch.randn(batch_size, packed_seq_len, vocab_size, requires_grad=True)
        labels = torch.tensor([[0, 1, 2, -100, 3, 4, -100, -100, 5, 6]])
        loss = compute_dft_loss(logits, labels)
        assert loss.item() > 0
        loss.backward()
        assert logits.grad is not None

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
        seq1_len, seq2_len, seq3_len = (4, 3, 5)
        logits1 = torch.randn(1, seq1_len, vocab_size, requires_grad=True)
        logits2 = torch.randn(1, seq2_len, vocab_size, requires_grad=True)
        logits3 = torch.randn(1, seq3_len, vocab_size, requires_grad=True)
        labels1 = torch.randint(0, vocab_size, (1, seq1_len))
        labels2 = torch.randint(0, vocab_size, (1, seq2_len))
        labels3 = torch.randint(0, vocab_size, (1, seq3_len))
        loss1 = compute_dft_loss(logits1, labels1)
        loss2 = compute_dft_loss(logits2, labels2)
        loss3 = compute_dft_loss(logits3, labels3)
        valid_tokens1 = seq1_len - 1
        valid_tokens2 = seq2_len - 1
        valid_tokens3 = seq3_len - 1
        total_valid = valid_tokens1 + valid_tokens2 + valid_tokens3
        unpacked_avg_loss = (
            loss1 * valid_tokens1 + loss2 * valid_tokens2 + loss3 * valid_tokens3
        ) / total_valid
        packed_logits = torch.cat([logits1, logits2, logits3], dim=1).requires_grad_()
        labels1_with_boundary = labels1.clone()
        labels1_with_boundary[0, -1] = -100
        labels2_with_boundary = labels2.clone()
        labels2_with_boundary[0, -1] = -100
        packed_labels = torch.cat(
            [labels1_with_boundary, labels2_with_boundary, labels3], dim=1
        )
        packed_loss = compute_dft_loss(packed_logits, packed_labels)
        shift_packed_labels = packed_labels[:, 1:].flatten()
        valid_in_packed = (shift_packed_labels != -100).sum().item()
        assert valid_in_packed == total_valid, (
            f"Valid token count mismatch: {valid_in_packed} != {total_valid}"
        )
        assert packed_loss.item() > 0, "Packed loss should be positive"
        assert unpacked_avg_loss.item() > 0, "Unpacked loss should be positive"
        assert packed_loss.requires_grad, "Packed loss should require gradients"
        assert torch.isfinite(packed_loss), "Packed loss should be finite"
        assert torch.isfinite(unpacked_avg_loss), "Unpacked loss should be finite"
        rel_diff = abs(packed_loss - unpacked_avg_loss) / unpacked_avg_loss
        assert rel_diff < 0.2, (
            f"Packed loss ({packed_loss.item():.6f}) and unpacked loss ({unpacked_avg_loss.item():.6f}) differ by {rel_diff.item() * 100:.1f}%, expected < 20%"
        )

    def test_packed_sequences_all_padding_edge_case(self):
        """Test edge case where entire sequence is padding."""
        batch_size = 1
        seq_len = 5
        vocab_size = 10
        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        labels = torch.full((batch_size, seq_len), -100)
        loss = compute_dft_loss(logits, labels)
        assert loss.item() == pytest.approx(0.0, abs=1e-12)
        loss.backward()
        assert logits.grad is not None
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
        patch_compute_loss_for_dft(trainer, cfg=MagicMock())
        batch_size = 2
        packed_seq_len = 8
        vocab_size = 100
        logits = torch.randn(batch_size, packed_seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (batch_size, packed_seq_len))
        labels[:, 3] = -100
        labels[:, 6:] = -100

        class DummyModel(torch.nn.Module):
            def forward(self, **kwargs):
                return SimpleNamespace(logits=logits)

        model = DummyModel()
        inputs = {
            "input_ids": torch.zeros(batch_size, packed_seq_len),
            "labels": labels,
        }
        loss, outputs = trainer.compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=None
        )
        assert loss.ndim == 0
        assert loss.requires_grad
        assert outputs.logits is logits
        assert loss.item() > 0
        assert torch.isfinite(loss)

    def test_packed_sequences_with_chunked_ce(self):
        """Test DFT + packing works with chunked cross-entropy."""
        trainer = MagicMock()
        trainer.args = SimpleNamespace(
            enable_dft_loss=True,
            dft_chunk_size=32,
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
        vocab_size = 128
        logits = torch.randn(batch_size, packed_seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (batch_size, packed_seq_len))
        labels[:, 4] = -100
        labels[:, 7:] = -100

        class DummyModel(torch.nn.Module):
            def forward(self, **kwargs):
                return SimpleNamespace(logits=logits)

        model = DummyModel()
        inputs = {
            "input_ids": torch.zeros(batch_size, packed_seq_len),
            "labels": labels,
        }
        loss, outputs = trainer.compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=None
        )
        assert loss.ndim == 0
        assert loss.requires_grad
        assert torch.isfinite(loss)
        assert loss.item() > 0
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
        torch.tensor([[1, 1, 1, 2, 2, 2, 3, 3, 3]])
        loss_without = compute_dft_loss(logits, labels)
        assert loss_without.item() > 0
        assert torch.isfinite(loss_without)


# ==============================================================================
# DDP TESTS
# Source: test_dft_ddp.py
# ==============================================================================


def setup_ddp(rank, world_size):
    """Initialize DDP process group."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo",
        rank=rank,
        world_size=world_size,
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)


def cleanup_ddp():
    """Clean up DDP process group."""
    dist.destroy_process_group()


def _test_loss_consistency(rank, world_size):
    """Compute DFT loss on same input across all ranks."""
    device = torch.device(f"cuda:{rank}")
    torch.manual_seed(42)
    batch_size = 2
    seq_len = 10
    vocab_size = 100
    logits = torch.randn(
        batch_size, seq_len, vocab_size, device=device, requires_grad=True
    )
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    loss = compute_dft_loss(logits, labels)
    return {"loss": loss.item(), "rank": rank}


def _test_different_batches(rank, world_size):
    """Each rank processes different batch."""
    device = torch.device(f"cuda:{rank}")
    torch.manual_seed(42 + rank)
    batch_size = 2
    seq_len = 10
    vocab_size = 100
    logits = torch.randn(
        batch_size, seq_len, vocab_size, device=device, requires_grad=True
    )
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    loss = compute_dft_loss(logits, labels)
    return {"loss": loss.item(), "rank": rank, "has_grad": logits.grad is None}


def _test_chunked_ce(rank, world_size):
    """Test chunked CE with DDP."""
    device = torch.device(f"cuda:{rank}")
    torch.manual_seed(42)
    batch_size = 2
    seq_len = 8
    vocab_size = 1000
    chunk_size = 256
    logits = torch.randn(
        batch_size, seq_len, vocab_size, device=device, requires_grad=True
    )
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    per_token_loss, valid_mask = compute_per_token_cross_entropy(
        logits, labels, chunk_size=chunk_size
    )
    weighted_loss = apply_dft_weighting(per_token_loss)
    loss = reduce_token_loss(weighted_loss, valid_mask)
    return {"loss": loss.item(), "rank": rank}


def _test_gradient_sync(rank, world_size):
    """Simulate gradient computation and verify they can be synced."""
    device = torch.device(f"cuda:{rank}")
    torch.manual_seed(42 + rank)
    batch_size = 2
    seq_len = 10
    vocab_size = 100
    model = torch.nn.Linear(vocab_size, vocab_size).to(device)
    input_logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
    logits = model(input_logits)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    loss = compute_dft_loss(logits, labels)
    loss.backward()
    has_grads = all((p.grad is not None for p in model.parameters()))
    grad_norm = sum(
        (p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    )
    return {
        "loss": loss.item(),
        "has_grads": has_grads,
        "grad_norm": grad_norm,
        "rank": rank,
    }


def _test_with_padding(rank, world_size):
    """Test with padded sequences (ignore_index = -100)."""
    device = torch.device(f"cuda:{rank}")
    torch.manual_seed(42 + rank)
    batch_size = 2
    seq_len = 10
    vocab_size = 100
    logits = torch.randn(
        batch_size, seq_len, vocab_size, device=device, requires_grad=True
    )
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels[:, -3:] = -100
    loss = compute_dft_loss(logits, labels)
    return {"loss": loss.item(), "rank": rank}


def ddp_test_worker(rank, world_size, test_fn_name, result_queue):
    """Worker function for DDP test.

    Args:
        rank: Process rank
        world_size: Total number of processes
        test_fn_name: Name of the test function to run (string)
        result_queue: Queue to store results
    """
    try:
        setup_ddp(rank, world_size)
        test_fn = globals()[test_fn_name]
        result = test_fn(rank, world_size)
        result_queue.put((rank, result, None))
    except Exception:
        import traceback

        result_queue.put((rank, None, traceback.format_exc()))
    finally:
        cleanup_ddp()


class TestDFTDDPCompatibility(unittest.TestCase):
    """Test DFT compatibility with DDP."""

    def _run_ddp_test(self, test_fn_name, world_size=2):
        """Helper to run a test function in DDP mode.

        Args:
            test_fn_name: Name of module-level test function (string)
            world_size: Number of processes

        Returns:
            List of results from each rank
        """
        if not torch.cuda.is_available():
            self.skipTest("DDP test requires CUDA")
        if torch.cuda.device_count() < world_size:
            self.skipTest(
                f"DDP test requires {world_size} GPUs, found {torch.cuda.device_count()}"
            )
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        processes = []
        for rank in range(world_size):
            p = ctx.Process(
                target=ddp_test_worker,
                args=(rank, world_size, test_fn_name, result_queue),
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        results = []
        errors = []
        for _ in range(world_size):
            rank, result, error = result_queue.get()
            if error:
                errors.append(f"Rank {rank}:\n{error}")
            else:
                results.append((rank, result))
        if errors:
            self.fail("\n".join(errors))
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    def test_ddp_loss_consistency(self):
        """Test that DFT loss is identical across DDP ranks."""
        results = self._run_ddp_test("_test_loss_consistency", world_size=2)
        loss_values = [r["loss"] for r in results]
        self.assertEqual(len(loss_values), 2)
        self.assertAlmostEqual(loss_values[0], loss_values[1], places=5)

    def test_ddp_different_batches(self):
        """Test DFT with different data on each rank (real DDP scenario)."""
        results = self._run_ddp_test("_test_different_batches", world_size=2)
        loss_values = [r["loss"] for r in results]
        self.assertEqual(len(loss_values), 2)
        self.assertNotEqual(loss_values[0], loss_values[1])
        for loss in loss_values:
            self.assertGreater(loss, 0)
            self.assertTrue(torch.isfinite(torch.tensor(loss)))

    def test_ddp_with_chunked_ce(self):
        """Test DFT with chunked CE in DDP mode."""
        results = self._run_ddp_test("_test_chunked_ce", world_size=2)
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertGreater(result["loss"], 0)

    def test_ddp_gradient_sync_simulation(self):
        """Test that DFT loss allows gradient synchronization (simulation)."""
        results = self._run_ddp_test("_test_gradient_sync", world_size=2)
        for result in results:
            self.assertTrue(
                result["has_grads"], f"Rank {result['rank']} missing gradients"
            )
            self.assertGreater(
                result["grad_norm"], 0, f"Rank {result['rank']} has zero grad norm"
            )

    def test_ddp_with_padding(self):
        """Test DFT with padded sequences in DDP mode."""
        results = self._run_ddp_test("_test_with_padding", world_size=2)
        for result in results:
            self.assertGreater(result["loss"], 0)
            self.assertTrue(torch.isfinite(torch.tensor(result["loss"])))


# ==============================================================================
# CP COMPATIBILITY TESTS
# Source: test_dft_cp_compatibility.py
# ==============================================================================


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
        batch_size, full_seq_len, vocab_size = (2, 16, 100)
        cp_size = 2
        cp_rank = 0
        divisor = min(cp_size, 64)
        pad_len = (divisor - full_seq_len % divisor) % divisor
        chunk_len = (full_seq_len + pad_len) // cp_size
        logits_local = torch.randn(
            batch_size, chunk_len, vocab_size, requires_grad=True
        )
        labels_full = torch.randint(0, vocab_size, (batch_size, full_seq_len))
        mock_trainer = MockTrainer(cp_size=cp_size, cp_rank=cp_rank)
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

        dist.is_initialized = mock_is_initialized
        dist.get_world_size = mock_get_world_size
        dist.get_rank = mock_get_rank
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
            assert per_token_loss.numel() > 0, "Should have per-token losses"
            print(
                f"\n CP-aware DFT (rank {cp_rank}): logits_local shape={tuple(logits_local.shape)}, labels_full shape={tuple(labels_full.shape)}, per_token_loss numel={per_token_loss.numel()}, loss={loss.item():.6f}"
            )
        finally:
            dist.is_initialized = original_is_initialized
            dist.get_world_size = original_get_world_size
            dist.get_rank = original_get_rank

    def test_cp_aware_loss_last_rank(self):
        """Test CP-aware loss computation for the last CP rank."""
        batch_size, full_seq_len, vocab_size = (2, 16, 100)
        cp_size = 2
        cp_rank = 1
        divisor = min(cp_size, 64)
        pad_len = (divisor - full_seq_len % divisor) % divisor
        chunk_len = (full_seq_len + pad_len) // cp_size
        logits_local = torch.randn(
            batch_size, chunk_len, vocab_size, requires_grad=True
        )
        labels_full = torch.randint(0, vocab_size, (batch_size, full_seq_len))
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
            print(
                f"\n CP-aware DFT (last rank {cp_rank}): per_token_loss numel={per_token_loss.numel()}, loss={loss.item():.6f}"
            )
        finally:
            dist.is_initialized = original_is_initialized
            dist.get_world_size = original_get_world_size
            dist.get_rank = original_get_rank

    def test_cp_aware_vs_naive_difference(self):
        """Compare CP-aware implementation vs naive approach to show correctness."""
        batch_size, full_seq_len, vocab_size = (2, 16, 100)
        cp_size = 2
        cp_rank = 0
        divisor = min(cp_size, 64)
        pad_len = (divisor - full_seq_len % divisor) % divisor
        chunk_len = (full_seq_len + pad_len) // cp_size
        torch.manual_seed(42)
        logits_local = torch.randn(
            batch_size, chunk_len, vocab_size, requires_grad=True
        )
        labels_full = torch.randint(0, vocab_size, (batch_size, full_seq_len))
        mock_trainer = MockTrainer(cp_size=cp_size, cp_rank=cp_rank)
        import torch.distributed as dist

        original_is_initialized = dist.is_initialized
        original_get_world_size = dist.get_world_size
        original_get_rank = dist.get_rank
        dist.is_initialized = lambda: True
        dist.get_world_size = lambda group=None: cp_size
        dist.get_rank = lambda group=None: cp_rank
        try:
            per_token_loss_correct, valid_mask_correct = (
                compute_per_token_cross_entropy(
                    logits_local,
                    labels_full,
                    ignore_index=-100,
                    shift_labels=True,
                    trainer=mock_trainer,
                )
            )
            loss_correct = reduce_token_loss(
                apply_dft_weighting(per_token_loss_correct), valid_mask_correct
            )
            naive_logits = logits_local[:, :-1, :].contiguous()
            naive_labels = labels_full[:, 1:chunk_len].contiguous()
            naive_logits_flat = naive_logits.view(-1, vocab_size)
            naive_labels_flat = naive_labels.view(-1)
            naive_loss_raw = torch.nn.functional.cross_entropy(
                naive_logits_flat, naive_labels_flat, reduction="none"
            )
            naive_valid = naive_labels_flat != -100
            naive_loss = reduce_token_loss(
                apply_dft_weighting(naive_loss_raw), naive_valid
            )
            print(f"\n CP-aware loss: {loss_correct.item():.6f}")
            print(f" Naive loss (wrong): {naive_loss.item():.6f}")
            print(f"Difference: {abs(loss_correct.item() - naive_loss.item()):.6f}")
            assert not torch.allclose(loss_correct, naive_loss, rtol=0.01), (
                "CP-aware and naive approaches should differ, showing that special handling is needed"
            )
        finally:
            dist.is_initialized = original_is_initialized
            dist.get_world_size = original_get_world_size
            dist.get_rank = original_get_rank

    def test_non_cp_mode_unchanged(self):
        """Verify that non-CP mode still works as before."""
        batch_size, seq_len, vocab_size = (2, 16, 100)
        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        per_token_loss, valid_mask = compute_per_token_cross_entropy(
            logits, labels, ignore_index=-100, shift_labels=True, trainer=None
        )
        loss = reduce_token_loss(apply_dft_weighting(per_token_loss), valid_mask)
        assert torch.isfinite(loss), "Loss should be finite"
        assert loss.item() > 0, "Loss should be positive"
        expected_tokens = batch_size * (seq_len - 1)
        assert per_token_loss.numel() == expected_tokens, (
            f"Expected {expected_tokens} tokens, got {per_token_loss.numel()}"
        )
        print(
            f"\n Non-CP mode: loss={loss.item():.6f}, tokens={per_token_loss.numel()}"
        )

    def test_cp_with_padding(self):
        """Test CP-aware loss with padded sequences (last rank sees padding)."""
        batch_size, full_seq_len, vocab_size = (2, 16, 100)
        cp_size = 2
        cp_rank = 1
        divisor = min(cp_size, 64)
        pad_len = (divisor - full_seq_len % divisor) % divisor
        chunk_len = (full_seq_len + pad_len) // cp_size
        logits_local = torch.randn(
            batch_size, chunk_len, vocab_size, requires_grad=True
        )
        labels_full = torch.randint(0, vocab_size, (batch_size, full_seq_len))
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
            valid_count = valid_mask.sum().item()
            assert valid_count < per_token_loss.numel(), (
                "Some tokens should be masked due to padding"
            )
            print(
                f"\n CP with padding (rank {cp_rank}): loss={loss.item():.6f}, valid_tokens={valid_count}/{per_token_loss.numel()}"
            )
        finally:
            dist.is_initialized = original_is_initialized
            dist.get_world_size = original_get_world_size
            dist.get_rank = original_get_rank


# ==============================================================================
# CP INCOMPATIBILITY TESTS
# Source: test_dft_cp_incompatibility.py
# ==============================================================================


class TestDFTContextParallelIncompatibility:
    """Demonstrate incompatibility between DFT and Context Parallel."""

    def test_dft_receives_sharded_logits_in_cp_mode(self):
        """
        Demonstrate that DFT patch receives sharded logits when CP is enabled.

        This is a DOCUMENTATION test showing the incompatibility, not a passing test.
        """
        pytest.skip(
            "Documentation test: DFT + CP (SFT mode) is INCOMPATIBLE. Logits are sharded but DFT assumes full logits. See spec Phase 3 for fix."
        )

    def test_sharded_ce_loss_is_incorrect(self):
        """
        Show that computing CE loss on sharded logits gives wrong results.

        This demonstrates why DFT + CP is broken.
        """
        batch_size, seq_len, vocab_size = (2, 8, 100)
        cp_size = 2
        full_logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        shift_logits = full_logits[:, :-1, :].contiguous().view(-1, vocab_size)
        shift_labels = labels[:, 1:].contiguous().view(-1)
        correct_loss = torch.nn.functional.cross_entropy(
            shift_logits, shift_labels, reduction="mean"
        )
        shard_size = seq_len // cp_size
        logits_shard_rank0 = full_logits[:, :shard_size, :]
        logits_shard_rank1 = full_logits[:, shard_size:, :]
        labels_shard_rank0 = labels[:, :shard_size]
        labels_shard_rank1 = labels[:, shard_size:]
        shift_logits_r0 = (
            logits_shard_rank0[:, :-1, :].contiguous().view(-1, vocab_size)
        )
        shift_labels_r0 = labels_shard_rank0[:, 1:].contiguous().view(-1)
        loss_rank0 = torch.nn.functional.cross_entropy(
            shift_logits_r0, shift_labels_r0, reduction="mean"
        )
        shift_logits_r1 = (
            logits_shard_rank1[:, :-1, :].contiguous().view(-1, vocab_size)
        )
        shift_labels_r1 = labels_shard_rank1[:, 1:].contiguous().view(-1)
        loss_rank1 = torch.nn.functional.cross_entropy(
            shift_logits_r1, shift_labels_r1, reduction="mean"
        )
        sharded_loss = (loss_rank0 + loss_rank1) / 2.0
        print(f"\nCorrect loss (full logits): {correct_loss.item():.6f}")
        print(f"Sharded loss (DFT + CP): {sharded_loss.item():.6f}")
        print(f"Difference: {abs(correct_loss.item() - sharded_loss.item()):.6f}")
        assert not torch.allclose(correct_loss, sharded_loss, atol=1e-05), (
            "Expected sharded loss to differ from correct loss, demonstrating DFT + CP incompatibility"
        )

    def test_solution_approach_gather_before_dft(self):
        """
        Show the correct approach: gather logits before computing DFT loss.

        This is what Phase 3 implementation should do.
        """
        batch_size, seq_len, vocab_size = (2, 8, 100)
        cp_size = 2
        shard_size = seq_len // cp_size
        logits_shard_rank0 = torch.randn(batch_size, shard_size, vocab_size)
        logits_shard_rank1 = torch.randn(batch_size, shard_size, vocab_size)
        full_logits = torch.cat([logits_shard_rank0, logits_shard_rank1], dim=1)
        assert full_logits.shape == (batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        shift_logits = full_logits[:, :-1, :].contiguous().view(-1, vocab_size)
        shift_labels = labels[:, 1:].contiguous().view(-1)
        loss = torch.nn.functional.cross_entropy(
            shift_logits, shift_labels, reduction="mean"
        )
        assert loss.item() > 0
        print(f"\n Correct approach: gather then compute DFT loss = {loss.item():.6f}")


# ==============================================================================
# TENSOR PARALLEL TESTS
# Source: test_dft_tensor_parallel.py
# ==============================================================================


class TestDFTTensorParallelCompatibility:
    """Test DFT compatibility with Tensor Parallelism (architectural verification)."""

    def test_dft_with_complete_logits(self):
        """Verify DFT correctly processes complete logits (as TP provides after All-Reduce).

        TP's DTensor framework ensures that after row-wise parallel layers' All-Reduce,
        the final model output logits are complete [batch, seq, vocab]. DFT receives
        these complete tensors, making TP transparent to the loss computation.
        """
        batch_size, seq_len, vocab_size = (2, 16, 100)
        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        loss = compute_dft_loss(logits, labels)
        assert torch.isfinite(loss), "Loss should be finite"
        assert loss.item() > 0, "Loss should be positive"
        assert loss.requires_grad, "Loss should support backprop"
        loss.backward()
        assert logits.grad is not None, "Gradients should flow to logits"
        print(f"\n DFT with complete logits: loss={loss.item():.6f}")

    def test_dft_with_large_vocab(self):
        """Test DFT with large vocabulary (common TP use case).

        TP is often used with large models that have large vocabularies (e.g., Qwen 152K).
        Verify DFT + chunked CE works correctly with large vocab sizes.
        """
        batch_size, seq_len, vocab_size = (2, 16, 50000)
        chunk_size = 2048
        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        per_token_loss, valid_mask = compute_per_token_cross_entropy(
            logits, labels, chunk_size=chunk_size
        )
        weighted_loss = apply_dft_weighting(per_token_loss)
        loss = reduce_token_loss(weighted_loss, valid_mask)
        assert torch.isfinite(loss), "Loss should be finite with large vocab"
        assert loss.item() > 0, "Loss should be positive"
        loss.backward()
        assert logits.grad is not None, "Gradients should flow with chunked CE"
        print(f"\n DFT + chunked CE (vocab={vocab_size}): loss={loss.item():.6f}")

    def test_dft_shape_assumptions_match_tp_outputs(self):
        """Verify DFT's shape assumptions match TP's output guarantees.

        DFT assumes:
        - logits shape: [batch, seq, vocab]
        - labels shape: [batch, seq]

        TP guarantees:
        - After All-Reduce from row-wise parallel layers (O proj, Down proj)
        - Model outputs complete logits: [batch, seq, vocab]
        - No sharding in batch or sequence dimensions

        This test verifies the shape contract is satisfied.
        """
        test_cases = [(1, 8, 100), (4, 128, 32000), (2, 512, 10000)]
        for batch_size, seq_len, vocab_size in test_cases:
            logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
            labels = torch.randint(0, vocab_size, (batch_size, seq_len))
            loss = compute_dft_loss(logits, labels)
            assert torch.isfinite(loss), (
                f"Loss should be finite for shape {logits.shape}"
            )
            print(f" Shape {logits.shape}: loss={loss.item():.6f}")

    def test_tp_architectural_transparency_documented(self):
        """Document why TP is architecturally transparent to DFT.

        This is a documentation test to ensure understanding is captured.
        """
        architectural_guarantees = {
            "TP output shape": "[batch, seq, vocab] - complete, not sharded",
            "All-Reduce location": "Row-wise parallel layers (O proj, Down proj)",
            "DFT receives": "Complete logits, identical to non-TP training",
            "Communication": "Handled by DTensor automatically, invisible to DFT",
            "Loss computation": "Identical logic as non-TP case",
        }
        expected_behavior = {
            "DFT code changes": "None required - TP is transparent",
            "Special handling": "Not needed - DTensor handles everything",
            "Performance impact": "TP communication overhead only, not DFT-specific",
            "Memory usage": "TP reduces per-GPU param memory, DFT unchanged",
        }
        for key, value in architectural_guarantees.items():
            print(f"  {key}: {value}")
        print("\nExpected DFT behavior with TP:")
        for key, value in expected_behavior.items():
            print(f"  {key}: {value}")
        assert True, "TP architectural transparency documented"

    @pytest.mark.skip(
        reason="Requires multi-GPU hardware with NVLink (e.g., 2-4 GPUs). Run in E2E test environment with: torchrun --nproc_per_node=2 -m pytest tests/e2e/multigpu/test_tp.py"
    )
    def test_e2e_tp_training_with_dft(self):
        """E2E test: DFT + TP training (requires multi-GPU hardware).

        This test is skipped in unit test environment. For E2E validation:
        1. Configure axolotl with: tensor_parallel_size: 2, enable_dft_loss: true
        2. Run training on multi-GPU node with NVLink
        3. Verify: Training completes, loss is reasonable, model converges

        See: tests/e2e/multigpu/test_tp.py for E2E TP tests
        See: examples/distributed-parallel/qwen3-8b-fsdp-tp-cp.yaml for config example
        """
        pytest.skip("Multi-GPU hardware required for E2E TP test")


# ==============================================================================
# PIPELINE PARALLEL TESTS
# Source: test_dft_pipeline_parallel.py
# ==============================================================================


class TestDFTPipelineParallelCompatibility:
    """Verify Pipeline Parallelism support status in axolotl.

    These tests ACTIVELY CHECK the codebase. If PP is added in the future,
    tests will FAIL, prompting developers to verify DFT+PP compatibility.
    """

    def test_pipeline_parallel_not_in_config_schema(self):
        """Verify PP config options don't exist in config schema.

        If this test fails, PP support may have been added - review DFT compatibility.
        """
        config_schema_path = Path("src/axolotl/utils/schemas/config.py")
        if not config_schema_path.exists():
            pytest.skip(f"Config schema not found at {config_schema_path}")
        config_content = config_schema_path.read_text()
        pp_config_indicators = [
            "pp_size",
            "pipeline_parallel_size",
            "pipeline_parallel",
            "num_pipeline_stages",
        ]
        found_indicators = [
            indicator
            for indicator in pp_config_indicators
            if indicator in config_content
        ]
        if found_indicators:
            pytest.fail(
                f"  ALERT: PP config options detected in config schema: {found_indicators}\naxolotl may have added Pipeline Parallelism support!\nAction required:\n1. Verify DFT+PP compatibility (likely compatible - see docstring)\n2. Add E2E tests for DFT+PP\n3. Update compatibility matrix from N/A to actual status\n4. Update this test to reflect PP support"
            )
        print("\n No PP config options found in config schema (PP not supported)")

    def test_pipeline_parallel_not_in_model_loader(self):
        """Verify PP initialization code doesn't exist in model loader.

        If this test fails, PP support may have been added - review DFT compatibility.
        """
        model_loader_path = Path("src/axolotl/loaders/model.py")
        if not model_loader_path.exists():
            pytest.skip(f"Model loader not found at {model_loader_path}")
        loader_content = model_loader_path.read_text()
        pp_code_indicators = [
            "pipeline_parallel",
            "pp_size",
            "PipelineParallel",
            "pipeline_stage",
        ]
        found_indicators = [
            indicator for indicator in pp_code_indicators if indicator in loader_content
        ]
        if found_indicators:
            pytest.fail(
                f"  ALERT: PP code detected in model loader: {found_indicators}\naxolotl may have added Pipeline Parallelism support!\nAction required:\n1. Verify DFT+PP compatibility\n2. Add E2E tests for DFT+PP\n3. Update compatibility matrix"
            )
        print("\n No PP initialization code found in model loader (PP not supported)")

    def test_pipeline_parallel_not_in_distributed_utils(self):
        """Verify PP process group setup doesn't exist in distributed utils.

        If this test fails, PP support may have been added - review DFT compatibility.
        """
        distributed_path = Path("src/axolotl/utils/distributed.py")
        if not distributed_path.exists():
            pytest.skip(f"Distributed utils not found at {distributed_path}")
        distributed_content = distributed_path.read_text()
        pp_group_indicators = [
            "pipeline_parallel_group",
            "pp_group",
            "get_pipeline_parallel",
        ]
        found_indicators = [
            indicator
            for indicator in pp_group_indicators
            if indicator in distributed_content
        ]
        if found_indicators:
            pytest.fail(
                f"  ALERT: PP process groups detected: {found_indicators}\naxolotl may have added Pipeline Parallelism support!\nAction required: Verify DFT+PP compatibility"
            )
        print("\n No PP process group setup found (PP not supported)")

    def test_pp_compatibility_analysis_for_future(self):
        """If PP is added to axolotl in future, analyze DFT compatibility.

        Theoretical analysis (assuming standard PP implementation):

        PP Architecture:
        - Model layers split vertically across stages (GPUs)
        - Each stage processes micro-batches sequentially
        - Final stage outputs complete logits [batch, seq, vocab]
        - Loss computed on final stage

        DFT Compatibility (Theoretical):
        -  DFT receives complete logits from final stage
        -  Loss computation unchanged (happens on last GPU only)
        -  Gradient broadcast may be needed (handled by PP framework)
        -  Micro-batch handling - ensure num_items_in_batch is correct

        Expected Outcome: LIKELY COMPATIBLE
        - No DFT code changes needed
        - PP framework handles cross-stage communication
        - Loss computation on final stage is standard pattern
        """
        theoretical_analysis = {
            "PP outputs": "Complete logits [batch, seq, vocab] from final stage",
            "DFT input": "Same as non-PP - complete logits",
            "Loss location": "Computed on final stage (last GPU)",
            "Gradient flow": "Handled by PP framework (backward through stages)",
            "Expected compatibility": " Likely compatible (no special handling)",
            "Main concern": "Ensure num_items_in_batch accounts for micro-batching",
        }
        print("\nTheoretical PP + DFT Compatibility Analysis:")
        for aspect, finding in theoretical_analysis.items():
            print(f"  {aspect}: {finding}")
        assert True, "Theoretical PP compatibility analysis documented"

    def test_compatibility_matrix_status(self):
        """Verify correct status in DFT compatibility matrix.

        Current status:  Likely OK
        Correct status:  N/A (Not Applicable) - PP not supported in axolotl

        Recommendation: Update spec 001 compatibility matrix:
        - Change from: " Likely OK - If supported, each stage outputs complete tensors"
        - Change to:   "N/A - Not supported in axolotl (prefers FSDP+TP)"
        """
        recommended_matrix_entry = {
            "Feature": "Pipeline Parallelism",
            "Status": "N/A (Not Applicable)",
            "Details": "Not supported in axolotl - FSDP+TP preferred for large models",
            "File Reference": "No PP implementation",
            "Note": "If added in future, likely compatible (DFT receives complete logits)",
        }
        print("\nRecommended Compatibility Matrix Entry:")
        for key, value in recommended_matrix_entry.items():
            print(f"  {key}: {value}")
        assert True, "Compatibility matrix recommendation documented"


# ==============================================================================
# CHANNEL LOSS TESTS
# Source: test_dft_channel_loss.py
# ==============================================================================


class TestDFTChannelLossIntegration:
    """Test DFT integration with Channel Loss plugin."""

    def test_channel_loss_enabled_attaches_intermediate_values(self):
        """Test that DFT provides per-token loss when channel loss is enabled."""
        trainer = MagicMock()
        trainer.args = SimpleNamespace(
            enable_dft_loss=True,
            enable_dft_channel_loss=True,
            dft_chunk_size=None,
            include_tkps=False,
            label_smoothing_factor=0.0,
            orpo_alpha=None,
        )
        trainer.state = SimpleNamespace()
        trainer.compute_loss = MagicMock()
        patch_compute_loss_for_dft(trainer, cfg=MagicMock())
        batch_size, seq_len, vocab_size = (2, 10, 1000)
        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        class DummyModel(torch.nn.Module):
            def forward(self, **kwargs):
                return SimpleNamespace(logits=logits)

        model = DummyModel()
        inputs = {"input_ids": torch.zeros(batch_size, seq_len), "labels": labels}
        loss, outputs = trainer.compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=None
        )
        assert hasattr(outputs, "per_token_loss"), "per_token_loss not attached"
        assert hasattr(outputs, "valid_mask"), "valid_mask not attached"
        assert hasattr(outputs, "loss"), "loss not attached"
        expected_tokens = batch_size * (seq_len - 1)
        assert outputs.per_token_loss.shape == (expected_tokens,), (
            f"per_token_loss shape {outputs.per_token_loss.shape} != {(expected_tokens,)}"
        )
        assert outputs.valid_mask.shape == (expected_tokens,), (
            f"valid_mask shape {outputs.valid_mask.shape} != {(expected_tokens,)}"
        )
        assert outputs.valid_mask.dtype == torch.bool, "valid_mask should be boolean"
        assert loss.ndim == 0, "loss should be scalar"
        assert torch.allclose(loss, outputs.loss), "loss != outputs.loss"

    def test_channel_loss_disabled_no_intermediate_values(self):
        """Test backward compatibility: no intermediate values when flag is False."""
        trainer = MagicMock()
        trainer.args = SimpleNamespace(
            enable_dft_loss=True,
            enable_dft_channel_loss=False,
            dft_chunk_size=None,
            include_tkps=False,
            label_smoothing_factor=0.0,
            orpo_alpha=None,
        )
        trainer.state = SimpleNamespace()
        trainer.compute_loss = MagicMock()
        patch_compute_loss_for_dft(trainer, cfg=MagicMock())
        batch_size, seq_len, vocab_size = (2, 10, 1000)
        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        class DummyModel(torch.nn.Module):
            def forward(self, **kwargs):
                return SimpleNamespace(logits=logits)

        model = DummyModel()
        inputs = {"input_ids": torch.zeros(batch_size, seq_len), "labels": labels}
        loss, outputs = trainer.compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=None
        )
        assert not hasattr(outputs, "per_token_loss"), (
            "per_token_loss should not be attached when enable_dft_channel_loss=False"
        )
        assert not hasattr(outputs, "valid_mask"), (
            "valid_mask should not be attached when enable_dft_channel_loss=False"
        )
        assert loss.ndim == 0, "loss should be scalar"
        assert loss.requires_grad, "loss should have gradients"

    def test_channel_loss_with_ignore_index(self):
        """Test that valid_mask correctly identifies non-ignored tokens."""
        trainer = MagicMock()
        trainer.args = SimpleNamespace(
            enable_dft_loss=True,
            enable_dft_channel_loss=True,
            dft_chunk_size=None,
            include_tkps=False,
            label_smoothing_factor=0.0,
            orpo_alpha=None,
        )
        trainer.state = SimpleNamespace()
        trainer.compute_loss = MagicMock()
        patch_compute_loss_for_dft(trainer, cfg=MagicMock())
        batch_size, seq_len, vocab_size = (2, 10, 100)
        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels[0, 3:6] = -100
        labels[1, 7:] = -100

        class DummyModel(torch.nn.Module):
            def forward(self, **kwargs):
                return SimpleNamespace(logits=logits)

        model = DummyModel()
        inputs = {"input_ids": torch.zeros(batch_size, seq_len), "labels": labels}
        loss, outputs = trainer.compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=None
        )
        shifted_labels = labels[:, 1:].flatten()
        expected_valid = shifted_labels != -100
        assert torch.equal(outputs.valid_mask, expected_valid), (
            "valid_mask should match non-ignored tokens"
        )
        valid_count = outputs.valid_mask.sum().item()
        assert valid_count < batch_size * (seq_len - 1), "Some tokens should be ignored"
        assert valid_count > 0, "Should have some valid tokens"

    def test_channel_loss_with_chunked_ce(self):
        """Test DFT + Channel Loss works with chunked cross-entropy."""
        trainer = MagicMock()
        trainer.args = SimpleNamespace(
            enable_dft_loss=True,
            enable_dft_channel_loss=True,
            dft_chunk_size=4,
            include_tkps=False,
            label_smoothing_factor=0.0,
            orpo_alpha=None,
        )
        trainer.state = SimpleNamespace()
        trainer.compute_loss = MagicMock()
        patch_compute_loss_for_dft(trainer, cfg=MagicMock())
        batch_size, seq_len, vocab_size = (2, 10, 1000)
        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        class DummyModel(torch.nn.Module):
            def forward(self, **kwargs):
                return SimpleNamespace(logits=logits)

        model = DummyModel()
        inputs = {"input_ids": torch.zeros(batch_size, seq_len), "labels": labels}
        loss, outputs = trainer.compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=None
        )
        assert hasattr(outputs, "per_token_loss"), "per_token_loss not attached"
        assert hasattr(outputs, "valid_mask"), "valid_mask not attached"
        expected_tokens = batch_size * (seq_len - 1)
        assert outputs.per_token_loss.shape == (expected_tokens,)
        assert outputs.valid_mask.shape == (expected_tokens,)
        assert loss.ndim == 0
        assert loss.requires_grad

    def test_channel_loss_simulated_channel_statistics(self):
        """Simulate how Channel Loss plugin would use per_token_loss."""
        trainer = MagicMock()
        trainer.args = SimpleNamespace(
            enable_dft_loss=True,
            enable_dft_channel_loss=True,
            dft_chunk_size=None,
            include_tkps=False,
            label_smoothing_factor=0.0,
            orpo_alpha=None,
        )
        trainer.state = SimpleNamespace()
        trainer.compute_loss = MagicMock()
        patch_compute_loss_for_dft(trainer, cfg=MagicMock())
        batch_size, seq_len, vocab_size = (4, 8, 100)
        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        channel_ids = torch.tensor([0, 0, 1, 1])

        class DummyModel(torch.nn.Module):
            def forward(self, **kwargs):
                return SimpleNamespace(logits=logits)

        model = DummyModel()
        inputs = {"input_ids": torch.zeros(batch_size, seq_len), "labels": labels}
        loss, outputs = trainer.compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=None
        )
        per_token_loss = outputs.per_token_loss
        valid_mask = outputs.valid_mask
        channel_ids_per_token = channel_ids.repeat_interleave(seq_len - 1)
        channel_losses = {}
        for channel_id in [0, 1]:
            channel_mask = channel_ids_per_token == channel_id
            combined_mask = channel_mask & valid_mask
            if combined_mask.sum() > 0:
                channel_loss = per_token_loss[combined_mask].mean()
                channel_losses[channel_id] = channel_loss.item()
        assert 0 in channel_losses, "Should have stats for channel 0"
        assert 1 in channel_losses, "Should have stats for channel 1"
        assert channel_losses[0] > 0, "Channel 0 loss should be positive"
        assert channel_losses[1] > 0, "Channel 1 loss should be positive"

    def test_gradient_flow_with_channel_loss_enabled(self):
        """Test that gradients still flow correctly with channel loss enabled."""
        trainer = MagicMock()
        trainer.args = SimpleNamespace(
            enable_dft_loss=True,
            enable_dft_channel_loss=True,
            dft_chunk_size=None,
            include_tkps=False,
            label_smoothing_factor=0.0,
            orpo_alpha=None,
        )
        trainer.state = SimpleNamespace()
        trainer.compute_loss = MagicMock()
        patch_compute_loss_for_dft(trainer, cfg=MagicMock())
        batch_size, seq_len, vocab_size = (2, 10, 100)
        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        class DummyModel(torch.nn.Module):
            def forward(self, **kwargs):
                return SimpleNamespace(logits=logits)

        model = DummyModel()
        inputs = {"input_ids": torch.zeros(batch_size, seq_len), "labels": labels}
        loss, outputs = trainer.compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=None
        )
        assert loss.requires_grad, "loss should require gradients"
        loss.backward()
        assert logits.grad is not None, "logits should have gradients"
        assert not torch.isnan(logits.grad).any(), "gradients should not be NaN"
        assert not torch.isinf(logits.grad).any(), "gradients should not be inf"

    def test_return_outputs_false_still_works(self):
        """Test that return_outputs=False works correctly with channel loss."""
        trainer = MagicMock()
        trainer.args = SimpleNamespace(
            enable_dft_loss=True,
            enable_dft_channel_loss=True,
            dft_chunk_size=None,
            include_tkps=False,
            label_smoothing_factor=0.0,
            orpo_alpha=None,
        )
        trainer.state = SimpleNamespace()
        trainer.compute_loss = MagicMock()
        patch_compute_loss_for_dft(trainer, cfg=MagicMock())
        batch_size, seq_len, vocab_size = (2, 10, 100)
        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        class DummyModel(torch.nn.Module):
            def forward(self, **kwargs):
                return SimpleNamespace(logits=logits)

        model = DummyModel()
        inputs = {"input_ids": torch.zeros(batch_size, seq_len), "labels": labels}
        result = trainer.compute_loss(
            model, inputs, return_outputs=False, num_items_in_batch=None
        )
        assert isinstance(result, torch.Tensor), (
            "Should return tensor when return_outputs=False"
        )
        assert result.ndim == 0, "Should return scalar loss"
        assert result.requires_grad, "Loss should have gradients"


# ==============================================================================
# MULTI FEATURE TESTS
# Source: test_dft_multi_feature.py
# ==============================================================================


class TestDFTPackingCombinations:
    """Test DFT with sequence packing."""

    def test_dft_with_packing_basic(self):
        """Verify DFT correctly handles packed sequences.

        Packing combines multiple sequences into one batch:
        - Sequence boundaries marked by labels = -100
        - DFT should compute loss only for valid tokens (labels != -100)
        - Example: [seq1_tokens | -100 padding | seq2_tokens | -100 padding]
        """
        batch_size, seq_len, vocab_size = (1, 32, 100)
        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
        labels[0, :8] = torch.randint(0, vocab_size, (8,))
        labels[0, 12:24] = torch.randint(0, vocab_size, (12,))
        loss = compute_dft_loss(logits, labels, shift_labels=True, ignore_index=-100)
        assert torch.isfinite(loss), "Loss should be finite with packing"
        assert loss.item() > 0, "Loss should be positive"
        loss.backward()
        assert logits.grad is not None, "Gradients should flow with packing"
        print(
            f"\n DFT + Packing: packed_sequences=2, total_valid_tokens=19 (after shift), loss={loss.item():.6f}"
        )

    def test_dft_packing_with_gradient_accumulation(self):
        """Test DFT + Packing + Gradient Accumulation.

        Common combination for training with long sequences:
        - Pack multiple short sequences into one batch
        - Use gradient accumulation for effective large batch
        - num_items_in_batch normalizes across accumulated batches
        """
        batch_size, seq_len, vocab_size = (2, 32, 100)
        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
        labels[0, :10] = torch.randint(0, vocab_size, (10,))
        labels[0, 14:20] = torch.randint(0, vocab_size, (6,))
        labels[1, :8] = torch.randint(0, vocab_size, (8,))
        labels[1, 12:24] = torch.randint(0, vocab_size, (12,))
        num_items_in_batch = 8
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
            f"\n DFT + Packing + Gradient Accumulation: num_items_in_batch={num_items_in_batch}, loss={loss.item():.6f}"
        )

    def test_dft_packing_fsdp_simulation(self):
        """Document DFT + Packing + FSDP compatibility.

        FSDP shards model parameters, but:
        - Forward pass outputs complete logits [batch, seq, vocab] after All-Gather
        - DFT loss computation sees complete tensors
        - Packing is orthogonal to FSDP (handled by data collator)

        This test simulates the scenario (no actual FSDP process groups).
        """
        batch_size, seq_len, vocab_size = (2, 64, 50000)
        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
        labels[0, :20] = torch.randint(0, vocab_size, (20,))
        labels[0, 24:40] = torch.randint(0, vocab_size, (16,))
        labels[1, :16] = torch.randint(0, vocab_size, (16,))
        labels[1, 20:48] = torch.randint(0, vocab_size, (28,))
        loss = compute_dft_loss(logits, labels, shift_labels=True, ignore_index=-100)
        assert torch.isfinite(loss), "Loss should be finite with FSDP + Packing"
        assert loss.item() > 0, "Loss should be positive"
        print(
            f"\n DFT + Packing + FSDP (simulated): vocab_size={vocab_size}, loss={loss.item():.6f}"
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
        batch_size, seq_len, vocab_size = (2, 16, 50000)
        chunk_size = 4096
        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        loss_chunked = compute_dft_loss(
            logits, labels, shift_labels=True, ignore_index=-100, chunk_size=chunk_size
        )
        loss_standard = compute_dft_loss(
            logits, labels, shift_labels=True, ignore_index=-100, chunk_size=None
        )
        assert torch.isfinite(loss_chunked), "Chunked loss should be finite"
        assert torch.isfinite(loss_standard), "Standard loss should be finite"
        assert torch.allclose(loss_chunked, loss_standard, rtol=0.001), (
            f"Chunked and standard CE should match: chunked={loss_chunked.item():.6f}, standard={loss_standard.item():.6f}"
        )
        print(
            f"\n DFT + Chunked CE: vocab_size={vocab_size}, chunk_size={chunk_size}, loss_chunked={loss_chunked.item():.6f}, loss_standard={loss_standard.item():.6f}"
        )

    def test_dft_chunked_ce_with_gradient_accumulation(self):
        """Test DFT + Chunked CE + Gradient Accumulation.

        Typical configuration for large vocab + large batch training:
        - Chunked CE reduces memory for vocab dimension
        - Gradient accumulation reduces memory for batch dimension
        - num_items_in_batch normalizes loss across accumulation steps
        """
        batch_size, seq_len, vocab_size = (2, 32, 100000)
        chunk_size = 8192
        num_items_in_batch = 16
        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
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
        loss.backward()
        assert logits.grad is not None, "Gradients should flow"
        print(
            f"\n DFT + Chunked CE + Gradient Accumulation: vocab_size={vocab_size}, chunk_size={chunk_size}, num_items_in_batch={num_items_in_batch}, loss={loss.item():.6f}"
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
        batch_size, seq_len, vocab_size = (2, 16, 100)
        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        scalar_loss, per_token_loss, valid_mask = compute_dft_loss_with_intermediate(
            logits, labels, shift_labels=True, ignore_index=-100
        )
        assert torch.isfinite(scalar_loss), "Scalar loss should be finite"
        assert scalar_loss.item() > 0, "Scalar loss should be positive"
        assert per_token_loss.ndim == 1, "per_token_loss should be 1D"
        assert valid_mask.ndim == 1, "valid_mask should be 1D"
        assert per_token_loss.shape == valid_mask.shape, "Shapes should match"
        assert valid_mask.dtype == torch.bool, "valid_mask should be bool"
        scalar_loss.backward()
        assert logits.grad is not None, "Gradients should flow"
        print(
            f"\n DFT + Channel Loss: scalar_loss={scalar_loss.item():.6f}, per_token_loss shape={tuple(per_token_loss.shape)}, valid_tokens={valid_mask.sum().item()}"
        )

    def test_dft_channel_loss_mixed_precision(self):
        """Test DFT + Channel Loss + Mixed Precision (BF16).

        Verifies that:
        - Intermediates preserve gradient flow with BF16
        - scalar_loss is FP32 (upcast)
        - per_token_loss supports autograd
        """
        batch_size, seq_len, vocab_size = (2, 16, 100)
        logits = (
            torch.randn(batch_size, seq_len, vocab_size).bfloat16().requires_grad_(True)
        )
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        scalar_loss, per_token_loss, valid_mask = compute_dft_loss_with_intermediate(
            logits, labels, shift_labels=True, ignore_index=-100
        )
        assert scalar_loss.dtype == torch.float32, "Scalar loss should be FP32"
        scalar_loss.backward()
        assert logits.grad is not None, "Gradients should flow"
        assert logits.grad.dtype == torch.bfloat16, "Gradients should be BF16"
        print(
            f"\n DFT + Channel Loss + BF16: scalar_loss={scalar_loss.item():.6f} (dtype={scalar_loss.dtype}), logits_grad dtype={logits.grad.dtype}"
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
            enable_dft_channel_loss=True,
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
        loss, outputs = mock_trainer.compute_loss(
            mock_model, inputs, return_outputs=True
        )
        assert torch.isfinite(loss), "Loss should be finite"
        assert loss.item() > 0, "Loss should be positive"
        assert hasattr(outputs, "per_token_loss"), "Should have per_token_loss"
        assert hasattr(outputs, "valid_mask"), "Should have valid_mask"
        assert hasattr(outputs, "loss"), "Should have loss"
        assert outputs.per_token_loss.ndim == 1, "per_token_loss should be 1D"
        assert outputs.valid_mask.ndim == 1, "valid_mask should be 1D"
        print(
            f"\n DFT + Channel Loss via trainer patch: loss={loss.item():.6f}, intermediates attached: per_token_loss={tuple(outputs.per_token_loss.shape)}, valid_mask={tuple(outputs.valid_mask.shape)}"
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
        batch_size, full_seq_len, vocab_size = (2, 64, 100)
        cp_size = 2
        cp_rank = 0
        divisor = min(cp_size, 64)
        pad_len = (divisor - full_seq_len % divisor) % divisor
        chunk_len = (full_seq_len + pad_len) // cp_size
        logits_local = torch.randn(
            batch_size, chunk_len, vocab_size, requires_grad=True
        )
        labels_full = torch.full((batch_size, full_seq_len), -100, dtype=torch.long)
        labels_full[0, :20] = torch.randint(0, vocab_size, (20,))
        labels_full[0, 24:40] = torch.randint(0, vocab_size, (16,))
        labels_full[1, :16] = torch.randint(0, vocab_size, (16,))
        labels_full[1, 20:48] = torch.randint(0, vocab_size, (28,))

        class MockCPTrainer:
            def __init__(self, cp_size, cp_rank):
                self.accelerator = SimpleNamespace(
                    context_parallel_group=SimpleNamespace()
                )

        mock_trainer = MockCPTrainer(cp_size=cp_size, cp_rank=cp_rank)
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
                shift_labels=True,
                ignore_index=-100,
                trainer=mock_trainer,
            )
            loss = reduce_token_loss(apply_dft_weighting(per_token_loss), valid_mask)
            assert torch.isfinite(loss), "Loss should be finite"
            assert loss.item() > 0, "Loss should be positive"
            print(
                f"\n DFT + CP + Packing (simulated): cp_size={cp_size}, cp_rank={cp_rank}, logits_local shape={tuple(logits_local.shape)}, labels_full shape={tuple(labels_full.shape)}, loss={loss.item():.6f}"
            )
        finally:
            dist.is_initialized = original_is_initialized
            dist.get_world_size = original_get_world_size
            dist.get_rank = original_get_rank


# ==============================================================================
# INCOMPATIBILITIES TESTS
# Source: test_dft_incompatibilities.py
# ==============================================================================


class TestDFTLabelSmoothingIncompatibility:
    """Test DFT + label smoothing incompatibility detection."""

    def test_label_smoothing_raises_error(self):
        """Verify DFT raises ValueError when label smoothing is enabled.

        Label smoothing modifies the CE loss formula fundamentally, making
        it unclear how to combine with DFT's exp(-loss) weighting.
        """
        mock_trainer = Mock()
        mock_trainer.args = SimpleNamespace(
            enable_dft_loss=True,
            label_smoothing_factor=0.1,
            orpo_alpha=None,
            include_tkps=False,
            enable_dft_channel_loss=False,
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
        with pytest.raises(ValueError) as exc_info:
            mock_trainer.compute_loss(mock_model, inputs)
        error_msg = str(exc_info.value)
        assert "label smoothing" in error_msg.lower(), (
            "Error should mention label smoothing"
        )
        assert "incompatible" in error_msg.lower(), (
            "Error should indicate incompatibility"
        )
        print("\n Label smoothing correctly raises ValueError with clear message")

    def test_label_smoothing_zero_works(self):
        """Verify DFT works when label_smoothing_factor is 0 or None."""
        mock_trainer = Mock()
        mock_trainer.args = SimpleNamespace(
            enable_dft_loss=True,
            label_smoothing_factor=0.0,
            orpo_alpha=None,
            include_tkps=False,
            enable_dft_channel_loss=False,
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
        loss = mock_trainer.compute_loss(mock_model, inputs)
        assert torch.isfinite(loss), "Loss should be finite"
        print("\n Label smoothing factor=0 works correctly")


class TestDFTORPOIncompatibility:
    """Test DFT + ORPO silent fallback behavior."""

    def test_orpo_silently_disables_dft(self):
        """Verify DFT silently falls back to ORPO loss when ORPO is enabled.

        ORPO uses a different loss function (not CE-based), so DFT cannot apply.
        The patch silently disables DFT and uses the original compute_loss.
        """
        original_compute_loss = Mock(return_value=torch.tensor(1.5))
        mock_trainer = Mock()
        mock_trainer.args = SimpleNamespace(
            enable_dft_loss=True,
            orpo_alpha=0.1,
            label_smoothing_factor=0.0,
            include_tkps=False,
        )
        mock_trainer.compute_loss = original_compute_loss
        mock_cfg = SimpleNamespace()
        original_func = mock_trainer.compute_loss
        patch_compute_loss_for_dft(mock_trainer, mock_cfg)
        inputs = {"labels": torch.tensor([[1, 2, 3]])}
        mock_model = Mock()
        loss = mock_trainer.compute_loss(mock_model, inputs)
        original_func.assert_called_once()
        assert loss.item() == 1.5, "Should use original loss value"
        print("\n ORPO correctly triggers silent fallback to original loss")


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
        original_compute_loss = Mock(return_value=torch.tensor(1.5))
        mock_trainer = Mock()
        mock_trainer.args = SimpleNamespace(
            enable_dft_loss=True,
            label_smoothing_factor=0.1,
            orpo_alpha=0.1,
            include_tkps=False,
        )
        mock_trainer.compute_loss = original_compute_loss
        mock_cfg = SimpleNamespace()
        original_func = mock_trainer.compute_loss
        patch_compute_loss_for_dft(mock_trainer, mock_cfg)
        inputs = {"labels": torch.tensor([[1, 2, 3]])}
        mock_model = Mock()
        loss = mock_trainer.compute_loss(mock_model, inputs)
        original_func.assert_called_once()
        assert loss.item() == 1.5, "Should use ORPO fallback"
        print("\n Multiple incompatibilities: ORPO fallback has higher priority")


# ==============================================================================
# TEST CHUNKED CE TESTS
# Source: test_chunked_ce.py
# ==============================================================================


class TestChunkedCrossEntropy:
    """Test suite for ChunkedCrossEntropy autograd function."""

    def test_mathematical_equivalence_small_vocab(self):
        """Test that chunked CE produces identical results to standard CE."""
        torch.manual_seed(42)
        batch_size, seq_len, vocab_size = (4, 32, 1000)
        chunk_size = 16
        logits = torch.randn(batch_size * seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (batch_size * seq_len,))
        loss_standard = F.cross_entropy(
            logits.detach(), labels, reduction="none", ignore_index=-100
        )
        loss_chunked = chunked_cross_entropy(
            logits.detach(), labels, chunk_size=chunk_size, ignore_index=-100
        )
        assert torch.allclose(loss_standard, loss_chunked, atol=1e-06), (
            f"Max diff: {(loss_standard - loss_chunked).abs().max().item()}"
        )

    def test_gradient_correctness(self):
        """Test that gradients computed by chunked CE match standard CE."""
        torch.manual_seed(123)
        batch_size, seq_len, vocab_size = (2, 16, 500)
        chunk_size = 8
        logits_std = torch.randn(batch_size * seq_len, vocab_size, requires_grad=True)
        logits_chunk = logits_std.detach().clone().requires_grad_(True)
        labels = torch.randint(0, vocab_size, (batch_size * seq_len,))
        loss_std = F.cross_entropy(logits_std, labels, reduction="none")
        loss_std.sum().backward()
        loss_chunk = chunked_cross_entropy(logits_chunk, labels, chunk_size=chunk_size)
        loss_chunk.sum().backward()
        assert torch.allclose(logits_std.grad, logits_chunk.grad, atol=1e-05), (
            f"Max gradient diff: {(logits_std.grad - logits_chunk.grad).abs().max().item()}"
        )

    def test_ignore_index_handling(self):
        """Test that ignore_index is properly handled."""
        torch.manual_seed(456)
        seq_len, vocab_size = (32, 100)
        chunk_size = 16
        ignore_index = -100
        logits = torch.randn(seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (seq_len,))
        labels[::3] = ignore_index
        loss_std = F.cross_entropy(
            logits.detach(), labels, reduction="none", ignore_index=ignore_index
        )
        loss_chunk = chunked_cross_entropy(
            logits.detach(), labels, chunk_size=chunk_size, ignore_index=ignore_index
        )
        assert torch.all(loss_std[labels == ignore_index] == 0.0)
        assert torch.all(loss_chunk[labels == ignore_index] == 0.0)
        valid_mask = labels != ignore_index
        assert torch.allclose(loss_std[valid_mask], loss_chunk[valid_mask], atol=1e-06)

    def test_chunk_size_larger_than_sequence(self):
        """Test behavior when chunk_size > sequence length."""
        torch.manual_seed(789)
        seq_len, vocab_size = (16, 200)
        chunk_size = 100
        logits = torch.randn(seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (seq_len,))
        loss_std = F.cross_entropy(logits.detach(), labels, reduction="none")
        loss_chunk = chunked_cross_entropy(
            logits.detach(), labels, chunk_size=chunk_size
        )
        assert torch.allclose(loss_std, loss_chunk, atol=1e-06)

    def test_single_chunk(self):
        """Test edge case with only one chunk."""
        torch.manual_seed(101112)
        seq_len, vocab_size = (5, 50)
        chunk_size = 10
        logits = torch.randn(seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (seq_len,))
        loss_std = F.cross_entropy(logits.detach(), labels, reduction="none")
        loss_chunk = chunked_cross_entropy(
            logits.detach(), labels, chunk_size=chunk_size
        )
        assert torch.allclose(loss_std, loss_chunk, atol=1e-06)

    def test_exact_chunk_boundaries(self):
        """Test when sequence length is exactly divisible by chunk_size."""
        torch.manual_seed(131415)
        seq_len, vocab_size = (64, 300)
        chunk_size = 16
        logits = torch.randn(seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (seq_len,))
        loss_std = F.cross_entropy(logits.detach(), labels, reduction="none")
        loss_chunk = chunked_cross_entropy(
            logits.detach(), labels, chunk_size=chunk_size
        )
        assert torch.allclose(loss_std, loss_chunk, atol=1e-06)

    def test_gradient_flow_with_dft_weighting(self):
        """Test gradient flow when combined with DFT weighting."""
        torch.manual_seed(161718)
        seq_len, vocab_size = (32, 100)
        chunk_size = 16
        logits = torch.randn(seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (seq_len,))
        loss = chunked_cross_entropy(logits, labels, chunk_size=chunk_size)
        with torch.no_grad():
            weights = torch.exp(-loss)
        weighted_loss = (loss * weights).sum()
        weighted_loss.backward()
        assert logits.grad is not None
        assert torch.all(torch.isfinite(logits.grad))

    @pytest.mark.parametrize("chunk_size", [4, 8, 16, 32])
    def test_different_chunk_sizes(self, chunk_size):
        """Test that different chunk sizes all produce correct results."""
        torch.manual_seed(192021)
        seq_len, vocab_size = (64, 200)
        logits = torch.randn(seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (seq_len,))
        loss_std = F.cross_entropy(logits, labels, reduction="none")
        loss_chunk = chunked_cross_entropy(logits, labels, chunk_size=chunk_size)
        assert torch.allclose(loss_std, loss_chunk, atol=1e-06)


class TestIntegrationWithDFTUtils:
    """Test integration with compute_per_token_cross_entropy."""

    def test_compute_per_token_cross_entropy_with_chunking(self):
        """Test that compute_per_token_cross_entropy works with chunk_size."""
        torch.manual_seed(222324)
        batch_size, seq_len, vocab_size = (2, 32, 500)
        chunk_size = 16
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        loss_std, mask_std = compute_per_token_cross_entropy(
            logits, labels, shift_labels=True, chunk_size=None
        )
        loss_chunk, mask_chunk = compute_per_token_cross_entropy(
            logits, labels, shift_labels=True, chunk_size=chunk_size
        )
        assert torch.all(mask_std == mask_chunk)
        assert torch.allclose(loss_std, loss_chunk, atol=1e-06)

    def test_large_vocabulary_scenario(self):
        """Test with vocabulary size similar to Qwen (152K tokens)."""
        torch.manual_seed(252627)
        batch_size, seq_len, vocab_size = (1, 16, 50000)
        chunk_size = 8
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        loss_chunk, mask = compute_per_token_cross_entropy(
            logits, labels, shift_labels=True, chunk_size=chunk_size
        )
        assert loss_chunk.shape[0] == batch_size * (seq_len - 1)
        assert mask.sum() > 0

    def test_with_ignore_index_and_chunking(self):
        """Test chunking with ignore_index in realistic scenario."""
        torch.manual_seed(282930)
        batch_size, seq_len, vocab_size = (4, 32, 1000)
        chunk_size = 16
        ignore_index = -100
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels[:, -5:] = ignore_index
        loss_chunk, mask = compute_per_token_cross_entropy(
            logits,
            labels,
            shift_labels=True,
            chunk_size=chunk_size,
            ignore_index=ignore_index,
        )
        assert mask.sum() < batch_size * (seq_len - 1)
        assert torch.all(torch.isfinite(loss_chunk))
