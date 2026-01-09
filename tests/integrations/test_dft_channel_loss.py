"""Integration tests for DFT + Channel Loss compatibility.

This module tests the integration between DFT loss and Channel Loss tracking,
verifying that intermediate values (per_token_loss, valid_mask) are correctly
attached to model outputs when enable_dft_channel_loss is enabled.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from axolotl.integrations.dft.patch import patch_compute_loss_for_dft


class TestDFTChannelLossIntegration:
    """Test DFT integration with Channel Loss plugin."""

    def test_channel_loss_enabled_attaches_intermediate_values(self):
        """Test that DFT provides per-token loss when channel loss is enabled."""
        # Setup trainer with both DFT and Channel Loss enabled
        trainer = MagicMock()
        trainer.args = SimpleNamespace(
            enable_dft_loss=True,
            enable_dft_channel_loss=True,  # Enable channel loss integration
            dft_chunk_size=None,
            include_tkps=False,
            label_smoothing_factor=0.0,
            orpo_alpha=None,
        )
        trainer.state = SimpleNamespace()
        trainer.compute_loss = MagicMock()

        # Apply DFT patch
        patch_compute_loss_for_dft(trainer, cfg=MagicMock())

        # Create test data
        batch_size, seq_len, vocab_size = 2, 10, 1000
        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        class DummyModel(torch.nn.Module):
            def forward(self, **kwargs):
                return SimpleNamespace(logits=logits)

        model = DummyModel()
        inputs = {"input_ids": torch.zeros(batch_size, seq_len), "labels": labels}

        # Compute loss
        loss, outputs = trainer.compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=None
        )

        # Verify intermediate values are attached
        assert hasattr(outputs, "per_token_loss"), "per_token_loss not attached"
        assert hasattr(outputs, "valid_mask"), "valid_mask not attached"
        assert hasattr(outputs, "loss"), "loss not attached"

        # Verify shapes
        expected_tokens = batch_size * (seq_len - 1)  # Due to shift
        assert outputs.per_token_loss.shape == (expected_tokens,), (
            f"per_token_loss shape {outputs.per_token_loss.shape} != {(expected_tokens,)}"
        )
        assert outputs.valid_mask.shape == (expected_tokens,), (
            f"valid_mask shape {outputs.valid_mask.shape} != {(expected_tokens,)}"
        )
        assert outputs.valid_mask.dtype == torch.bool, "valid_mask should be boolean"
        assert loss.ndim == 0, "loss should be scalar"

        # Verify loss is the same as outputs.loss
        assert torch.allclose(loss, outputs.loss), "loss != outputs.loss"

    def test_channel_loss_disabled_no_intermediate_values(self):
        """Test backward compatibility: no intermediate values when flag is False."""
        # Setup trainer with DFT but WITHOUT Channel Loss
        trainer = MagicMock()
        trainer.args = SimpleNamespace(
            enable_dft_loss=True,
            enable_dft_channel_loss=False,  # Channel loss NOT enabled
            dft_chunk_size=None,
            include_tkps=False,
            label_smoothing_factor=0.0,
            orpo_alpha=None,
        )
        trainer.state = SimpleNamespace()
        trainer.compute_loss = MagicMock()

        # Apply DFT patch
        patch_compute_loss_for_dft(trainer, cfg=MagicMock())

        # Create test data
        batch_size, seq_len, vocab_size = 2, 10, 1000
        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        class DummyModel(torch.nn.Module):
            def forward(self, **kwargs):
                return SimpleNamespace(logits=logits)

        model = DummyModel()
        inputs = {"input_ids": torch.zeros(batch_size, seq_len), "labels": labels}

        # Compute loss
        loss, outputs = trainer.compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=None
        )

        # Verify intermediate values are NOT attached (backward compatibility)
        assert not hasattr(outputs, "per_token_loss"), (
            "per_token_loss should not be attached when enable_dft_channel_loss=False"
        )
        assert not hasattr(outputs, "valid_mask"), (
            "valid_mask should not be attached when enable_dft_channel_loss=False"
        )

        # Loss should still be valid
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

        batch_size, seq_len, vocab_size = 2, 10, 100
        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Set some labels to -100 (ignore_index)
        labels[0, 3:6] = -100  # Ignore tokens 3-5 in first batch
        labels[1, 7:] = -100  # Ignore tokens 7-9 in second batch

        class DummyModel(torch.nn.Module):
            def forward(self, **kwargs):
                return SimpleNamespace(logits=logits)

        model = DummyModel()
        inputs = {"input_ids": torch.zeros(batch_size, seq_len), "labels": labels}

        loss, outputs = trainer.compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=None
        )

        # Verify valid_mask correctly identifies non-ignored tokens
        # After shift: labels become [..., 1:].flatten()
        shifted_labels = labels[:, 1:].flatten()
        expected_valid = shifted_labels != -100

        assert torch.equal(outputs.valid_mask, expected_valid), (
            "valid_mask should match non-ignored tokens"
        )

        # Verify per_token_loss only includes valid tokens in reduction
        valid_count = outputs.valid_mask.sum().item()
        assert valid_count < batch_size * (seq_len - 1), "Some tokens should be ignored"
        assert valid_count > 0, "Should have some valid tokens"

    def test_channel_loss_with_chunked_ce(self):
        """Test DFT + Channel Loss works with chunked cross-entropy."""
        trainer = MagicMock()
        trainer.args = SimpleNamespace(
            enable_dft_loss=True,
            enable_dft_channel_loss=True,
            dft_chunk_size=4,  # Enable chunked CE
            include_tkps=False,
            label_smoothing_factor=0.0,
            orpo_alpha=None,
        )
        trainer.state = SimpleNamespace()
        trainer.compute_loss = MagicMock()

        patch_compute_loss_for_dft(trainer, cfg=MagicMock())

        batch_size, seq_len, vocab_size = 2, 10, 1000
        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        class DummyModel(torch.nn.Module):
            def forward(self, **kwargs):
                return SimpleNamespace(logits=logits)

        model = DummyModel()
        inputs = {"input_ids": torch.zeros(batch_size, seq_len), "labels": labels}

        # Compute loss with chunked CE
        loss, outputs = trainer.compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=None
        )

        # Verify intermediate values are still attached
        assert hasattr(outputs, "per_token_loss"), "per_token_loss not attached"
        assert hasattr(outputs, "valid_mask"), "valid_mask not attached"

        # Verify shapes
        expected_tokens = batch_size * (seq_len - 1)
        assert outputs.per_token_loss.shape == (expected_tokens,)
        assert outputs.valid_mask.shape == (expected_tokens,)

        # Loss should be valid
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

        batch_size, seq_len, vocab_size = 4, 8, 100
        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Simulate channel IDs (e.g., 2 channels)
        # Channel 0: batches 0, 1; Channel 1: batches 2, 3
        channel_ids = torch.tensor([0, 0, 1, 1])

        class DummyModel(torch.nn.Module):
            def forward(self, **kwargs):
                return SimpleNamespace(logits=logits)

        model = DummyModel()
        inputs = {"input_ids": torch.zeros(batch_size, seq_len), "labels": labels}

        loss, outputs = trainer.compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=None
        )

        # Simulate Channel Loss plugin logic
        per_token_loss = outputs.per_token_loss
        valid_mask = outputs.valid_mask

        # Expand channel_ids to match per-token shape (after shift)
        # Each batch item contributes (seq_len - 1) tokens
        channel_ids_per_token = channel_ids.repeat_interleave(seq_len - 1)

        # Compute per-channel statistics
        channel_losses = {}
        for channel_id in [0, 1]:
            channel_mask = channel_ids_per_token == channel_id
            combined_mask = channel_mask & valid_mask

            if combined_mask.sum() > 0:
                channel_loss = per_token_loss[combined_mask].mean()
                channel_losses[channel_id] = channel_loss.item()

        # Verify we got statistics for both channels
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

        batch_size, seq_len, vocab_size = 2, 10, 100
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

        # Verify loss requires grad
        assert loss.requires_grad, "loss should require gradients"

        # Compute gradients
        loss.backward()

        # Verify gradients were computed
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

        batch_size, seq_len, vocab_size = 2, 10, 100
        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        class DummyModel(torch.nn.Module):
            def forward(self, **kwargs):
                return SimpleNamespace(logits=logits)

        model = DummyModel()
        inputs = {"input_ids": torch.zeros(batch_size, seq_len), "labels": labels}

        # Compute loss WITHOUT return_outputs
        result = trainer.compute_loss(
            model, inputs, return_outputs=False, num_items_in_batch=None
        )

        # Should return only loss (not tuple)
        assert isinstance(result, torch.Tensor), (
            "Should return tensor when return_outputs=False"
        )
        assert result.ndim == 0, "Should return scalar loss"
        assert result.requires_grad, "Loss should have gradients"
