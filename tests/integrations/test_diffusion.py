"""Tests for diffusion trainer integration."""

# pylint: disable=redefined-outer-name,protected-access

from unittest.mock import Mock

import pytest
import torch

from axolotl.integrations.diffusion import DiffusionTrainer
from axolotl.integrations.diffusion.utils import create_bidirectional_attention_mask
from axolotl.utils.dict import DictDefault


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = Mock()
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.pad_token_id = 0
    return tokenizer


@pytest.fixture
def diffusion_config():
    """Create a diffusion config."""
    return DictDefault(
        {
            "diffusion": {
                "mask_token_id": 32000,
                "eps": 1e-3,
                "importance_weighting": False,
            },
            "sample_packing": False,
        }
    )


@pytest.fixture
def diffusion_trainer_instance(mock_tokenizer, diffusion_config):
    """Create a diffusion trainer instance for testing methods directly."""
    # Create a minimal trainer instance just for testing methods
    trainer = object.__new__(DiffusionTrainer)  # Bypass __init__
    trainer.cfg = diffusion_config
    trainer._special_token_ids = {0, 1, 2}  # pad, bos, eos
    trainer.processing_class = mock_tokenizer
    trainer.store_metrics = Mock()  # Mock metrics storage
    return trainer


class TestDiffusionTrainer:
    """Test the DiffusionTrainer class."""

    def test_forward_process_basic(self, diffusion_trainer_instance):
        """Test basic forward process without labels."""
        input_ids = torch.tensor([[1, 10, 20, 30, 2]], dtype=torch.long)

        noisy_batch, masked_indices, p_mask = (
            diffusion_trainer_instance._forward_process(input_ids, eps=0.1)
        )

        # Check shapes
        assert noisy_batch.shape == input_ids.shape
        assert masked_indices.shape == input_ids.shape
        assert p_mask.shape == input_ids.shape

        # Check that special tokens are not masked
        special_token_positions = (input_ids == 1) | (input_ids == 2) | (input_ids == 0)
        assert not masked_indices[special_token_positions].any()

        # Check that mask token is applied
        mask_token_id = diffusion_trainer_instance.cfg.diffusion.mask_token_id
        masked_positions = masked_indices
        if masked_positions.any():
            assert (noisy_batch[masked_positions] == mask_token_id).all()

    def test_forward_process_with_labels(self, diffusion_trainer_instance):
        """Test forward process with SFT labels."""
        input_ids = torch.tensor([[1, 10, 20, 30, 2]], dtype=torch.long)
        labels = torch.tensor([[-100, -100, 20, 30, 2]], dtype=torch.long)

        noisy_batch, masked_indices, p_mask = (
            diffusion_trainer_instance._forward_process(
                input_ids, labels=labels, eps=0.1
            )
        )

        # Check shapes
        assert noisy_batch.shape == input_ids.shape
        assert masked_indices.shape == input_ids.shape
        assert p_mask.shape == input_ids.shape

        # Check that only answer tokens can be masked (where labels != -100)
        non_answer_mask = labels == -100

        # No masking should occur on non-answer tokens
        assert not masked_indices[non_answer_mask].any()

        # p_mask should be the same for all positions (sampled timestep),
        # but masking is only applied to answer tokens
        assert p_mask.shape == input_ids.shape
        # Verify that masked_indices respects the answer mask
        assert not masked_indices[non_answer_mask].any()

    def test_forward_process_with_attention_mask(self, diffusion_trainer_instance):
        """Test forward process with attention mask."""
        input_ids = torch.tensor([[1, 10, 20, 0]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)

        _, masked_indices, p_mask = diffusion_trainer_instance._forward_process(
            input_ids, attention_mask=attention_mask, eps=0.1
        )

        # Check that padding tokens are not masked
        padding_positions = attention_mask == 0
        assert not masked_indices[padding_positions].any()
        assert (p_mask[padding_positions] == 0).all()

    def test_bidirectional_attention_mask_no_packing(self, diffusion_trainer_instance):
        """Test bidirectional attention mask without sample packing."""
        input_ids = torch.tensor([[1, 10, 20, 2]], dtype=torch.long)

        mask = create_bidirectional_attention_mask(input_ids)

        # Should be all-to-all attention
        expected_shape = (1, 1, 4, 4)
        assert mask.shape == expected_shape
        assert mask.all()

    def test_bidirectional_attention_mask_with_packing(
        self, diffusion_trainer_instance
    ):
        """Test bidirectional attention mask with sample packing."""
        diffusion_trainer_instance.cfg.sample_packing = True
        input_ids = torch.tensor([[1, 10, 20, 30, 40, 2]], dtype=torch.long)
        # Sample IDs: first sample (1), second sample (2)
        attention_mask = torch.tensor([[1, 1, 1, 2, 2, 2]], dtype=torch.long)

        mask = create_bidirectional_attention_mask(
            input_ids, attention_mask, sample_packing=True
        )

        # Check that tokens within same sample can attend to each other
        # but not across samples
        assert mask[0, 0, 0, 1].item()  # First sample tokens can attend to each other
        assert mask[0, 0, 1, 2].item()
        assert not mask[0, 0, 0, 3].item()  # Can't attend across samples
        assert not mask[0, 0, 2, 4].item()
        assert mask[0, 0, 3, 4].item()  # Second sample tokens can attend to each other

    def test_compute_loss_basic(self, diffusion_trainer_instance):
        """Test basic loss computation."""
        # Mock model that returns logits
        mock_model = Mock()
        mock_outputs = Mock()
        vocab_size = 1000
        seq_len = 5
        mock_outputs.logits = torch.randn(1, seq_len, vocab_size, requires_grad=True)
        mock_model.return_value = mock_outputs
        mock_model.training = True

        input_ids = torch.tensor([[1, 10, 20, 30, 2]], dtype=torch.long)

        loss, outputs = diffusion_trainer_instance._compute_diffusion_loss(
            mock_model, input_ids
        )

        # Check that loss is computed
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert outputs == mock_outputs

        # Check that metrics were stored
        diffusion_trainer_instance.store_metrics.assert_called_once()

    def test_compute_loss_sft(self, diffusion_trainer_instance):
        """Test loss computation with SFT labels."""
        # Mock model
        mock_model = Mock()
        mock_outputs = Mock()
        vocab_size = 1000
        seq_len = 5
        mock_outputs.logits = torch.randn(1, seq_len, vocab_size, requires_grad=True)
        mock_model.return_value = mock_outputs
        mock_model.training = True
        diffusion_trainer_instance.cfg.datasets = Mock()

        input_ids = torch.tensor([[1, 10, 20, 30, 2]], dtype=torch.long)
        labels = torch.tensor([[-100, -100, 20, 30, 2]], dtype=torch.long)

        loss, _ = diffusion_trainer_instance._compute_diffusion_loss(
            mock_model, input_ids, labels=labels
        )

        # Check that loss is computed
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad

        # Check that SFT metrics were added
        call_args = diffusion_trainer_instance.store_metrics.call_args[0][0]
        assert "answer_ratio" in call_args
        assert "avg_answer_length" in call_args

    def test_compute_loss_no_masked_tokens(self, diffusion_trainer_instance):
        """Test loss computation when no tokens are masked."""
        # Mock model
        mock_model = Mock()
        mock_outputs = Mock()
        vocab_size = 1000
        seq_len = 3
        mock_outputs.logits = torch.randn(1, seq_len, vocab_size)
        mock_model.return_value = mock_outputs
        mock_model.training = True

        # Only special tokens (which won't be masked)
        input_ids = torch.tensor([[1, 0, 2]], dtype=torch.long)

        loss, _ = diffusion_trainer_instance._compute_diffusion_loss(
            mock_model, input_ids
        )

        # Loss should be zero when no tokens are masked
        assert loss.item() == 0.0
        assert loss.requires_grad

    def test_cache_special_token_ids(self, mock_tokenizer):
        """Test caching of special token IDs."""
        trainer = object.__new__(DiffusionTrainer)
        trainer.processing_class = mock_tokenizer
        trainer._cache_special_token_ids()
        assert trainer._special_token_ids == {0, 1, 2}

    def test_cache_special_token_ids_no_tokenizer(self):
        """Test caching when no tokenizer is available."""
        trainer = object.__new__(DiffusionTrainer)
        trainer.processing_class = None
        trainer._cache_special_token_ids()

        assert trainer._special_token_ids == set()

    def test_main_compute_loss_interface(self, diffusion_trainer_instance):
        """Test the main compute_loss interface."""
        # Mock model
        mock_model = Mock()
        mock_outputs = Mock()
        mock_outputs.logits = torch.randn(1, 5, 1000)
        mock_model.return_value = mock_outputs
        mock_model.training = True

        inputs = {
            "input_ids": torch.tensor([[1, 10, 20, 30, 2]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.long),
            "labels": torch.tensor([[-100, -100, 20, 30, 2]], dtype=torch.long),
        }

        # Test without return_outputs
        loss = diffusion_trainer_instance.compute_loss(mock_model, inputs)
        assert isinstance(loss, torch.Tensor)

        # Test with return_outputs
        loss, outputs = diffusion_trainer_instance.compute_loss(
            mock_model, inputs, return_outputs=True
        )
        assert isinstance(loss, torch.Tensor)
        assert outputs == mock_outputs

    def test_missing_input_ids_raises_error(self, diffusion_trainer_instance):
        """Test that missing input_ids raises ValueError."""
        mock_model = Mock()
        inputs = {"attention_mask": torch.tensor([[1, 1, 1]])}

        with pytest.raises(ValueError, match="input_ids is required"):
            diffusion_trainer_instance.compute_loss(mock_model, inputs)
