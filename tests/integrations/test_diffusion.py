"""Tests for diffusion model integration."""

# pylint: disable=redefined-outer-name,protected-access

from unittest.mock import Mock, patch

import pytest
import torch

from axolotl.integrations.diffusion.configuration import LlamaForDiffusionConfig
from axolotl.integrations.diffusion.models import LlamaForDiffusionLM
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
    return LlamaForDiffusionConfig(
        mask_token_id=32000,
        eps=1e-3,
        importance_weighting=False,
        sample_packing=False,
        # Basic llama config fields - smaller for testing
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
    )


@pytest.fixture
def diffusion_model_instance(mock_tokenizer, diffusion_config):
    """Create a diffusion model instance for testing methods directly."""
    # Create a minimal model instance for testing
    model = object.__new__(LlamaForDiffusionLM)
    model.config = diffusion_config
    model._special_token_ids = {0, 1, 2}  # pad, bos, eos
    model.training = True
    
    # Set tokenizer
    model.set_tokenizer(mock_tokenizer)
    
    return model


class TestDiffusionModel:
    """Test the DiffusionModel class."""

    def test_forward_process_basic(self, diffusion_model_instance):
        """Test basic forward process without labels."""
        input_ids = torch.tensor([[1, 10, 20, 30, 2]], dtype=torch.long)

        noisy_batch, masked_indices, p_mask = (
            diffusion_model_instance._forward_process(input_ids, eps=0.1)
        )

        # Check shapes
        assert noisy_batch.shape == input_ids.shape
        assert masked_indices.shape == input_ids.shape
        assert p_mask.shape == input_ids.shape

        # Check that special tokens are not masked
        special_token_positions = (input_ids == 1) | (input_ids == 2) | (input_ids == 0)
        assert not masked_indices[special_token_positions].any()

        # Check that mask token is applied
        mask_token_id = diffusion_model_instance.config.mask_token_id
        masked_positions = masked_indices
        if masked_positions.any():
            assert (noisy_batch[masked_positions] == mask_token_id).all()

    def test_forward_process_with_labels(self, diffusion_model_instance):
        """Test forward process with SFT labels."""
        input_ids = torch.tensor([[1, 10, 20, 30, 2]], dtype=torch.long)
        labels = torch.tensor([[-100, -100, 20, 30, 2]], dtype=torch.long)

        noisy_batch, masked_indices, p_mask = (
            diffusion_model_instance._forward_process(
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

    def test_forward_process_with_attention_mask(self, diffusion_model_instance):
        """Test forward process with attention mask."""
        input_ids = torch.tensor([[1, 10, 20, 0]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)

        _, masked_indices, p_mask = diffusion_model_instance._forward_process(
            input_ids, attention_mask=attention_mask, eps=0.1
        )

        # Check that padding tokens are not masked
        padding_positions = attention_mask == 0
        assert not masked_indices[padding_positions].any()
        assert (p_mask[padding_positions] == 0).all()

    def test_bidirectional_attention_mask_no_packing(self, diffusion_model_instance):
        """Test bidirectional attention mask without sample packing."""
        input_ids = torch.tensor([[1, 10, 20, 2]], dtype=torch.long)

        mask = diffusion_model_instance._create_bidirectional_attention_mask(
            input_ids
        )

        # Should be all-to-all attention
        expected_shape = (1, 1, 4, 4)
        assert mask.shape == expected_shape
        assert mask.all()

    def test_bidirectional_attention_mask_with_packing(
        self, diffusion_model_instance
    ):
        """Test bidirectional attention mask with sample packing."""
        diffusion_model_instance.config.sample_packing = True
        input_ids = torch.tensor([[1, 10, 20, 30, 40, 2]], dtype=torch.long)
        # Sample IDs: first sample (1), second sample (2)
        attention_mask = torch.tensor([[1, 1, 1, 2, 2, 2]], dtype=torch.long)

        mask = diffusion_model_instance._create_bidirectional_attention_mask(
            input_ids, attention_mask
        )

        # Check that tokens within same sample can attend to each other
        # but not across samples
        assert mask[0, 0, 0, 1].item()  # First sample tokens can attend to each other
        assert mask[0, 0, 1, 2].item()
        assert not mask[0, 0, 0, 3].item()  # Can't attend across samples
        assert not mask[0, 0, 2, 4].item()
        assert mask[0, 0, 3, 4].item()  # Second sample tokens can attend to each other

    def test_compute_loss_basic(self, diffusion_model_instance):
        """Test basic loss computation."""
        input_ids = torch.tensor([[1, 10, 20, 30, 2]], dtype=torch.long)
        
        # Create mock data for loss computation
        vocab_size = 1000
        seq_len = 5
        logits = torch.randn(1, seq_len, vocab_size, requires_grad=True)
        
        # Create a simple masked indices tensor (mask middle tokens)
        masked_indices = torch.tensor([[False, True, True, False, False]], dtype=torch.bool)
        p_mask = torch.tensor([[0.1, 0.5, 0.5, 0.1, 0.1]], dtype=torch.float)

        loss = diffusion_model_instance._compute_diffusion_loss(
            input_ids=input_ids,
            logits=logits,
            masked_indices=masked_indices,
            p_mask=p_mask,
        )

        # Check that loss is computed
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad

    def test_compute_loss_with_labels(self, diffusion_model_instance):
        """Test loss computation with SFT labels."""
        input_ids = torch.tensor([[1, 10, 20, 30, 2]], dtype=torch.long)
        labels = torch.tensor([[-100, -100, 20, 30, 2]], dtype=torch.long)
        
        # Create mock data for loss computation
        vocab_size = 1000
        seq_len = 5
        logits = torch.randn(1, seq_len, vocab_size, requires_grad=True)
        
        # Create masked indices that only covers answer tokens
        masked_indices = torch.tensor([[False, False, True, True, False]], dtype=torch.bool)
        p_mask = torch.tensor([[0.1, 0.1, 0.5, 0.5, 0.1]], dtype=torch.float)

        loss = diffusion_model_instance._compute_diffusion_loss(
            input_ids=input_ids,
            labels=labels,
            logits=logits,
            masked_indices=masked_indices,
            p_mask=p_mask,
        )

        # Check that loss is computed
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad

    def test_compute_loss_no_masked_tokens(self, diffusion_model_instance):
        """Test loss computation when no tokens are masked."""
        input_ids = torch.tensor([[1, 0, 2]], dtype=torch.long)
        
        # Create mock data for loss computation
        vocab_size = 1000
        seq_len = 3
        logits = torch.randn(1, seq_len, vocab_size)
        
        # No tokens masked
        masked_indices = torch.tensor([[False, False, False]], dtype=torch.bool)
        p_mask = torch.tensor([[0.1, 0.1, 0.1]], dtype=torch.float)

        loss = diffusion_model_instance._compute_diffusion_loss(
            input_ids=input_ids,
            logits=logits,
            masked_indices=masked_indices,
            p_mask=p_mask,
        )

        # Loss should be zero when no tokens are masked
        assert loss.item() == 0.0
        assert loss.requires_grad

    def test_cache_special_token_ids(self, diffusion_model_instance):
        """Test caching of special token IDs."""
        # Should cache BOS, EOS, PAD tokens
        expected_tokens = {0, 1, 2}  # pad, bos, eos
        assert diffusion_model_instance._special_token_ids == expected_tokens

    def test_cache_special_token_ids_no_tokenizer(self):
        """Test caching when no tokenizer is available."""
        # Mock the parent model initialization to avoid loading pretrained weights
        with patch('transformers.models.llama.modeling_llama.LlamaForCausalLM.__init__'):
            model = LlamaForDiffusionLM.__new__(LlamaForDiffusionLM)
            model._cache_special_token_ids(None)
            assert model._special_token_ids == set()

    def test_forward_training_mode(self, diffusion_model_instance):
        """Test forward pass in training mode."""
        input_ids = torch.tensor([[1, 10, 20, 30, 2]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.bool)
        
        # Mock the parent forward method
        with patch.object(diffusion_model_instance.__class__.__bases__[1], 'forward') as mock_forward:
            mock_output = Mock()
            mock_output.logits = torch.randn(1, 5, 32000)
            mock_forward.return_value = mock_output
            
            # Set training mode
            diffusion_model_instance.training = True
            
            result = diffusion_model_instance.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            # Should call parent forward and compute loss
            assert mock_forward.called
            assert hasattr(result, 'loss')

    def test_forward_inference_mode(self, diffusion_model_instance):
        """Test forward pass in inference mode."""
        input_ids = torch.tensor([[1, 10, 20, 30, 2]], dtype=torch.long)
        
        # Mock the parent forward method
        with patch.object(diffusion_model_instance.__class__.__bases__[1], 'forward') as mock_forward:
            mock_output = Mock()
            mock_forward.return_value = mock_output
            
            # Set inference mode
            diffusion_model_instance.training = False
            
            result = diffusion_model_instance.forward(
                input_ids=input_ids,
                return_dict=True
            )
            
            # Should just call parent forward without diffusion processing
            assert mock_forward.called
            assert result == mock_output
