"""Tests for diffusion trainer integration."""

# pylint: disable=redefined-outer-name,protected-access

from unittest.mock import Mock, patch

import pytest
import torch

from axolotl.integrations.diffusion.args import DiffusionArgs
from axolotl.integrations.diffusion.loss import (
    ForDiffusionLMLoss,
    register_diffusion_loss,
)
from axolotl.integrations.diffusion.model_patch import (
    _create_bidirectional_attention_mask,
    _forward_process,
    patch_model_for_bidirectional_attention,
)
from axolotl.integrations.diffusion.plugin import DiffusionPlugin


@pytest.fixture
def diffusion_config():
    """Create a diffusion config."""
    return DiffusionArgs(
        eps=1e-3,
        importance_weighting=False,
        mask_token_id=32000,
        generate_samples=False,
    )


@pytest.fixture
def mock_model():
    """Create a mock model."""
    model = Mock()
    model.config = Mock()
    model.config.loss_type = "ForDiffusionLM"
    model.config.diffusion_config = {
        "eps": 1e-3,
        "importance_weighting": False,
        "mask_token_id": 32000,
    }
    model.training = True
    return model


class TestDiffusionLoss:
    """Test the ForDiffusionLMLoss function."""

    def test_loss_with_diffusion_info(self, mock_model):
        """Test loss computation with stored diffusion info."""
        batch_size, seq_len, vocab_size = 1, 5, 1000

        # Mock stored diffusion info
        original_input_ids = torch.tensor([[1, 10, 20, 30, 2]], dtype=torch.long)
        masked_indices = torch.tensor(
            [[False, True, True, False, False]], dtype=torch.bool
        )
        p_mask = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5]], dtype=torch.float)

        mock_model._diffusion_info = {
            "original_input_ids": original_input_ids,
            "masked_indices": masked_indices,
            "p_mask": p_mask,
        }

        # Mock logits
        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        labels = torch.tensor([[-100, -100, 20, 30, 2]], dtype=torch.long)

        loss = ForDiffusionLMLoss(
            logits=logits,
            labels=labels,
            vocab_size=vocab_size,
            config=mock_model.config,
            model=mock_model,
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() >= 0

    def test_loss_fallback_without_diffusion_info(self, mock_model):
        """Test fallback to causal LM loss when no diffusion info."""
        batch_size, seq_len, vocab_size = 1, 5, 1000

        # Remove diffusion info to trigger fallback
        if hasattr(mock_model, "_diffusion_info"):
            delattr(mock_model, "_diffusion_info")

        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        labels = torch.tensor([[1, 10, 20, 30, 2]], dtype=torch.long)

        loss = ForDiffusionLMLoss(
            logits=logits,
            labels=labels,
            vocab_size=vocab_size,
            config=mock_model.config,
            model=mock_model,
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad

    def test_loss_no_masked_tokens(self, mock_model):
        """Test loss when no tokens are masked."""
        batch_size, seq_len, vocab_size = 1, 3, 1000

        # No masked tokens
        original_input_ids = torch.tensor([[1, 10, 2]], dtype=torch.long)
        masked_indices = torch.tensor([[False, False, False]], dtype=torch.bool)
        p_mask = torch.tensor([[0.1, 0.1, 0.1]], dtype=torch.float)

        mock_model._diffusion_info = {
            "original_input_ids": original_input_ids,
            "masked_indices": masked_indices,
            "p_mask": p_mask,
        }

        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.tensor([[1, 10, 2]], dtype=torch.long)

        loss = ForDiffusionLMLoss(
            logits=logits,
            labels=labels,
            vocab_size=vocab_size,
            config=mock_model.config,
            model=mock_model,
        )

        assert loss.item() == 0.0


class TestModelPatch:
    """Test the model patching functionality."""

    def test_forward_process_basic(self):
        """Test basic forward process."""
        input_ids = torch.tensor([[1, 10, 20, 30, 2]], dtype=torch.long)
        diffusion_config = {"eps": 0.1, "mask_token_id": 32000}

        noisy_input_ids, masked_indices, p_mask = _forward_process(
            input_ids, diffusion_config=diffusion_config
        )

        # Check shapes
        assert noisy_input_ids.shape == input_ids.shape
        assert masked_indices.shape == input_ids.shape
        assert p_mask.shape == input_ids.shape

        # Check that mask token is applied where masked
        if masked_indices.any():
            assert (noisy_input_ids[masked_indices] == 32000).all()

    def test_forward_process_with_labels(self):
        """Test forward process with SFT labels."""
        input_ids = torch.tensor([[1, 10, 20, 30, 2]], dtype=torch.long)
        labels = torch.tensor([[-100, -100, 20, 30, 2]], dtype=torch.long)
        diffusion_config = {"eps": 0.1, "mask_token_id": 32000}

        _, masked_indices, _ = _forward_process(
            input_ids, labels=labels, diffusion_config=diffusion_config
        )

        # Check that only answer tokens can be masked (where labels != -100)
        non_answer_mask = labels == -100
        assert not masked_indices[non_answer_mask].any()

    def test_forward_process_with_attention_mask(self):
        """Test forward process with attention mask."""
        input_ids = torch.tensor([[1, 10, 20, 0]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)
        diffusion_config = {"eps": 0.1, "mask_token_id": 32000}

        _, masked_indices, p_mask = _forward_process(
            input_ids, attention_mask=attention_mask, diffusion_config=diffusion_config
        )

        # Check that padding tokens are not masked
        padding_positions = attention_mask == 0
        assert not masked_indices[padding_positions].any()
        assert (p_mask[padding_positions] == 0).all()

    def test_bidirectional_attention_mask(self):
        """Test bidirectional attention mask creation."""
        input_ids = torch.tensor([[1, 10, 20, 2]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.long)

        mask = _create_bidirectional_attention_mask(input_ids, attention_mask)

        # Should be all-to-all attention
        expected_shape = (1, 1, 4, 4)
        assert mask.shape == expected_shape
        assert mask.all()

    def test_bidirectional_attention_mask_with_padding(self):
        """Test bidirectional attention mask with padding."""
        input_ids = torch.tensor([[1, 10, 20, 0]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)

        mask = _create_bidirectional_attention_mask(input_ids, attention_mask)

        # Padding positions should not attend or be attended to
        assert not mask[0, 0, 3, :].any()  # Padding can't attend to anything
        assert not mask[0, 0, :, 3].any()  # Nothing can attend to padding

    def test_patch_model_for_bidirectional_attention(self):
        """Test that model patching works."""
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.loss_type = "ForDiffusionLM"
        mock_model.config.diffusion_config = {"eps": 1e-3, "mask_token_id": 32000}
        mock_model.training = True

        original_forward = Mock()
        mock_model.forward = original_forward

        # Patch the model
        patch_model_for_bidirectional_attention(mock_model)

        # Check that forward method was replaced
        assert mock_model.forward != original_forward


class TestDiffusionPlugin:
    """Test the DiffusionPlugin."""

    def test_plugin_registers_loss_function(self):
        """Test that plugin registers diffusion loss function."""
        with patch('axolotl.integrations.diffusion.plugin.register_diffusion_loss', return_value=True) as mock_register:
            plugin = DiffusionPlugin()
            mock_register.assert_called_once()

    def test_post_model_load_configuration(self):
        """Test that post_model_load configures model correctly."""
        plugin = DiffusionPlugin()
        
        # Mock model and config
        mock_model = Mock()
        mock_model.config = Mock()
        mock_cfg = Mock()
        mock_cfg.eps = 1e-3
        mock_cfg.importance_weighting = True
        mock_cfg.mask_token_id = 32000
        
        with patch('axolotl.integrations.diffusion.plugin.patch_model_for_bidirectional_attention') as mock_patch:
            result = plugin.post_model_load(mock_cfg, mock_model)
            
            # Check model configuration
            assert mock_model.config.loss_type == "ForDiffusionLM"
            assert mock_model.config.diffusion_config is not None
            assert mock_model.config.diffusion_config['eps'] == 1e-3
            
            # Check model was patched
            mock_patch.assert_called_once_with(mock_model)
            
            # Should return the model
            assert result == mock_model

    def test_post_trainer_create_stores_config(self, diffusion_config):
        """Test that post_trainer_create stores config on trainer."""
        plugin = DiffusionPlugin()
        mock_trainer = Mock()
        mock_cfg = Mock()
        
        # Set config attributes
        for attr, value in diffusion_config.model_dump().items():
            setattr(mock_cfg, attr, value)
        
        plugin.post_trainer_create(mock_cfg, mock_trainer)
        
        # Check that diffusion config was stored on trainer
        assert hasattr(mock_trainer, 'diffusion_config')
        assert mock_trainer.diffusion_config.eps == diffusion_config.eps

    def test_add_callbacks_post_trainer_with_generation_enabled(self):
        """Test callback addition when generation is enabled."""
        plugin = DiffusionPlugin()
        mock_trainer = Mock()
        mock_cfg = Mock()
        
        # Mock trainer with diffusion config that has generation enabled
        mock_trainer.diffusion_config = DiffusionArgs(generate_samples=True)
        
        with patch('axolotl.integrations.diffusion.plugin.DiffusionGenerationCallback') as mock_callback_class:
            callbacks = plugin.add_callbacks_post_trainer(mock_cfg, mock_trainer)
            
            # Should return one callback
            assert len(callbacks) == 1
            mock_callback_class.assert_called_once_with(mock_trainer)

    def test_add_callbacks_post_trainer_with_generation_disabled(self):
        """Test callback addition when generation is disabled."""
        plugin = DiffusionPlugin()
        mock_trainer = Mock()
        mock_cfg = Mock()
        
        # Mock trainer with diffusion config that has generation disabled
        mock_trainer.diffusion_config = DiffusionArgs(generate_samples=False)
        
        callbacks = plugin.add_callbacks_post_trainer(mock_cfg, mock_trainer)
        
        # Should return no callbacks
        assert len(callbacks) == 0


class TestLossRegistration:
    """Test loss function registration."""

    def test_register_diffusion_loss(self):
        """Test that loss function can be registered."""
        with patch("transformers.loss.loss_utils.LOSS_MAPPING", {}) as mock_mapping:
            result = register_diffusion_loss()
            assert result is True
            assert "ForDiffusionLM" in mock_mapping
            assert mock_mapping["ForDiffusionLM"] == ForDiffusionLMLoss

    def test_register_diffusion_loss_import_error(self):
        """Test fallback when LOSS_MAPPING import fails."""
        # Patch the import to raise ImportError
        with patch(
            "builtins.__import__",
            side_effect=ImportError("transformers.loss.loss_utils not found"),
        ):
            result = register_diffusion_loss()
            assert result is False
