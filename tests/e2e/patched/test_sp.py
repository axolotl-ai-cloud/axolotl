"""Tests for sequence parallelism functionality."""

# pylint: disable=redefined-outer-name,unused-argument

from unittest.mock import MagicMock, patch

import pytest
import torch
from accelerate.state import PartialState

from axolotl.monkeypatch.attention.ring_attn import (
    get_ring_attn_group,
    set_ring_attn_group,
)
from axolotl.utils.dict import DictDefault


@pytest.fixture
def partial_state():
    """Create a real PartialState instance for testing."""
    state = PartialState()
    return state


@pytest.fixture(name="cfg")
def fixture_cfg():
    cfg = DictDefault(
        {
            "base_model": "HuggingFaceTB/SmolLM2-135M",
            "datasets": [
                {
                    "path": "mhenrichsen/alpaca_2k_test",
                    "type": "alpaca",
                },
            ],
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1e-3,
            "output_dir": "./model-out",
            "sequence_len": 512,
            "special_tokens": {
                "pad_token": "<|endoftext|>",
            },
        }
    )

    return cfg


class TestRingAttention:
    """Tests for the ring attention functionality."""

    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.get_world_size")
    def test_get_ring_attn_group_no_registration(
        self, mock_world_size, mock_rank, partial_state
    ):
        """Test that get_ring_attn_group returns None when no group has been registered."""
        # Setup mocks
        mock_world_size.return_value = 4
        mock_rank.return_value = 0

        # Get the group without registration
        group = get_ring_attn_group()

        # Verify that None was returned
        assert group is None

    @patch("torch.distributed.new_group")
    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.get_world_size")
    def test_register_ring_attn(
        self, mock_world_size, mock_rank, mock_new_group, partial_state
    ):
        """Test that ring attention groups are created correctly."""
        from axolotl.monkeypatch.attention.ring_attn import register_ring_attn

        # Setup mocks
        mock_world_size.return_value = 8  # 8 GPUs total
        mock_rank.return_value = 3  # GPU #3
        mock_group = MagicMock()
        mock_new_group.return_value = mock_group

        # Call register_ring_attn with size 4
        register_ring_attn(sequence_parallel_degree=4, heads_k_stride=1)

        # Verify the number of calls without examining the arguments
        assert mock_new_group.call_count == 2

        # Verify that new_group was called
        mock_new_group.assert_called()

        # Clean up
        set_ring_attn_group(None)


# Mock a simplified DataCollator test
@patch("axolotl.monkeypatch.attention.ring_attn.get_ring_attn_group")
@patch("torch.distributed.get_rank")
@patch("torch.distributed.get_world_size")
def test_sequence_parallel_slicing(
    mock_world_size, mock_rank, mock_get_group, partial_state
):
    """Test the basic sequence slicing logic without full collator instantiation."""
    # Setup mocks
    mock_get_group.return_value = MagicMock()
    mock_rank.return_value = 1  # Second GPU
    mock_world_size.return_value = 4  # 4 GPUs total

    # Create a sample batch
    batch = {
        "input_ids": torch.tensor(
            [
                [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112],
                [201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212],
            ]
        ),
        "attention_mask": torch.ones(2, 12),
    }

    # Simplified slicing logic from SequenceParallelDataCollator
    def slice_batch(batch, rank, world_size):
        result = {}
        for key in batch:
            seq_len = batch[key].shape[1]
            slice_size = seq_len // world_size
            start_idx = rank * slice_size
            end_idx = start_idx + slice_size if rank < world_size - 1 else seq_len
            result[key] = batch[key][:, start_idx:end_idx]
        return result

    # Slice the batch
    result = slice_batch(
        batch, rank=mock_rank.return_value, world_size=mock_world_size.return_value
    )

    # Check slicing
    assert result["input_ids"].shape == (2, 3)  # 12 tokens / 4 GPUs = 3 tokens per GPU
    expected_input_ids = torch.tensor(
        [
            [104, 105, 106],  # Second slice of first sequence
            [204, 205, 206],  # Second slice of second sequence
        ]
    )
    assert torch.all(result["input_ids"] == expected_input_ids)


@patch.dict("sys.modules", {"ring_flash_attn": MagicMock()})
def test_config_validation_with_valid_inputs(cfg):
    """Test that valid sequence parallelism configurations pass validation."""
    # Import the actual model class with appropriate mocks
    from axolotl.utils.schemas.config import AxolotlInputConfig

    # Valid configuration: sequence_parallel_degree > 1 and flash_attention is True
    cfg = cfg | {
        "sequence_parallel_degree": 2,
        "flash_attention": True,
    }

    # Should validate without errors
    config = AxolotlInputConfig(**cfg)
    assert config.sequence_parallel_degree == 2
    assert config.flash_attention is True


def test_config_validation_with_invalid_inputs(cfg):
    """Test that invalid sequence parallelism configurations fail validation."""
    from axolotl.utils.schemas.config import AxolotlInputConfig

    # Invalid configuration: sequence_parallel_degree > 1 but flash_attention is False
    cfg = cfg | {
        "sequence_parallel_degree": 2,
        "flash_attention": False,
    }

    # Should raise ValidationError
    with pytest.raises(ValueError) as excinfo:
        AxolotlInputConfig(**cfg)

    # Verify error message
    assert "flash_attention: true must be set" in str(excinfo.value)
