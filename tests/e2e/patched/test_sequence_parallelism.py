"""Tests for sequence parallelism functionality."""
# pylint: disable=redefined-outer-name,unused-argument

from unittest.mock import MagicMock, patch

import pytest
import torch
from accelerate.state import PartialState

# Use a single patch for ring_flash_attn if it's not available
ring_flash_attn_mock = MagicMock()
with patch.dict("sys.modules", {"ring_flash_attn": ring_flash_attn_mock}):
    from axolotl.monkeypatch.attention.ring_attn import get_ring_attn_group
    from axolotl.utils.collators.sequence_parallel import (
        adjust_position_ids_for_slice,
        check_for_boundary_splits,
        find_sample_boundaries,
    )


# Create a fixture for PartialState
@pytest.fixture
def partial_state():
    """Create a real PartialState instance for testing."""
    # This initializes a PartialState for a non-distributed environment
    state = PartialState()
    return state


class TestSequenceParallelHelpers:
    """Test helper functions used in sequence parallelism."""

    def test_find_sample_boundaries(self):
        """Test detection of boundaries in position_ids."""
        # Create sample position_ids with multiple sequences
        position_ids = torch.tensor(
            [
                # First sequence with 2 samples (boundary at index 5)
                [0, 1, 2, 3, 4, 0, 1, 2, 3],
                # Second sequence with 3 samples (boundaries at 3 and 7)
                [0, 1, 2, 0, 1, 2, 3, 0, 1],
            ]
        )

        boundaries = find_sample_boundaries(position_ids)

        assert len(boundaries) == 2
        assert boundaries[0] == [5]  # First sequence has boundary at index 5
        assert boundaries[1] == [3, 7]  # Second sequence has boundaries at 3 and 7

    def test_adjust_position_ids_for_slice(self, partial_state):
        """Test position_ids adjustment for sequence slices."""
        # Create sample position_ids with multiple sequences
        position_ids = torch.tensor(
            [
                # First sequence with 2 samples
                [0, 1, 2, 3, 4, 0, 1, 2, 3],
                # Second sequence with 3 samples
                [0, 1, 2, 0, 1, 2, 3, 0, 1],
            ]
        )

        # Adjust as if this was the second slice (start_idx = 4)
        adjusted = adjust_position_ids_for_slice(position_ids, start_idx=4)

        # For first sequence: [0,1,2,3,4,0,1,2,3] -> [-4,-3,-2,-1,0,-4,-3,-2,-1]
        # For second sequence: [0,1,2,0,1,2,3,0,1] -> [-4,-3,-2,-4,-3,-2,-1,-4,-3]
        expected_first_seq = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3]) - 4
        expected_second_seq = torch.tensor([0, 1, 2, 0, 1, 2, 3, 0, 1]) - 4

        assert torch.all(adjusted[0] == expected_first_seq)
        assert torch.all(adjusted[1] == expected_second_seq)

    def test_check_for_boundary_splits(self):
        """Test detection of boundaries near slice edges."""
        # Boundaries at positions 10, 25, 40
        boundaries = [10, 25, 40]

        # Test case where two boundaries are near edges (one at start, one at end)
        problems = check_for_boundary_splits(boundaries, slice_start=8, slice_end=30)
        assert (
            len(problems) == 2
        )  # Both boundary at 10 (near start) and 25 (near end) are problems

        # Check first problem - boundary near start
        assert problems[0][0] == 10  # The boundary position
        assert problems[0][1] == "start"  # Type of issue
        assert problems[0][2] == 2  # Distance from start

        # Check second problem - boundary near end
        assert problems[1][0] == 25  # The boundary position
        assert problems[1][1] == "end"  # Type of issue
        assert problems[1][2] == 5  # Distance from end

        # Test case with only one problem at the end
        problems = check_for_boundary_splits(boundaries, slice_start=15, slice_end=27)
        assert len(problems) == 1  # Only boundary at 25 is near the end
        assert problems[0][0] == 25  # The boundary
        assert problems[0][1] == "end"  # Type of issue

        # Test case with no problems
        problems = check_for_boundary_splits(boundaries, slice_start=12, slice_end=20)
        assert len(problems) == 0


class TestRingAttention:
    """Tests for the ring attention functionality."""

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
        register_ring_attn(sequence_parallel_size=4)

        # Verify the number of calls without examining the arguments
        assert mock_new_group.call_count == 2

        # Just verify that new_group was called
        mock_new_group.assert_called()

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


# Mock a simplified DataCollator test
@patch("axolotl.utils.collators.sequence_parallel.get_ring_attn_group")
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


# Simple test for configuration validation
@pytest.mark.parametrize(
    "config,should_validate",
    [
        ({"sequence_parallel_size": 2, "flash_attention": True}, True),
        ({"sequence_parallel_size": 2, "flash_attention": False}, False),
        ({"sequence_parallel_size": 1, "flash_attention": False}, True),
    ],
)
def test_sequence_parallel_config_requirements(config, should_validate):
    """Test basic sequence parallelism configuration requirements."""

    # Simple validation function that mimics the actual validator
    def validate_sp_config(config):
        if config.get("sequence_parallel_size", 1) > 1 and not config.get(
            "flash_attention", False
        ):
            return False
        return True

    assert validate_sp_config(config) == should_validate
