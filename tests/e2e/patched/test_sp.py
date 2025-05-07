"""Tests for sequence parallelism functionality."""

# pylint: disable=redefined-outer-name,unused-argument

import functools
import sys
from unittest.mock import MagicMock, patch

import pytest
import torch
from accelerate.state import PartialState

from axolotl.monkeypatch.attention.ring_attn import (
    get_ring_attn_group,
    register_ring_attn,
    set_ring_attn_group,
)
from axolotl.utils.ctx_managers.sequence_parallel import apply_sequence_parallelism
from axolotl.utils.dict import DictDefault
from axolotl.utils.schemas.enums import RingAttnFunc


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


@pytest.fixture
def sequence_parallel_batch():
    """Create a test batch for sequence parallelism tests."""
    batch_size = 1
    seq_len = 8

    # Create test tensors
    input_ids = torch.arange(batch_size * seq_len).reshape(batch_size, seq_len)
    attention_mask = torch.ones(batch_size, seq_len)
    position_ids = torch.arange(seq_len).expand(batch_size, seq_len)

    # Create test batch
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }

    return batch


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
        # Setup mocks
        mock_world_size.return_value = 8  # 8 GPUs total
        mock_rank.return_value = 3  # GPU #3
        mock_group = MagicMock()
        mock_new_group.return_value = mock_group

        # Call register_ring_attn with size 4
        register_ring_attn(
            sequence_parallel_degree=4,
            heads_k_stride=1,
            ring_attn_func=RingAttnFunc.VARLEN_LLAMA3,
        )

        # Verify the number of calls without examining the arguments
        assert mock_new_group.call_count == 2

        # Verify that new_group was called
        mock_new_group.assert_called()

        # Clean up
        set_ring_attn_group(None)


class TestConfigValidation:
    """Tests for validating sequence parallelism configurations."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self, monkeypatch):
        """Set up mocks for all tests in this class."""
        # Mock the ring_flash_attn module
        monkeypatch.setitem(sys.modules, "ring_flash_attn", MagicMock())

    @pytest.fixture
    def base_cfg(self):
        """Create a base configuration for testing."""
        return DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "datasets": [{"path": "mhenrichsen/alpaca_2k_test", "type": "alpaca"}],
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "learning_rate": 1e-3,
                "output_dir": "./model-out",
                "sequence_len": 512,
                "special_tokens": {"pad_token": "<|endoftext|>"},
            }
        )

    @pytest.mark.parametrize(
        "config_updates, expected_values, should_pass, error_msg",
        [
            # Valid configuration
            (
                {"sequence_parallel_degree": 2, "flash_attention": True},
                {"sequence_parallel_degree": 2, "flash_attention": True},
                True,
                None,
            ),
            # Default sequence_parallel_degree
            ({}, {"sequence_parallel_degree": 1}, True, None),
            # Invalid: sequence_parallel_degree > 1 without flash_attention
            (
                {"sequence_parallel_degree": 2, "flash_attention": False},
                None,
                False,
                "flash_attention: true must be set",
            ),
            # Invalid: sequence_parallel_degree > 1 with sample_packing and micro_batch_size > 1
            (
                {
                    "sequence_parallel_degree": 2,
                    "flash_attention": True,
                    "sample_packing": True,
                    "micro_batch_size": 2,
                    "pad_to_sequence_len": True,
                },
                None,
                False,
                "micro_batch_size must be set to 1",
            ),
        ],
        ids=[
            "valid_config",
            "default_sp_degree",
            "without_flash_attention",
            "sample_packing_with_large_batch",
        ],
    )
    def test_sequence_parallel_config_validation(
        self, base_cfg, config_updates, expected_values, should_pass, error_msg
    ):
        """Test various sequence parallelism configuration scenarios."""
        from axolotl.utils.schemas.config import AxolotlInputConfig

        # Apply updates to base config
        cfg = base_cfg
        cfg.update(config_updates)

        if should_pass:
            # Should validate without errors
            config = AxolotlInputConfig(**cfg)

            # Check expected values
            for key, value in expected_values.items():
                assert getattr(config, key) == value
        else:
            # Should raise exception
            with pytest.raises(ValueError) as excinfo:
                AxolotlInputConfig(**cfg)
            assert error_msg in str(excinfo.value)

    @pytest.mark.parametrize(
        "ring_attn_func, sample_packing, expected_func",
        [
            (None, True, RingAttnFunc.VARLEN_LLAMA3),
            (None, False, RingAttnFunc.BATCH_RING),
        ],
        ids=["default_with_sample_packing", "default_without_sample_packing"],
    )
    def test_ring_attn_func_validation(
        self, base_cfg, ring_attn_func, sample_packing, expected_func
    ):
        """Test ring_attn_func validation and defaults."""
        from axolotl.utils.schemas.config import AxolotlInputConfig

        # Apply updates to base config
        cfg = base_cfg | {
            "sequence_parallel_degree": 2,
            "flash_attention": True,
            "sample_packing": sample_packing,
        }

        if ring_attn_func is not None:
            cfg["ring_attn_func"] = ring_attn_func

        # Should validate without errors
        config = AxolotlInputConfig(**cfg)

        # Check ring_attn_func value
        assert config.ring_attn_func.value == expected_func

    def test_invalid_ring_attn_func(self, base_cfg):
        """Test that an invalid ring_attn_func is rejected."""
        from axolotl.utils.schemas.config import AxolotlInputConfig

        # Invalid configuration with invalid ring_attn_func
        cfg = base_cfg | {
            "sequence_parallel_degree": 2,
            "flash_attention": True,
            "ring_attn_func": "INVALID_FUNC",
        }

        # Should raise ValidationError
        with pytest.raises(ValueError) as excinfo:
            AxolotlInputConfig(**cfg)

        # Verify error message
        assert "ring_attn_func: INVALID_FUNC must be in" in str(excinfo.value)


class TestApplySequenceParallelism:
    """Tests for the apply_sequence_parallelism function."""

    @pytest.fixture(autouse=True)
    def mock_distributed(self, monkeypatch):
        """Mock torch.distributed functions for testing."""
        # Mock is_initialized to return True
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

        # Mock get_rank to return 0 by default
        monkeypatch.setattr(torch.distributed, "get_rank", lambda *args, **kwargs: 0)

        # Mock get_world_size to return 2 by default
        monkeypatch.setattr(
            torch.distributed, "get_world_size", lambda *args, **kwargs: 2
        )

        # Mock the process group
        monkeypatch.setattr(
            "axolotl.monkeypatch.attention.ring_attn.get_ring_attn_group",
            MagicMock,
        )

        # Mock update_ring_attn_params
        monkeypatch.setattr(
            "axolotl.monkeypatch.attention.ring_attn.update_ring_attn_params",
            lambda **kwargs: None,
        )

    def test_world_size_one(self, sequence_parallel_batch):
        """Test that function returns original batch when world size is 1."""
        result, _, _ = apply_sequence_parallelism(
            batch=sequence_parallel_batch,
            local_rank=0,
            local_world_size=1,
            ring_attn_func=RingAttnFunc.BATCH_RING,
        )

        # Should return the original batch unchanged
        assert result == sequence_parallel_batch

    def test_batch_ring_rank0(self, sequence_parallel_batch):
        """Test BATCH_RING sharding for rank 0 in a 2-process group."""
        batch = sequence_parallel_batch
        seq_len = batch["input_ids"].size(1)

        result, _, _ = apply_sequence_parallelism(
            batch=batch,
            local_rank=0,
            local_world_size=2,
            ring_attn_func=RingAttnFunc.BATCH_RING,
        )

        # Check that sequence dimension was sharded correctly
        assert result["input_ids"].shape[1] == seq_len // 2
        assert result["attention_mask"].shape[1] == seq_len // 2

        # Verify content: rank 0 should get the first half of the sequence
        assert torch.equal(result["input_ids"], batch["input_ids"][:, : seq_len // 2])
        assert torch.equal(
            result["position_ids"], batch["position_ids"][:, : seq_len // 2]
        )

    def test_batch_ring_rank1(self, sequence_parallel_batch):
        """Test BATCH_RING sharding for rank 1 in a 2-process group."""
        batch = sequence_parallel_batch
        seq_len = batch["input_ids"].size(1)
        original_input_ids = batch["input_ids"].clone()

        result, _, _ = apply_sequence_parallelism(
            batch=batch,
            local_rank=1,
            local_world_size=2,
            ring_attn_func=RingAttnFunc.BATCH_RING,
        )

        # Verify content: rank 1 should get the second half of the sequence
        assert torch.equal(result["input_ids"], original_input_ids[:, seq_len // 2 :])

    # def test_batch_zigzag(self, sequence_parallel_batch):
    #     """Test BATCH_ZIGZAG sharding pattern."""
    #     batch = sequence_parallel_batch
    #     original_input_ids = batch["input_ids"].clone()
    #     seq_len = batch["input_ids"].size(1)

    #     # Test rank 0
    #     result_rank0 = apply_sequence_parallelism(
    #         batch={k: v.clone() for k, v in batch.items()},
    #         local_rank=0,
    #         local_world_size=2,
    #         ring_attn_func=RingAttnFunc.BATCH_ZIGZAG,
    #     )

    #     # Test rank 1
    #     result_rank1 = apply_sequence_parallelism(
    #         batch={k: v.clone() for k, v in batch.items()},
    #         local_rank=1,
    #         local_world_size=2,
    #         ring_attn_func=RingAttnFunc.BATCH_ZIGZAG,
    #     )

    #     # Checks for both ranks
    #     assert result_rank0["input_ids"].shape[1] == seq_len // 2
    #     assert result_rank1["input_ids"].shape[1] == seq_len // 2

    #     # For a 2-rank system with 8 tokens, check specific zigzag pattern
    #     # Rank 0 should get chunks [0, 1] and [6, 7]
    #     # Rank 1 should get chunks [2, 3] and [4, 5]
    #     if seq_len == 8:
    #         # Create expected tensors for comparison
    #         rank0_expected = torch.cat(
    #             [original_input_ids[:, :2], original_input_ids[:, 6:8]], dim=1
    #         )

    #         rank1_expected = torch.cat(
    #             [original_input_ids[:, 2:4], original_input_ids[:, 4:6]], dim=1
    #         )

    #         assert torch.equal(result_rank0["input_ids"], rank0_expected)
    #         assert torch.equal(result_rank1["input_ids"], rank1_expected)

    def test_partial_application(self, sequence_parallel_batch):
        """Test that we can create a partially applied version of the function."""
        batch = sequence_parallel_batch
        original_input_ids = batch["input_ids"].clone()

        # Create a partially applied function
        rank0_ring_parallel = functools.partial(
            apply_sequence_parallelism,
            local_rank=0,
            local_world_size=2,
            ring_attn_func=RingAttnFunc.BATCH_RING,
        )

        # Use the partially applied function
        result, _, _ = rank0_ring_parallel(batch=batch)

        # Verify it works as expected
        assert result["input_ids"].shape[1] == original_input_ids.shape[1] // 2
        assert torch.equal(
            result["input_ids"],
            original_input_ids[:, : original_input_ids.shape[1] // 2],
        )

    def test_missing_position_ids(self, sequence_parallel_batch):
        """Test handling of batch without position_ids."""
        # Create a batch without position_ids
        batch = {
            k: v for k, v in sequence_parallel_batch.items() if k != "position_ids"
        }
        original_input_ids = batch["input_ids"].clone()

        # This should run without error even though position_ids is missing
        result, _, _ = apply_sequence_parallelism(
            batch=batch,
            local_rank=0,
            local_world_size=2,
            ring_attn_func=RingAttnFunc.BATCH_RING,
        )

        # Verification should pass
        assert "position_ids" not in result
        assert result["input_ids"].shape[1] == original_input_ids.shape[1] // 2
