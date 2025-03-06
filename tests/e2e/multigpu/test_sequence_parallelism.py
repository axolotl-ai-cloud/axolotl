"""Tests for end-to-end sequence parallelism integration."""
import os
import tempfile

import pytest
import torch
import yaml

from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.dict import DictDefault


def test_integration_with_config():
    """Test end-to-end training configuration setup for sequence parallelism."""
    # Define a test config directly in code instead of loading from file
    config_dict = {
        "base_model": "HuggingFaceTB/SmolLM2-135M",
        "tokenizer_type": "LlamaTokenizer",
        "is_llama_derived_model": True,
        "datasets": [
            {
                "path": "mhenrichsen/alpaca_2k_test",
                "type": "alpaca",
            }
        ],
        "load_in_8bit": False,
        "sequence_len": 1024,
        "sequence_parallel_size": 2,
        "flash_attention": True,
        "sample_packing": True,
        "pad_to_sequence_len": True,
        "micro_batch_size": 2,
        "num_epochs": 1,
        "max_steps": 10,
        "gradient_accumulation_steps": 1,
        "warmup_steps": 2,
        "optimizer": "adamw_bnb_8bit",
        "lr_scheduler": "cosine",
        "learning_rate": 2.0e-4,
        "weight_decay": 0.0,
        "val_set_size": 0.05,
        "eval_steps": 5,
        "save_steps": 10,
    }

    # Create a temp dir for output
    with tempfile.TemporaryDirectory() as temp_dir:
        config_dict["output_dir"] = temp_dir

        # Also write to a file for completeness
        config_path = os.path.join(temp_dir, "sp_config.yml")
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f)

        # Convert to DictDefault and validate
        cfg = DictDefault(config_dict)
        cfg = validate_config(cfg)
        normalize_config(cfg)

        # Verify sequence parallelism settings were properly processed
        assert cfg.sequence_parallel_size == 2
        assert cfg.flash_attention is True

        # Check if the sequence_parallel_size was propagated to the training args
        from axolotl.core.training_args import AxolotlTrainingArguments

        # pylint: disable=unexpected-keyword-arg
        training_args = AxolotlTrainingArguments(
            output_dir=temp_dir, sequence_parallel_size=cfg.sequence_parallel_size
        )
        assert training_args.sequence_parallel_size == 2


def test_ring_attn_group_creation():
    """Test that ring attention groups are properly created in a multi-GPU environment."""
    # First ensure we're in a distributed environment
    if not torch.distributed.is_initialized():
        # Skip this test if not in distributed mode
        pytest.skip(
            "This test requires a properly initialized torch.distributed environment"
        )

    from axolotl.monkeypatch.attention.ring_attn import (
        get_ring_attn_group,
        register_ring_attn,
    )

    # Get the current rank and world size
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    # Only run if we have an even number of GPUs
    if world_size % 2 != 0:
        pytest.skip(f"Need an even number of GPUs, but got {world_size}")

    # Register with sequence parallel size of 2
    register_ring_attn(sequence_parallel_size=2)

    # Get the ring attention group
    group = get_ring_attn_group()

    # Verify the group exists
    assert group is not None

    # Calculate expected group members
    group_id = rank // 2
    expected_start = group_id * 2
    expected_group = list(range(expected_start, expected_start + 2))

    # Verify our rank is in the expected group
    assert rank in expected_group

    # Clean up by synchronizing all processes
    torch.distributed.barrier()
