"""Shared fixtures for differential transformer conversion tests."""

import pytest
from click.testing import CliRunner


@pytest.fixture(scope="class")
def base_config():
    """Basic config for testing."""
    return {
        "base_model": "HuggingFaceTB/SmolLM2-135M",
        "datasets": [
            {
                "path": "axolotl-ai-co/alpaca_100_test",
                "type": "alpaca",
            },
        ],
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-4,
        "val_set_size": 0.1,
        "micro_batch_size": 1,
        "sequence_len": 2048,
        "special_tokens": {
            "pad_token": "<|endoftext|>",
        },
    }


@pytest.fixture(scope="class")
def cli_runner():
    return CliRunner()
