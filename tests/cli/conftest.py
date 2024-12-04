"""
shared pytest fixtures for cli module
"""

import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

VALID_TEST_CONFIG = """
base_model: HuggingFaceTB/SmolLM2-135M
datasets:
  - path: mhenrichsen/alpaca_2k_test
    type: alpaca
sequence_len: 2048
max_steps: 1
micro_batch_size: 1
gradient_accumulation_steps: 1
learning_rate: 1e-3
special_tokens:
  pad_token: <|end_of_text|>
"""


@pytest.fixture
def cli_runner():
    return CliRunner()


@pytest.fixture
def config_path(tmp_path):
    """Creates a temporary config file"""
    path = tmp_path / "config.yml"
    path.write_text(VALID_TEST_CONFIG)

    return path


@pytest.fixture(autouse=True)
def cleanup_model_out():
    yield

    # Clean up after the test
    if Path("model-out").exists():
        shutil.rmtree("model-out")
