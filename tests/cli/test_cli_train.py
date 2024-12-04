"""Test the train CLI command"""
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from axolotl.cli.main import cli

VALID_TEST_CONFIG = """
base_model: HuggingFaceTB/SmolLM2-135M
datasets:
  - path: mhenrichsen/alpaca_2k_test
    type: alpaca
sequence_len: 2048
micro_batch_size: 1
gradient_accumulation_steps: 1
max_steps: 1
val_set_size: 0
learning_rate: 1e-3
special_tokens:
  pad_token: <|end_of_text|>
"""


@pytest.fixture(autouse=True)
def cleanup_model_out():
    yield

    # Clean up after the test
    if Path("model-out").exists():
        shutil.rmtree("model-out")


def test_train_cli_validation(cli_runner):
    """Test CLI validation"""
    # Test missing config file
    result = cli_runner.invoke(cli, ["train"])
    assert result.exit_code != 0

    # Test non-existent config file
    result = cli_runner.invoke(cli, ["train", "nonexistent.yml", "--no-accelerate"])
    assert result.exit_code != 0
    assert "No such file" in str(result.exception)


def test_train_basic_execution(cli_runner, tmp_path):
    """Test basic successful execution"""
    config_path = tmp_path / "config.yml"
    config_path.write_text(VALID_TEST_CONFIG)

    result = cli_runner.invoke(
        cli,
        [
            "train",
            str(config_path),
            "--no-accelerate",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0


def test_train_basic_execution_accelerate(cli_runner, tmp_path):
    """Test basic successful execution"""
    config_path = tmp_path / "config.yml"
    config_path.write_text(VALID_TEST_CONFIG)

    result = cli_runner.invoke(
        cli,
        [
            "train",
            str(config_path),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0


def test_train_cli_overrides(cli_runner, tmp_path):
    """Test CLI arguments properly override config values"""
    config_path = tmp_path / "config.yml"
    output_dir = tmp_path / "model-out"

    test_config = VALID_TEST_CONFIG.replace(
        "output_dir: model-out", f"output_dir: {output_dir}"
    )
    config_path.write_text(test_config)

    with patch("axolotl.cli.train.train") as mock_train:
        mock_train.return_value = (MagicMock(), MagicMock())

        result = cli_runner.invoke(
            cli,
            [
                "train",
                str(config_path),
                "--learning-rate",
                "1e-4",
                "--micro-batch-size",
                "2",
                "--no-accelerate",
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        mock_train.assert_called_once()
        cfg = mock_train.call_args[1]["cfg"]
        assert cfg["learning_rate"] == 1e-4
        assert cfg["micro_batch_size"] == 2
