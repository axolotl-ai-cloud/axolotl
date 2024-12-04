"""
pytest tests for axolotl CLI train command
"""
# pylint: disable=redefined-outer-name
from unittest.mock import MagicMock, patch

import pytest

from axolotl.cli.main import cli


@pytest.fixture
def mock_train_deps(common_mocks):
    """Mock dependencies for training"""
    with patch.multiple(
        "axolotl.cli",
        **common_mocks,
    ) as cli_mocks, patch("axolotl.train.train") as mock_train, patch(
        "axolotl.integrations.base.PluginManager"
    ) as mock_plugin_manager:
        mock_plugin_instance = MagicMock()
        mock_plugin_manager.get_instance.return_value = mock_plugin_instance
        mocks = {
            "train": mock_train,
            "PluginManager": mock_plugin_manager,
            "plugin_manager_instance": mock_plugin_instance,
            **cli_mocks,
        }
        yield mocks


def test_train_regular_flow(cli_runner, default_config, mock_train_deps):
    """Test regular training flow"""
    mock_cfg = MagicMock()
    mock_cfg.rl = False
    mock_dataset_meta = {"some": "metadata"}
    mock_train_deps["load_datasets"].return_value = mock_dataset_meta
    mock_train_deps["load_cfg"].return_value = mock_cfg

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_train_deps["train"].return_value = (mock_model, mock_tokenizer)

    result = cli_runner.invoke(cli, ["train", str(default_config), "--no-accelerate"])
    print(f"CLI Result: {result.output}")
    print(f"Exit code: {result.exit_code}")

    mock_train_deps["check_accelerate_default_config"].assert_called_once()
    mock_train_deps["check_user_token"].assert_called_once()
    mock_train_deps["load_datasets"].assert_called_once()
    mock_train_deps["train"].assert_called_once()


def test_train_rl_flow(cli_runner, default_config, mock_train_deps):
    """Test RL training flow"""
    mock_cfg = MagicMock()
    mock_cfg.rl = True
    mock_dataset_meta = {"some": "metadata"}
    mock_train_deps["load_rl_datasets"].return_value = mock_dataset_meta
    mock_train_deps["load_cfg"].return_value = mock_cfg

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_train_deps["train"].return_value = (mock_model, mock_tokenizer)

    result = cli_runner.invoke(cli, ["train", str(default_config), "--no-accelerate"])
    print(f"CLI Result: {result.output}")
    print(f"Exit code: {result.exit_code}")

    mock_train_deps["check_accelerate_default_config"].assert_called_once()
    mock_train_deps["check_user_token"].assert_called_once()
    mock_train_deps["load_rl_datasets"].assert_called_once()
    assert not mock_train_deps["load_datasets"].called
    mock_train_deps["train"].assert_called_once()


def test_train_config_cli_merge(cli_runner, default_config, mock_train_deps):
    """Test that CLI args properly override config values"""
    mock_cfg = MagicMock()
    mock_train_deps["load_cfg"].return_value = mock_cfg

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_train_deps["train"].return_value = (mock_model, mock_tokenizer)

    result = cli_runner.invoke(
        cli,
        [
            "train",
            str(default_config),
            "--no-accelerate",
            "--learning-rate",
            "1e-4",
            "--batch-size",
            "8",
        ],
    )
    print(f"CLI Result: {result.output}")
    print(f"Exit code: {result.exit_code}")

    args, kwargs = mock_train_deps["load_cfg"].call_args
    assert args[0] == str(default_config)
    assert kwargs["learning_rate"] == "1e-4"
    assert kwargs["batch_size"] == "8"
    mock_train_deps["train"].assert_called_once()


def test_train_config_not_found(cli_runner):
    """Test train fails when config not found"""
    result = cli_runner.invoke(cli, ["train", "nonexistent.yml", "--no-accelerate"])
    assert result.exit_code != 0
