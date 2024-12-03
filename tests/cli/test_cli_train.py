"""
pytest tests for axolotl CLI train command
"""
from unittest.mock import DEFAULT, MagicMock, patch

import pytest

from axolotl.cli.main import cli


@pytest.fixture
def mock_train_deps():
    """Mock dependencies for training"""
    with patch.multiple(
        'axolotl.cli',
        load_datasets=DEFAULT,
        load_rl_datasets=DEFAULT,
        load_cfg=DEFAULT,
        check_accelerate_default_config=DEFAULT,
        check_user_token=DEFAULT,
    ) as cli_mocks, patch(
        'axolotl.train.train'
    ) as mock_train, patch(
        'axolotl.integrations.base.PluginManager'
    ) as mock_plugin_manager:
        mock_plugin_instance = MagicMock()
        mock_plugin_manager.get_instance.return_value = mock_plugin_instance
        mocks = {
            'train': mock_train,
            'PluginManager': mock_plugin_manager,
            'plugin_manager_instance': mock_plugin_instance,
            **cli_mocks,
        }
        yield mocks


def test_train_command_args(cli_runner):
    """Test train command accepts various arguments"""
    with patch("subprocess.run") as mock_run:
        cli_runner.invoke(
            cli, ["train", "config.yml", "--learning-rate", "1e-4", "--batch-size", "8"]
        )

        cmd = mock_run.call_args[0][0]
        assert "--learning-rate" in cmd
        assert "1e-4" in cmd


def test_train_regular_flow(
    cli_runner, mock_train_deps
):  # pylint: disable=redefined-outer-name
    """Test normal training flow without RL"""
    mock_cfg = MagicMock()
    mock_cfg.rl = False
    mock_dataset_meta = {"some": "metadata"}
    mock_train_deps["load_datasets"].return_value = mock_dataset_meta

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_train_deps["train"].return_value = (mock_model, mock_tokenizer)

    cli_runner.invoke(cli, ["train", "config.yml"])

    mock_train_deps["check_accelerate_default_config"].assert_called_once()
    mock_train_deps["check_user_token"].assert_called_once()
    mock_train_deps["load_datasets"].assert_called_once()
    mock_train_deps["train"].assert_called_once()
    mock_train_deps["plugin_manager_instance"].post_train_unload.assert_called_once()


def test_train_rl_flow(
    cli_runner, mock_train_deps
):  # pylint: disable=redefined-outer-name
    """Test RL training flow"""
    mock_cfg = MagicMock()
    mock_cfg.rl = True
    mock_dataset_meta = {"some": "metadata"}
    mock_train_deps["load_rl_datasets"].return_value = mock_dataset_meta

    cli_runner.invoke(cli, ["train", "config.yml"])

    mock_train_deps["load_rl_datasets"].assert_called_once()
    assert not mock_train_deps["load_datasets"].called


def test_train_config_cli_merge(cli_runner):
    """Test that CLI args properly override config values"""
    with patch.multiple(
        "axolotl.cli.train", load_cfg=DEFAULT, do_train=DEFAULT
    ) as mocks:
        mock_cfg = MagicMock()
        mocks["load_cfg"].return_value = mock_cfg

        cli_runner.invoke(
            cli, ["train", "config.yml", "--learning-rate", "1e-4", "--batch-size", "8"]
        )

        mocks["load_cfg"].assert_called_with(
            "config.yml", learning_rate="1e-4", batch_size="8"
        )
