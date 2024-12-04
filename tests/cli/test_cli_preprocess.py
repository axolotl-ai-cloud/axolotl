"""
pytest tests for axolotl CLI preprocess command
"""
# pylint: disable=redefined-outer-name
from unittest.mock import patch

import pytest

from axolotl.cli.main import cli
from axolotl.common.const import DEFAULT_DATASET_PREPARED_PATH
from axolotl.utils.dict import DictDefault


@pytest.fixture
def mock_preprocess_deps(common_mocks):
    """Mock core dependencies for preprocessing"""
    with patch.multiple(
        "axolotl.cli",
        **common_mocks,
    ) as cli_mocks, patch(
        "transformers.AutoModelForCausalLM"
    ) as mock_auto_model, patch("accelerate.init_empty_weights") as mock_init_weights:
        mocks = {
            "AutoModelForCausalLM": mock_auto_model,
            "init_empty_weights": mock_init_weights,
            **cli_mocks,
        }
        yield mocks


def test_preprocess_dataset_loading(cli_runner, default_config, mock_preprocess_deps):
    """Test dataset loading paths in preprocess command"""
    mock_preprocess_deps["load_cfg"].return_value = DictDefault(
        {"dataset_prepared_path": None, "rl": False, "base_model": "mock_model"}
    )

    cli_runner.invoke(cli, ["preprocess", str(default_config), "--download"])

    mock_preprocess_deps["load_datasets"].assert_called_once()


def test_preprocess_rl_dataset_loading(
    cli_runner, default_config, mock_preprocess_deps
):
    """Test RL dataset loading path"""
    mock_preprocess_deps["load_cfg"].return_value = DictDefault(
        {
            "dataset_prepared_path": "/path/to/data",
            "rl": True,
            "base_model": "mock_model",
        }
    )

    cli_runner.invoke(cli, ["preprocess", str(default_config)])

    mock_preprocess_deps["load_rl_datasets"].assert_called_once()
    assert not mock_preprocess_deps["load_datasets"].called


def test_preprocess_model_download(cli_runner, default_config, mock_preprocess_deps):
    """Test model validation during preprocessing"""
    mock_preprocess_deps["load_cfg"].return_value = DictDefault(
        {
            "dataset_prepared_path": "/path/to/data",
            "rl": False,
            "base_model": "mock_model",
        }
    )

    cli_runner.invoke(cli, ["preprocess", str(default_config), "--download"])

    mock_preprocess_deps["AutoModelForCausalLM"].from_pretrained.assert_called_with(
        "mock_model", trust_remote_code=True
    )


def test_preprocess_default_path(cli_runner, default_config, mock_preprocess_deps):
    """Test default dataset path handling"""
    mock_preprocess_deps["load_cfg"].return_value = DictDefault(
        {"dataset_prepared_path": None, "rl": False, "base_model": "mock_model"}
    )

    cli_runner.invoke(cli, ["preprocess", str(default_config)])

    mock_preprocess_deps["load_datasets"].assert_called_once()
    kwargs = mock_preprocess_deps["load_datasets"].call_args.kwargs
    assert kwargs["cfg"].dataset_prepared_path == DEFAULT_DATASET_PREPARED_PATH


def test_preprocess_config_not_found(cli_runner):
    """Test preprocess fails when config not found"""
    result = cli_runner.invoke(cli, ["preprocess", "nonexistent.yml"])
    assert result.exit_code != 0
