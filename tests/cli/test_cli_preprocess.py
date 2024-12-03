"""
pytest tests for axolotl CLI preprocess command
"""
from unittest.mock import DEFAULT, patch

import pytest

from axolotl.cli.main import cli
from axolotl.common.const import DEFAULT_DATASET_PREPARED_PATH
from axolotl.utils.dict import DictDefault


@pytest.fixture
def mock_preprocess_deps():
    """Mock core dependencies for preprocessing"""
    with patch.multiple(
        "axolotl.cli.preprocess",
        load_datasets=DEFAULT,
        load_rl_datasets=DEFAULT,
        load_cfg=DEFAULT,
        init_empty_weights=DEFAULT,
        AutoModelForCausalLM=DEFAULT,
    ) as mocks:
        yield mocks


def test_preprocess_dataset_loading(
    cli_runner, mock_preprocess_deps
):  # pylint: disable=redefined-outer-name
    """Test dataset loading paths in preprocess command"""
    mock_preprocess_deps["load_cfg"].return_value = DictDefault(
        {"dataset_prepared_path": None, "rl": False, "base_model": "mock_model"}
    )

    cli_runner.invoke(cli, ["preprocess", "config.yml", "--download"])

    mock_preprocess_deps["load_datasets"].assert_called_once()
    assert not mock_preprocess_deps["load_rl_datasets"].called


def test_preprocess_rl_dataset_loading(
    cli_runner, mock_preprocess_deps
):  # pylint: disable=redefined-outer-name
    """Test RL dataset loading path"""
    mock_preprocess_deps["load_cfg"].return_value = DictDefault(
        {
            "dataset_prepared_path": "/path/to/data",
            "rl": True,
            "base_model": "mock_model",
        }
    )

    cli_runner.invoke(cli, ["preprocess", "config.yml"])

    mock_preprocess_deps["load_rl_datasets"].assert_called_once()
    assert not mock_preprocess_deps["load_datasets"].called


def test_preprocess_model_download(
    cli_runner, mock_preprocess_deps
):  # pylint: disable=redefined-outer-name
    """Test model validation during preprocessing"""
    mock_preprocess_deps["load_cfg"].return_value = DictDefault(
        {
            "dataset_prepared_path": "/path/to/data",
            "rl": False,
            "base_model": "mock_model",
        }
    )

    cli_runner.invoke(cli, ["preprocess", "config.yml", "--download"])

    mock_preprocess_deps["AutoModelForCausalLM"].from_pretrained.assert_called_with(
        "mock_model", trust_remote_code=True
    )


def test_preprocess_default_path(
    cli_runner, mock_preprocess_deps
):  # pylint: disable=redefined-outer-name
    """Test default dataset path handling"""
    mock_preprocess_deps["load_cfg"].return_value = DictDefault(
        {"dataset_prepared_path": None, "rl": False, "base_model": "mock_model"}
    )

    cli_runner.invoke(cli, ["preprocess", "config.yml"])

    assert (
        mock_preprocess_deps["load_datasets"].call_args[1]["cfg"].dataset_prepared_path
        == DEFAULT_DATASET_PREPARED_PATH
    )
