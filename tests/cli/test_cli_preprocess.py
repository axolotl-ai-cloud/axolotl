"""
pytest tests for axolotl CLI preprocess command
"""
from unittest.mock import DEFAULT, MagicMock, patch

import pytest

from axolotl.cli.main import cli
from axolotl.common.const import DEFAULT_DATASET_PREPARED_PATH
from axolotl.utils.dict import DictDefault


@pytest.fixture
def mock_preprocess_deps():
    """Mock core dependencies for preprocessing"""
    with patch.multiple(
        'axolotl.cli.preprocess',  # patch where the functions are used
        load_datasets=DEFAULT,
        load_rl_datasets=DEFAULT,
        load_cfg=DEFAULT,
        check_accelerate_default_config=DEFAULT,
        check_user_token=DEFAULT,
        print_axolotl_text_art=DEFAULT,  # might need this too
    ) as cli_mocks, patch(
        'transformers.AutoModelForCausalLM'
    ) as mock_auto_model, patch(
        'accelerate.init_empty_weights'
    ) as mock_init_weights:
        mocks = {
            'AutoModelForCausalLM': mock_auto_model,
            'init_empty_weights': mock_init_weights,
            **cli_mocks,
        }
        yield mocks


def test_preprocess_dataset_loading(cli_runner, mock_preprocess_deps):
    """Test dataset loading paths in preprocess command"""
    mock_preprocess_deps["load_cfg"].return_value = DictDefault({
        "dataset_prepared_path": None, 
        "rl": False, 
        "base_model": "mock_model"
    })

    result = cli_runner.invoke(cli, ["preprocess", "config.yml", "--download"])
    print(f"CLI Result: {result.output}")  # see what happened
    print(f"Exit Code: {result.exit_code}")

    mock_preprocess_deps["load_datasets"].assert_called_once()


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


def test_preprocess_default_path(cli_runner, mock_preprocess_deps):
    """Test default dataset path handling"""
    mock_preprocess_deps["load_cfg"].return_value = DictDefault({
        "dataset_prepared_path": None,
        "rl": False,
        "base_model": "mock_model"
    })

    cli_runner.invoke(cli, ["preprocess", "config.yml"])

    mock_preprocess_deps["load_datasets"].assert_called_once()
    kwargs = mock_preprocess_deps["load_datasets"].call_args.kwargs
    assert kwargs["cfg"].dataset_prepared_path == DEFAULT_DATASET_PREPARED_PATH
