"""
pytest tests for axolotl CLI inference command
"""
from unittest.mock import DEFAULT, MagicMock, patch

import pytest

from axolotl.cli.main import cli


@pytest.fixture
def mock_inference_deps():
    """Mock dependencies for inference"""
    with patch.multiple(
        "axolotl.cli",
        load_cfg=DEFAULT,
        do_inference=DEFAULT,
        do_inference_gradio=DEFAULT,
        print_axolotl_text_art=DEFAULT,
    ) as cli_mocks:
        yield cli_mocks


def test_inference_basic(
    cli_runner, default_config, mock_inference_deps
):  # pylint: disable=redefined-outer-name
    mock_cfg = MagicMock()
    mock_inference_deps["load_cfg"].return_value = mock_cfg

    result = cli_runner.invoke(cli, ["inference", str(default_config)])
    assert result.exit_code == 0

    mock_inference_deps["load_cfg"].assert_called_once()
    mock_inference_deps["do_inference"].assert_called_once()

    args = mock_inference_deps["do_inference"].call_args[1]
    assert args["cfg"] == mock_cfg
    assert args["cli_args"].inference is True
    assert mock_cfg.sample_packing is False


def test_inference_gradio(
    cli_runner, default_config, mock_inference_deps
):  # pylint: disable=redefined-outer-name
    """Test inference with gradio interface"""
    mock_cfg = MagicMock()
    mock_inference_deps["load_cfg"].return_value = mock_cfg

    cli_runner.invoke(cli, ["inference", str(default_config), "--gradio"])

    mock_inference_deps["do_inference_gradio"].assert_called_once()
    mock_inference_deps["do_inference"].assert_not_called()
    assert mock_cfg.sample_packing is False

    # Verify CLI args
    cli_args = mock_inference_deps["do_inference_gradio"].call_args[1]["cli_args"]
    assert cli_args.inference is True


def test_inference_model_paths(
    cli_runner, default_config, mock_inference_deps
):  # pylint: disable=redefined-outer-name
    """Test inference with model path options"""
    mock_cfg = MagicMock()
    mock_inference_deps["load_cfg"].return_value = mock_cfg

    cli_runner.invoke(
        cli,
        [
            "inference",
            str(default_config),
            "--lora-model-dir",
            "lora/path",
            "--base-model",
            "base/model/path",
        ],
    )

    assert mock_inference_deps["load_cfg"].call_args[1] == {
        "lora_model_dir": "lora/path",
        "base_model": "base/model/path",
    }
    assert mock_cfg.sample_packing is False


def test_inference_all_options(
    cli_runner, default_config, mock_inference_deps
):  # pylint: disable=redefined-outer-name
    """Test inference with all possible options"""
    mock_cfg = MagicMock()
    mock_inference_deps["load_cfg"].return_value = mock_cfg

    cli_runner.invoke(
        cli,
        [
            "inference",
            str(default_config),
            "--load-in-8bit",
            "--base-model",
            "base/model/path",
            "--lora-model-dir",
            "lora/path",
            "--prompter",
            "my_prompter",
        ],
    )

    # Check all options were passed through
    cli_args = mock_inference_deps["do_inference"].call_args[1]["cli_args"]
    assert cli_args.load_in_8bit is True
    assert cli_args.prompter == "my_prompter"


def test_inference_config_not_found(cli_runner):
    """Test inference fails when config not found"""
    result = cli_runner.invoke(cli, ["inference", "nonexistent.yml"])
    assert result.exit_code != 0
