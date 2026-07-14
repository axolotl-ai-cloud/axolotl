"""pytest tests for axolotl CLI preprocess command."""

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from axolotl.cli.args import PreprocessCliArgs
from axolotl.cli.main import cli
from axolotl.cli.preprocess import do_preprocess
from axolotl.utils.dict import DictDefault


@pytest.fixture(autouse=True)
def cleanup_last_run_prepared():
    yield

    if Path("last_run_prepared").exists():
        shutil.rmtree("last_run_prepared")


def test_preprocess_config_not_found(cli_runner):
    """Test preprocess fails when config not found"""
    result = cli_runner.invoke(cli, ["preprocess", "nonexistent.yml"])
    assert result.exit_code != 0


def test_preprocess_basic(cli_runner, config_path):
    """Test basic preprocessing with minimal config"""
    with patch("axolotl.cli.preprocess.do_cli") as mock_do_cli:
        with patch("axolotl.cli.preprocess.load_datasets") as mock_load_datasets:
            mock_load_datasets.return_value = MagicMock()

            result = cli_runner.invoke(cli, ["preprocess", str(config_path)])
            assert result.exit_code == 0

            mock_do_cli.assert_called_once()
            assert mock_do_cli.call_args.kwargs["config"] == str(config_path)
            assert mock_do_cli.call_args.kwargs["download"] is True


def test_preprocess_without_download(cli_runner, config_path):
    """Test preprocessing without model download"""
    with patch("axolotl.cli.preprocess.do_cli") as mock_do_cli:
        result = cli_runner.invoke(
            cli, ["preprocess", str(config_path), "--no-download"]
        )
        assert result.exit_code == 0

        mock_do_cli.assert_called_once()
        assert mock_do_cli.call_args.kwargs["config"] == str(config_path)
        assert mock_do_cli.call_args.kwargs["download"] is False


def test_preprocess_custom_path(cli_runner, tmp_path, valid_test_config):
    """Test preprocessing with custom dataset path"""
    config_path = tmp_path / "config.yml"
    custom_path = tmp_path / "custom_prepared"
    config_path.write_text(valid_test_config)

    with patch("axolotl.cli.preprocess.do_cli") as mock_do_cli:
        with patch("axolotl.cli.preprocess.load_datasets") as mock_load_datasets:
            mock_load_datasets.return_value = MagicMock()

            result = cli_runner.invoke(
                cli,
                [
                    "preprocess",
                    str(config_path),
                    "--dataset-prepared-path",
                    str(custom_path.absolute()),
                ],
            )
            assert result.exit_code == 0

            mock_do_cli.assert_called_once()
            assert mock_do_cli.call_args.kwargs["config"] == str(config_path)
            assert mock_do_cli.call_args.kwargs["dataset_prepared_path"] == str(
                custom_path.absolute()
            )


@pytest.mark.parametrize(
    "trust_remote_code, expected",
    [(None, False), (False, False), (True, True)],
)
def test_preprocess_download_respects_trust_remote_code(trust_remote_code, expected):
    """The --download pre-fetch must honor cfg.trust_remote_code, not hardcode True."""
    cfg = DictDefault(
        base_model="HuggingFaceTB/SmolLM2-135M",
        dataset_prepared_path="last_run_prepared",
        trust_remote_code=trust_remote_code,
    )
    cli_args = PreprocessCliArgs(download=True)

    with (
        patch("axolotl.cli.preprocess.check_accelerate_default_config"),
        patch("axolotl.cli.preprocess.check_user_token"),
        patch("axolotl.cli.preprocess.PluginManager") as mock_plugin_manager,
        patch("axolotl.cli.preprocess.load_datasets"),
        patch("axolotl.cli.preprocess.AutoModelForCausalLM") as mock_auto_model,
    ):
        mock_plugin_manager.get_instance.return_value.load_datasets.return_value = False

        do_preprocess(cfg, cli_args)

    mock_auto_model.from_pretrained.assert_called_once()
    call = mock_auto_model.from_pretrained.call_args
    assert call.args[0] == cfg.base_model
    assert call.kwargs["trust_remote_code"] is expected
