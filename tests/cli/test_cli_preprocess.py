"""pytest tests for axolotl CLI preprocess command."""

import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from axolotl.cli.main import cli


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
