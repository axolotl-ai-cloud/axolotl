"""
pytest tests for axolotl CLI shard command
"""
from unittest.mock import patch

# pylint: disable=duplicate-code
import pytest

from axolotl.cli.main import cli


@pytest.mark.integration
def test_shard_with_accelerate(cli_runner, config_path):
    """Test shard command with accelerate"""
    result = cli_runner.invoke(cli, ["shard", str(config_path), "--accelerate"])

    assert result.exit_code == 0


def test_shard_no_accelerate(cli_runner, config_path):
    """Test shard command without accelerate"""
    with patch("axolotl.cli.shard.do_cli") as mock:
        result = cli_runner.invoke(cli, ["shard", str(config_path), "--no-accelerate"])

        assert mock.called
        assert result.exit_code == 0


def test_shard_with_model_dir(cli_runner, config_path):
    """Test shard command with model_dir option"""
    with patch("axolotl.cli.shard.do_cli") as mock:
        result = cli_runner.invoke(
            cli,
            [
                "shard",
                str(config_path),
                "--no-accelerate",
                "--model-dir",
                "/path/to/model",
            ],
        )  # pylint: disable=duplicate-code

        assert mock.called
        assert mock.call_args.kwargs["config"] == str(config_path)
        assert mock.call_args.kwargs["model_dir"] == "/path/to/model"
        assert result.exit_code == 0


def test_shard_with_save_dir(cli_runner, config_path):
    with patch("axolotl.cli.shard.do_cli") as mock:
        result = cli_runner.invoke(
            cli,
            [
                "shard",
                str(config_path),
                "--no-accelerate",
                "--save-dir",
                "/path/to/save",
            ],
        )

        assert mock.called
        assert mock.call_args.kwargs["config"] == str(config_path)
        assert mock.call_args.kwargs["save_dir"] == "/path/to/save"
        assert result.exit_code == 0
