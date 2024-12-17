"""Tests for convert-differential-transformer CLI command."""

from pathlib import Path
from unittest.mock import patch

from axolotl.cli.main import cli


def test_cli_validation(cli_runner):
    """Test CLI validation for a command.

    Args:
        cli_runner: CLI runner fixture
    """
    # Test missing config file
    result = cli_runner.invoke(cli, ["convert-differential-transformer"])
    assert result.exit_code != 0
    assert "Error: Missing argument 'CONFIG'." in result.output

    # Test non-existent config file
    result = cli_runner.invoke(
        cli, ["convert-differential-transformer", "nonexistent.yml"]
    )
    assert result.exit_code != 0
    assert "Error: Invalid value for 'CONFIG'" in result.output


def test_basic_execution(cli_runner, tmp_path: Path, valid_test_config: str):
    """Test basic execution.

    Args:
        cli_runner: CLI runner fixture
        tmp_path: Temporary path fixture
        valid_test_config: Valid config fixture
    """
    config_path = tmp_path / "config.yml"
    config_path.write_text(valid_test_config)

    with patch(
        "axolotl.cli.integrations.convert_differential_transformer.do_cli"
    ) as mock_do_cli:
        result = cli_runner.invoke(
            cli, ["convert-differential-transformer", str(config_path)]
        )
        assert result.exit_code == 0

        mock_do_cli.assert_called_once()
        assert mock_do_cli.call_args.kwargs["config"] == str(config_path)
