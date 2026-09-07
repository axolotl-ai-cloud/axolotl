"""pytest tests for axolotl CLI fetch command."""

from unittest.mock import patch

from axolotl.cli.main import fetch


def test_fetch_cli_examples(cli_runner):
    """Test fetch command with examples directory"""
    with patch("axolotl.cli.main.fetch_from_github") as mock_fetch:
        result = cli_runner.invoke(fetch, ["examples"])

        assert result.exit_code == 0
        mock_fetch.assert_called_once_with("examples/", None)


def test_fetch_cli_deepspeed(cli_runner):
    """Test fetch command with deepspeed_configs directory"""
    with patch("axolotl.cli.main.fetch_from_github") as mock_fetch:
        result = cli_runner.invoke(fetch, ["deepspeed_configs"])

        assert result.exit_code == 0
        mock_fetch.assert_called_once_with("deepspeed_configs/", None)


def test_fetch_cli_with_dest(cli_runner, tmp_path):
    """Test fetch command with custom destination"""
    with patch("axolotl.cli.main.fetch_from_github") as mock_fetch:
        custom_dir = tmp_path / "tmp_examples"
        result = cli_runner.invoke(fetch, ["examples", "--dest", str(custom_dir)])

        assert result.exit_code == 0
        mock_fetch.assert_called_once_with("examples/", str(custom_dir))


def test_fetch_cli_invalid_directory(cli_runner):
    """Test fetch command with invalid directory choice"""
    result = cli_runner.invoke(fetch, ["invalid"])
    assert result.exit_code != 0
