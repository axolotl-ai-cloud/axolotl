"""pytest tests for axolotl CLI merge_lora command."""

from unittest.mock import patch

from axolotl.cli.main import cli


def test_merge_lora_basic(cli_runner, config_path):
    """Test basic merge_lora command"""
    with patch("axolotl.cli.merge_lora.do_cli") as mock_do_cli:
        result = cli_runner.invoke(cli, ["merge-lora", str(config_path)])
        assert result.exit_code == 0

        mock_do_cli.assert_called_once()
        assert mock_do_cli.call_args.kwargs["config"] == str(config_path)


def test_merge_lora_with_dirs(cli_runner, config_path, tmp_path):
    """Test merge_lora with custom lora and output directories"""
    lora_dir = tmp_path / "lora"
    output_dir = tmp_path / "output"
    lora_dir.mkdir()

    with patch("axolotl.cli.merge_lora.do_cli") as mock_do_cli:
        result = cli_runner.invoke(
            cli,
            [
                "merge-lora",
                str(config_path),
                "--lora-model-dir",
                str(lora_dir),
                "--output-dir",
                str(output_dir),
            ],
        )
        assert result.exit_code == 0

        mock_do_cli.assert_called_once()
        assert mock_do_cli.call_args.kwargs["config"] == str(config_path)
        assert mock_do_cli.call_args.kwargs["lora_model_dir"] == str(lora_dir)
        assert mock_do_cli.call_args.kwargs["output_dir"] == str(output_dir)


def test_merge_lora_nonexistent_config(cli_runner, tmp_path):
    """Test merge_lora with nonexistent config"""
    config_path = tmp_path / "nonexistent.yml"
    result = cli_runner.invoke(cli, ["merge-lora", str(config_path)])
    assert result.exit_code != 0


def test_merge_lora_nonexistent_lora_dir(cli_runner, config_path, tmp_path):
    """Test merge_lora with nonexistent lora directory"""
    lora_dir = tmp_path / "nonexistent"
    result = cli_runner.invoke(
        cli, ["merge-lora", str(config_path), "--lora-model-dir", str(lora_dir)]
    )
    assert result.exit_code != 0
