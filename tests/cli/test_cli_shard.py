"""pytest tests for axolotl CLI shard command."""
# pylint: disable=duplicate-code
from unittest.mock import patch

from axolotl.cli.main import cli


def test_shard_with_accelerate(cli_runner, config_path):
    """Test shard command with accelerate"""
    with patch("subprocess.run") as mock:
        result = cli_runner.invoke(cli, ["shard", str(config_path), "--accelerate"])

        assert mock.called
        assert mock.call_args.args[0] == [
            "accelerate",
            "launch",
            "-m",
            "axolotl.cli.shard",
            str(config_path),
            "--debug-num-examples",
            "0",
        ]
        assert mock.call_args.kwargs == {"check": True}
        assert result.exit_code == 0


def test_shard_no_accelerate(cli_runner, config_path):
    """Test shard command without accelerate"""
    with patch("axolotl.cli.shard.do_cli") as mock:
        result = cli_runner.invoke(cli, ["shard", str(config_path), "--no-accelerate"])

        assert mock.called
        assert result.exit_code == 0


def test_shard_with_model_dir(cli_runner, config_path, tmp_path):
    """Test shard command with model_dir option"""
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    with patch("axolotl.cli.shard.do_cli") as mock:
        result = cli_runner.invoke(
            cli,
            [
                "shard",
                str(config_path),
                "--no-accelerate",
                "--model-dir",
                str(model_dir),
            ],
            catch_exceptions=False,
        )

        assert mock.called
        assert mock.call_args.kwargs["config"] == str(config_path)
        assert mock.call_args.kwargs["model_dir"] == str(model_dir)
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
