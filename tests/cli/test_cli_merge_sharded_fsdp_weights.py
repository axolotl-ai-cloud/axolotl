"""pytest tests for axolotl CLI merge_sharded_fsdp_weights command."""
# pylint: disable=duplicate-code
from unittest.mock import patch

from axolotl.cli.main import cli


def test_merge_sharded_fsdp_weights_no_accelerate(cli_runner, config_path):
    """Test merge_sharded_fsdp_weights command without accelerate"""
    with patch("axolotl.cli.merge_sharded_fsdp_weights.do_cli") as mock:
        result = cli_runner.invoke(
            cli, ["merge-sharded-fsdp-weights", str(config_path), "--no-accelerate"]
        )

        assert mock.called
        assert mock.call_args.kwargs["config"] == str(config_path)
        assert result.exit_code == 0


def test_merge_sharded_fsdp_weights_with_model_dir(cli_runner, config_path, tmp_path):
    """Test merge_sharded_fsdp_weights command with model_dir option"""
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    with patch("axolotl.cli.merge_sharded_fsdp_weights.do_cli") as mock:
        result = cli_runner.invoke(
            cli,
            [
                "merge-sharded-fsdp-weights",
                str(config_path),
                "--no-accelerate",
                "--model-dir",
                str(model_dir),
            ],
        )

        assert mock.called
        assert mock.call_args.kwargs["config"] == str(config_path)
        assert mock.call_args.kwargs["model_dir"] == str(model_dir)
        assert result.exit_code == 0


def test_merge_sharded_fsdp_weights_with_save_path(cli_runner, config_path):
    """Test merge_sharded_fsdp_weights command with save_path option"""
    with patch("axolotl.cli.merge_sharded_fsdp_weights.do_cli") as mock:
        result = cli_runner.invoke(
            cli,
            [
                "merge-sharded-fsdp-weights",
                str(config_path),
                "--no-accelerate",
                "--save-path",
                "/path/to/save",
            ],
        )

        assert mock.called
        assert mock.call_args.kwargs["config"] == str(config_path)
        assert mock.call_args.kwargs["save_path"] == "/path/to/save"
        assert result.exit_code == 0
