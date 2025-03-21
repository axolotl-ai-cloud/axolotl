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
