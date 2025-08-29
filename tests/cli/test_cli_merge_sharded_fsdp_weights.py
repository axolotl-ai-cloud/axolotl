"""pytest tests for axolotl CLI merge_sharded_fsdp_weights command."""

from unittest.mock import patch

from axolotl.cli.main import cli


def test_merge_sharded_fsdp_weights_no_accelerate(cli_runner, config_path):
    """Test merge_sharded_fsdp_weights command without accelerate"""
    with patch("axolotl.cli.merge_sharded_fsdp_weights.do_cli") as mock:
        result = cli_runner.invoke(
            cli,
            ["merge-sharded-fsdp-weights", str(config_path), "--launcher", "python"],
        )

        assert mock.called
        assert mock.call_args.kwargs["config"] == str(config_path)
        assert result.exit_code == 0


def test_merge_sharded_fsdp_weights_with_launcher_args_torchrun(
    cli_runner, config_path
):
    """Test merge-sharded-fsdp-weights with torchrun launcher arguments"""
    with patch("subprocess.run") as mock_subprocess:
        result = cli_runner.invoke(
            cli,
            [
                "merge-sharded-fsdp-weights",
                str(config_path),
                "--launcher",
                "torchrun",
                "--",
                "--nproc_per_node=2",
                "--nnodes=1",
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        mock_subprocess.assert_called_once()

        # Verify launcher args are passed to torchrun
        called_cmd = mock_subprocess.call_args.args[0]
        assert called_cmd[0] == "torchrun"
        assert "--nproc_per_node=2" in called_cmd
        assert "--nnodes=1" in called_cmd
        assert "-m" in called_cmd
        assert "axolotl.cli.merge_sharded_fsdp_weights" in called_cmd


def test_merge_sharded_fsdp_weights_with_launcher_args_accelerate(
    cli_runner, config_path
):
    """Test merge-sharded-fsdp-weights with accelerate launcher arguments"""
    with patch("subprocess.run") as mock_subprocess:
        result = cli_runner.invoke(
            cli,
            [
                "merge-sharded-fsdp-weights",
                str(config_path),
                "--launcher",
                "accelerate",
                "--",
                "--config_file=accelerate_config.yml",
                "--num_processes=4",
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        mock_subprocess.assert_called_once()

        # Verify launcher args are passed to accelerate
        called_cmd = mock_subprocess.call_args.args[0]
        assert called_cmd[0] == "accelerate"
        assert called_cmd[1] == "launch"
        assert "--config_file=accelerate_config.yml" in called_cmd
        assert "--num_processes=4" in called_cmd
        assert "-m" in called_cmd
        assert "axolotl.cli.merge_sharded_fsdp_weights" in called_cmd


def test_merge_sharded_fsdp_weights_backward_compatibility_no_launcher_args(
    cli_runner, config_path
):
    """Test that existing merge-sharded-fsdp-weights commands work without launcher args"""
    with patch("subprocess.run") as mock_subprocess:
        result = cli_runner.invoke(
            cli,
            [
                "merge-sharded-fsdp-weights",
                str(config_path),
                "--launcher",
                "accelerate",
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        mock_subprocess.assert_called_once()

        # Verify no launcher args contamination
        called_cmd = mock_subprocess.call_args.args[0]
        assert called_cmd[0] == "accelerate"
        assert called_cmd[1] == "launch"
        # Should not contain any extra launcher args
        launcher_section = called_cmd[2 : called_cmd.index("-m")]
        assert len(launcher_section) == 0  # No launcher args between 'launch' and '-m'
