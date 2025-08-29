"""pytest tests for axolotl CLI inference command."""

from unittest.mock import patch

from axolotl.cli.main import cli


def test_inference_basic(cli_runner, config_path):
    """Test basic inference"""
    with patch("axolotl.cli.inference.do_inference") as mock:
        result = cli_runner.invoke(
            cli,
            ["inference", str(config_path), "--launcher", "python"],
            catch_exceptions=False,
        )

        assert mock.called
        assert result.exit_code == 0


def test_inference_gradio(cli_runner, config_path):
    """Test basic inference (gradio path)"""
    with patch("axolotl.cli.inference.do_inference_gradio") as mock:
        result = cli_runner.invoke(
            cli,
            ["inference", str(config_path), "--launcher", "python", "--gradio"],
            catch_exceptions=False,
        )

        assert mock.called
        assert result.exit_code == 0


def test_inference_with_launcher_args_torchrun(cli_runner, config_path):
    """Test inference with torchrun launcher arguments"""
    with patch("subprocess.run") as mock_subprocess:
        result = cli_runner.invoke(
            cli,
            [
                "inference",
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
        assert "axolotl.cli.inference" in called_cmd


def test_inference_with_launcher_args_accelerate(cli_runner, config_path):
    """Test inference with accelerate launcher arguments"""
    with patch("subprocess.run") as mock_subprocess:
        result = cli_runner.invoke(
            cli,
            [
                "inference",
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
        assert "axolotl.cli.inference" in called_cmd


def test_inference_gradio_with_launcher_args(cli_runner, config_path):
    """Test inference with gradio and launcher arguments"""
    with patch("subprocess.run") as mock_subprocess:
        result = cli_runner.invoke(
            cli,
            [
                "inference",
                str(config_path),
                "--launcher",
                "accelerate",
                "--gradio",
                "--",
                "--num_processes=2",
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        mock_subprocess.assert_called_once()

        # Verify both gradio flag and launcher args are present
        called_cmd = mock_subprocess.call_args.args[0]
        assert called_cmd[0] == "accelerate"
        assert called_cmd[1] == "launch"
        assert "--num_processes=2" in called_cmd
        assert "--gradio" in called_cmd
        assert "-m" in called_cmd
        assert "axolotl.cli.inference" in called_cmd


def test_inference_backward_compatibility_no_launcher_args(cli_runner, config_path):
    """Test that existing inference commands work without launcher args"""
    with patch("subprocess.run") as mock_subprocess:
        result = cli_runner.invoke(
            cli,
            [
                "inference",
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
