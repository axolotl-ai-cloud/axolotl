"""Tests for evaluate CLI command."""

from unittest.mock import patch

from axolotl.cli.main import cli

from .test_cli_base import BaseCliTest


class TestEvaluateCommand(BaseCliTest):
    """Test cases for evaluate command."""

    cli = cli

    def test_evaluate_cli_validation(self, cli_runner):
        """Test CLI validation"""
        self._test_cli_validation(cli_runner, "evaluate")

    def test_evaluate_basic_execution(self, cli_runner, tmp_path, valid_test_config):
        """Test basic successful execution"""
        self._test_basic_execution(
            cli_runner, tmp_path, valid_test_config, "evaluate", train=False
        )

    def test_evaluate_basic_execution_no_accelerate(
        self, cli_runner, tmp_path, valid_test_config
    ):
        """Test basic successful execution without accelerate"""
        config_path = tmp_path / "config.yml"
        config_path.write_text(valid_test_config)

        with patch("axolotl.cli.evaluate.do_evaluate") as mock_evaluate:
            result = cli_runner.invoke(
                cli,
                [
                    "evaluate",
                    str(config_path),
                    "--launcher",
                    "python",
                ],
                catch_exceptions=False,
            )

            assert result.exit_code == 0
            mock_evaluate.assert_called_once()

    def test_evaluate_cli_overrides(self, cli_runner, tmp_path, valid_test_config):
        """Test CLI arguments properly override config values"""
        config_path = self._test_cli_overrides(tmp_path, valid_test_config)

        with patch("axolotl.cli.evaluate.do_evaluate") as mock_evaluate:
            result = cli_runner.invoke(
                cli,
                [
                    "evaluate",
                    str(config_path),
                    "--micro-batch-size",
                    "2",
                    "--sequence-len",
                    "128",
                    "--launcher",
                    "python",
                ],
                catch_exceptions=False,
            )

            assert result.exit_code == 0
            mock_evaluate.assert_called_once()
            cfg = mock_evaluate.call_args[0][0]
            assert cfg.micro_batch_size == 2
            assert cfg.sequence_len == 128

    def test_evaluate_with_launcher_args_torchrun(
        self, cli_runner, tmp_path, valid_test_config
    ):
        """Test evaluate with torchrun launcher arguments"""
        config_path = tmp_path / "config.yml"
        config_path.write_text(valid_test_config)

        with patch("subprocess.run") as mock_subprocess:
            result = cli_runner.invoke(
                cli,
                [
                    "evaluate",
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
            assert "axolotl.cli.evaluate" in called_cmd

    def test_evaluate_with_launcher_args_accelerate(
        self, cli_runner, tmp_path, valid_test_config
    ):
        """Test evaluate with accelerate launcher arguments"""
        config_path = tmp_path / "config.yml"
        config_path.write_text(valid_test_config)

        with patch("subprocess.run") as mock_subprocess:
            result = cli_runner.invoke(
                cli,
                [
                    "evaluate",
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
            assert "axolotl.cli.evaluate" in called_cmd

    def test_evaluate_backward_compatibility_no_launcher_args(
        self, cli_runner, tmp_path, valid_test_config
    ):
        """Test that existing evaluate commands work without launcher args"""
        config_path = tmp_path / "config.yml"
        config_path.write_text(valid_test_config)

        with patch("subprocess.run") as mock_subprocess:
            result = cli_runner.invoke(
                cli,
                [
                    "evaluate",
                    str(config_path),
                    "--launcher",
                    "accelerate",
                    "--micro-batch-size",
                    "2",
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
            assert (
                len(launcher_section) == 0
            )  # No launcher args between 'launch' and '-m'
