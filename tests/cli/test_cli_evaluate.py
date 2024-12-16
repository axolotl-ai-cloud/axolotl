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
        self._test_basic_execution(cli_runner, tmp_path, valid_test_config, "evaluate")

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
                    "--no-accelerate",
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
                    "--no-accelerate",
                ],
                catch_exceptions=False,
            )

            assert result.exit_code == 0
            mock_evaluate.assert_called_once()
            cfg = mock_evaluate.call_args[0][0]
            assert cfg.micro_batch_size == 2
            assert cfg.sequence_len == 128
