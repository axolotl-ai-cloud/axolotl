"""Tests for train CLI command."""

from unittest.mock import MagicMock, patch

from axolotl.cli.main import cli

from .test_cli_base import BaseCliTest


class TestTrainCommand(BaseCliTest):
    """Test cases for train command."""

    cli = cli

    def test_train_cli_validation(self, cli_runner):
        """Test CLI validation"""
        self._test_cli_validation(cli_runner, "train")

    def test_train_basic_execution(self, cli_runner, tmp_path, valid_test_config):
        """Test basic successful execution"""
        self._test_basic_execution(cli_runner, tmp_path, valid_test_config, "train")

    def test_train_basic_execution_no_accelerate(
        self, cli_runner, tmp_path, valid_test_config
    ):
        """Test basic successful execution without accelerate"""
        config_path = tmp_path / "config.yml"
        config_path.write_text(valid_test_config)

        with patch("axolotl.cli.train.train") as mock_train:
            mock_train.return_value = (MagicMock(), MagicMock(), MagicMock())

            result = cli_runner.invoke(
                cli,
                [
                    "train",
                    str(config_path),
                    "--no-accelerate",
                ],
                catch_exceptions=False,
            )

            assert result.exit_code == 0
            mock_train.assert_called_once()

    def test_train_cli_overrides(self, cli_runner, tmp_path, valid_test_config):
        """Test CLI arguments properly override config values"""
        config_path = self._test_cli_overrides(tmp_path, valid_test_config)

        with patch("axolotl.cli.train.train") as mock_train:
            mock_train.return_value = (MagicMock(), MagicMock(), MagicMock())

            result = cli_runner.invoke(
                cli,
                [
                    "train",
                    str(config_path),
                    "--learning-rate",
                    "1e-4",
                    "--micro-batch-size",
                    "2",
                    "--no-accelerate",
                ],
                catch_exceptions=False,
            )

            assert result.exit_code == 0
            mock_train.assert_called_once()
            cfg = mock_train.call_args[1]["cfg"]
            assert cfg["learning_rate"] == 1e-4
            assert cfg["micro_batch_size"] == 2
