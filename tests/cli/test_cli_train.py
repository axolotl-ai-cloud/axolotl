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
        self._test_basic_execution(
            cli_runner, tmp_path, valid_test_config, "train", train=True
        )

    def test_train_basic_execution_no_accelerate(
        self, cli_runner, tmp_path, valid_test_config
    ):
        """Test basic successful execution without accelerate"""
        config_path = tmp_path / "config.yml"
        config_path.write_text(valid_test_config)

        with patch("axolotl.cli.train.train") as mock_train:
            mock_train.return_value = (MagicMock(), MagicMock(), MagicMock())
            with patch("axolotl.cli.train.load_datasets") as mock_load_datasets:
                mock_load_datasets.return_value = MagicMock()

                result = cli_runner.invoke(
                    cli,
                    [
                        "train",
                        str(config_path),
                        "--launcher",
                        "python",
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
            with patch("axolotl.cli.train.load_datasets") as mock_load_datasets:
                mock_load_datasets.return_value = MagicMock()

                result = cli_runner.invoke(
                    cli,
                    [
                        "train",
                        str(config_path),
                        "--learning-rate=1e-4",
                        "--micro-batch-size=2",
                        "--launcher",
                        "python",
                    ],
                    catch_exceptions=False,
                )

                assert result.exit_code == 0
                mock_train.assert_called_once()
                cfg = mock_train.call_args[1]["cfg"]
                assert cfg["learning_rate"] == 1e-4
                assert cfg["micro_batch_size"] == 2

    def test_train_with_launcher_args_torchrun(
        self, cli_runner, tmp_path, valid_test_config
    ):
        """Test train with torchrun launcher arguments"""
        config_path = tmp_path / "config.yml"
        config_path.write_text(valid_test_config)

        with patch("os.execvpe") as mock_subprocess:
            result = cli_runner.invoke(
                cli,
                [
                    "train",
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
            called_cmd = mock_subprocess.call_args.args[1]
            assert called_cmd[0] == "torchrun"
            assert "--nproc_per_node=2" in called_cmd
            assert "--nnodes=1" in called_cmd
            assert "-m" in called_cmd
            assert "axolotl.cli.train" in called_cmd

    def test_train_with_launcher_args_accelerate(
        self, cli_runner, tmp_path, valid_test_config
    ):
        """Test train with accelerate launcher arguments"""
        config_path = tmp_path / "config.yml"
        config_path.write_text(valid_test_config)

        with patch("os.execvpe") as mock_subprocess:
            result = cli_runner.invoke(
                cli,
                [
                    "train",
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
            assert mock_subprocess.call_args.args[0] == "accelerate"
            called_cmd = mock_subprocess.call_args.args[1]
            assert called_cmd[0] == "accelerate"
            assert called_cmd[1] == "launch"
            assert "--config_file=accelerate_config.yml" in called_cmd
            assert "--num_processes=4" in called_cmd
            assert "-m" in called_cmd
            assert "axolotl.cli.train" in called_cmd

    def test_train_backward_compatibility_no_launcher_args(
        self, cli_runner, tmp_path, valid_test_config
    ):
        """Test that existing train commands work without launcher args"""
        config_path = tmp_path / "config.yml"
        config_path.write_text(valid_test_config)

        with patch("os.execvpe") as mock_subprocess:
            result = cli_runner.invoke(
                cli,
                [
                    "train",
                    str(config_path),
                    "--launcher",
                    "accelerate",
                    "--learning-rate",
                    "1e-4",
                ],
                catch_exceptions=False,
            )

            assert result.exit_code == 0
            mock_subprocess.assert_called_once()

            # Verify no launcher args contamination
            assert mock_subprocess.call_args.args[0] == "accelerate"
            called_cmd = mock_subprocess.call_args.args[1]
            assert called_cmd[0] == "accelerate"
            assert called_cmd[1] == "launch"
            # Should not contain any extra launcher args
            launcher_section = called_cmd[2 : called_cmd.index("-m")]
            assert (
                len(launcher_section) == 0
            )  # No launcher args between 'launch' and '-m'

    def test_train_mixed_args_with_launcher_args(
        self, cli_runner, tmp_path, valid_test_config
    ):
        """Test train with both regular CLI args and launcher args"""
        config_path = tmp_path / "config.yml"
        config_path.write_text(valid_test_config)

        with patch("os.execvpe") as mock_subprocess:
            result = cli_runner.invoke(
                cli,
                [
                    "train",
                    str(config_path),
                    "--launcher",
                    "torchrun",
                    "--learning-rate",
                    "2e-4",
                    "--micro-batch-size",
                    "4",
                    "--",
                    "--nproc_per_node=8",
                ],
                catch_exceptions=False,
            )

            assert result.exit_code == 0
            mock_subprocess.assert_called_once()

            assert mock_subprocess.call_args.args[0] == "torchrun"
            called_cmd = mock_subprocess.call_args.args[1]
            # Verify launcher args
            assert "--nproc_per_node=8" in called_cmd
            # Verify axolotl args are also present
            assert "--learning-rate=2e-4" in called_cmd
            assert "--micro-batch-size=4" in called_cmd

    def test_train_cloud_with_launcher_args(
        self, cli_runner, tmp_path, valid_test_config
    ):
        """Test train with cloud and launcher arguments"""
        config_path = tmp_path / "config.yml"
        config_path.write_text(valid_test_config)

        cloud_path = tmp_path / "cloud.yml"
        cloud_path.write_text("provider: modal\ngpu: a100")

        with patch("axolotl.cli.cloud.do_cli_train") as mock_cloud_train:
            result = cli_runner.invoke(
                cli,
                [
                    "train",
                    str(config_path),
                    "--cloud",
                    str(cloud_path),
                    "--launcher",
                    "torchrun",
                    "--",
                    "--nproc_per_node=4",
                    "--nnodes=2",
                ],
                catch_exceptions=False,
            )

            assert result.exit_code == 0
            mock_cloud_train.assert_called_once()

            # Verify cloud training was called with launcher args
            call_kwargs = mock_cloud_train.call_args.kwargs
            assert call_kwargs["launcher"] == "torchrun"
            assert call_kwargs["launcher_args"] == ["--nproc_per_node=4", "--nnodes=2"]
