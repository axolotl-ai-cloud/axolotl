"""pytest tests for axolotl CLI train command."""
from unittest.mock import MagicMock, patch

from axolotl.cli.main import cli


def test_train_cli_validation(cli_runner):
    """Test CLI validation"""
    # Test missing config file
    result = cli_runner.invoke(cli, ["train", "--no-accelerate"])
    assert result.exit_code != 0

    # Test non-existent config file
    result = cli_runner.invoke(cli, ["train", "nonexistent.yml", "--no-accelerate"])
    assert result.exit_code != 0
    assert "Error: Invalid value for 'CONFIG'" in result.output


def test_train_basic_execution(cli_runner, tmp_path, valid_test_config):
    """Test basic successful execution"""
    config_path = tmp_path / "config.yml"
    config_path.write_text(valid_test_config)

    with patch("subprocess.run") as mock:
        result = cli_runner.invoke(cli, ["train", str(config_path)])

        assert mock.called
        assert mock.call_args.args[0] == [
            "accelerate",
            "launch",
            "-m",
            "axolotl.cli.train",
            str(config_path),
            "--debug-num-examples",
            "0",
        ]
        assert mock.call_args.kwargs == {"check": True}
        assert result.exit_code == 0


def test_train_basic_execution_no_accelerate(cli_runner, tmp_path, valid_test_config):
    """Test basic successful execution"""
    config_path = tmp_path / "config.yml"
    config_path.write_text(valid_test_config)

    with patch("axolotl.cli.train.train") as mock_train:
        mock_train.return_value = (MagicMock(), MagicMock())

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


def test_train_cli_overrides(cli_runner, tmp_path, valid_test_config):
    """Test CLI arguments properly override config values"""
    config_path = tmp_path / "config.yml"
    output_dir = tmp_path / "model-out"

    test_config = valid_test_config.replace(
        "output_dir: model-out", f"output_dir: {output_dir}"
    )
    config_path.write_text(test_config)

    with patch("axolotl.cli.train.train") as mock_train:
        mock_train.return_value = (MagicMock(), MagicMock())

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
