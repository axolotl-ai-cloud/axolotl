"""pytest tests for axolotl CLI evaluate command."""
from unittest.mock import patch

from axolotl.cli.main import cli


def test_evaluate_cli_validation(cli_runner):
    """Test CLI validation"""
    # Test missing config file
    result = cli_runner.invoke(cli, ["evaluate", "--no-accelerate"])
    assert result.exit_code != 0

    # Test non-existent config file
    result = cli_runner.invoke(cli, ["evaluate", "nonexistent.yml", "--no-accelerate"])
    assert result.exit_code != 0
    assert "Error: Invalid value for 'CONFIG'" in result.output


def test_evaluate_basic_execution(cli_runner, tmp_path, valid_test_config):
    """Test basic successful execution"""
    config_path = tmp_path / "config.yml"
    config_path.write_text(valid_test_config)

    with patch("subprocess.run") as mock:
        result = cli_runner.invoke(cli, ["evaluate", str(config_path)])

        assert mock.called
        assert mock.call_args.args[0] == [
            "accelerate",
            "launch",
            "-m",
            "axolotl.cli.evaluate",
            str(config_path),
            "--debug-num-examples",
            "0",
        ]
        assert mock.call_args.kwargs == {"check": True}
        assert result.exit_code == 0


def test_evaluate_basic_execution_no_accelerate(
    cli_runner, tmp_path, valid_test_config
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
                "--micro-batch-size",
                "2",
                "--no-accelerate",
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        mock_evaluate.assert_called_once()
        cfg = mock_evaluate.call_args[0][0]
        assert cfg.micro_batch_size == 2


def test_evaluate_cli_overrides(cli_runner, tmp_path, valid_test_config):
    """Test CLI arguments properly override config values"""
    config_path = tmp_path / "config.yml"
    output_dir = tmp_path / "model-out"

    test_config = valid_test_config.replace(
        "output_dir: model-out", f"output_dir: {output_dir}"
    )
    config_path.write_text(test_config)

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


def test_evaluate_with_rl_dpo(cli_runner, tmp_path, valid_test_config):
    """Test evaluation with DPO reinforcement learning"""
    config_path = tmp_path / "config.yml"
    config_path.write_text(valid_test_config)

    with patch("axolotl.cli.evaluate.do_evaluate") as mock_evaluate:
        result = cli_runner.invoke(
            cli,
            [
                "evaluate",
                str(config_path),
                "--rl",
                "dpo",
                "--no-accelerate",
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        mock_evaluate.assert_called_once()
        cfg = mock_evaluate.call_args[0][0]
        assert cfg.rl == "dpo"
