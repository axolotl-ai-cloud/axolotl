"""Base test class for CLI commands."""

from pathlib import Path
from unittest.mock import patch

from axolotl.cli.main import cli


class BaseCliTest:
    """Base class for CLI command tests."""

    def _test_cli_validation(self, cli_runner, command: str):
        """Test CLI validation for a command.

        Args:
            cli_runner: CLI runner fixture
            command: Command to test (train/evaluate)
        """
        # Test missing config file
        result = cli_runner.invoke(cli, [command, "--launcher", "python"])
        assert result.exit_code != 0

        # Test non-existent config file
        result = cli_runner.invoke(
            cli, [command, "nonexistent.yml", "--launcher", "python"]
        )
        assert result.exit_code != 0
        assert "Error: Invalid value for 'CONFIG'" in result.output

    def _test_basic_execution(
        self,
        cli_runner,
        tmp_path: Path,
        valid_test_config: str,
        command: str,
        train: bool = True,
    ):
        """Test basic execution with accelerate.

        Args:
            cli_runner: CLI runner fixture
            tmp_path: Temporary path fixture
            valid_test_config: Valid config fixture
            command: Command to test (train/evaluate)
            train: Whether to test training (default) or evaluation
        """
        config_path = tmp_path / "config.yml"
        config_path.write_text(valid_test_config)

        mock_fn = "os.execvpe" if command == "train" else "subprocess.run"

        with patch(mock_fn) as mock:
            result = cli_runner.invoke(cli, [command, str(config_path)])

            assert mock.called

            expected = [
                "accelerate",
                "launch",
                "-m",
                f"axolotl.cli.{command}",
                str(config_path),
                "--debug=False",
                "--debug-text-only=False",
                "--debug-num-examples=0",
            ]
            if train:
                expected.append("--shard=False")

            if command == "train":
                assert mock.call_args.args[0] == "accelerate"
                assert mock.call_args.args[1] == expected
            else:
                assert mock.call_args.args[0] == expected
                assert mock.call_args.kwargs == {"check": True}
            assert result.exit_code == 0

    def _test_cli_overrides(self, tmp_path: Path, valid_test_config: str):
        """Test CLI argument overrides.

        Args:
            tmp_path: Temporary path fixture
            valid_test_config: Valid config fixture
            command: Command to test (train/evaluate)
        """
        config_path = tmp_path / "config.yml"
        output_dir = tmp_path / "model-out"

        test_config = valid_test_config.replace(
            "output_dir: model-out", f"output_dir: {output_dir}"
        )
        config_path.write_text(test_config)
        return config_path
