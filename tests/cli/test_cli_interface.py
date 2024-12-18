"""General pytest tests for axolotl.cli.main interface."""

from axolotl.cli.main import build_command, cli


def test_build_command():
    """Test converting dict of options to CLI arguments"""
    base_cmd = ["accelerate", "launch"]
    options = {
        "learning_rate": 1e-4,
        "batch_size": 8,
        "debug": True,
        "use_fp16": False,
        "null_value": None,
    }

    result = build_command(base_cmd, options)
    assert result == [
        "accelerate",
        "launch",
        "--learning-rate",
        "0.0001",
        "--batch-size",
        "8",
        "--debug",
    ]


def test_invalid_command_options(cli_runner):
    """Test handling of invalid command options"""
    result = cli_runner.invoke(
        cli,
        [
            "train",
            "config.yml",
            "--invalid-option",
            "value",
        ],
    )
    assert result.exit_code != 0
    assert "No such option" in result.output


def test_required_config_argument(cli_runner):
    """Test commands fail properly when config argument is missing"""
    result = cli_runner.invoke(cli, ["train"])
    assert result.exit_code != 0
    assert "Missing argument 'CONFIG'" in result.output
