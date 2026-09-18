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
        "--learning-rate=0.0001",
        "--batch-size=8",
        "--debug=True",
        "--use-fp16=False",
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
    assert "does not exist" in result.output


def test_required_config_argument(cli_runner):
    """Test commands fail properly when config argument is missing"""
    result = cli_runner.invoke(cli, ["train"])
    assert result.exit_code != 0
    assert "Missing argument 'CONFIG'" in result.output


def test_config_schema_runtime(cli_runner):
    """Test that config-schema --runtime dumps the runtime file schema"""
    result = cli_runner.invoke(cli, ["config-schema", "--runtime"])

    assert result.exit_code == 0
    for key in ("launcher", "ray", "torchrun", "accelerate", "env"):
        assert f'"{key}"' in result.output


def test_ray_group_help(cli_runner):
    """Test that the ray command group lists its subcommands without ray installed"""
    result = cli_runner.invoke(cli, ["ray", "--help"])

    assert result.exit_code == 0
    for subcommand in ("up", "down", "status"):
        assert subcommand in result.output
