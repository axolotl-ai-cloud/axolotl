"""pytest tests for axolotl CLI --version"""

from axolotl.cli.main import cli


def test_print_version(cli_runner):
    """Test that version is printed when --version is used."""

    result = cli_runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "axolotl, version " in result.output
