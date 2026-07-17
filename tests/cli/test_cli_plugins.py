"""Tests for plugin-contributed CLI commands."""

import sys
from importlib.metadata import EntryPoint

import click
import pytest

from axolotl.cli import plugins
from axolotl.cli.main import cli
from axolotl.cli.plugins import PluginCommand, PluginCommandGroup


@click.command()
@click.option("--rank-ratio", type=float, default=0.01)
def dummy_command(rank_ratio):
    """dummy plugin command"""
    click.echo(f"dummy ran rank_ratio={rank_ratio}")


not_a_command = "definitely not a click command"

DUMMY_TARGET = "tests.cli.test_cli_plugins:dummy_command"


@pytest.fixture
def group(monkeypatch):
    """A group whose plugin commands come only from patched entry points."""

    def _build(entry_points_, builtins=None):
        monkeypatch.setattr(plugins, "BUILTIN_COMMANDS", builtins or {})
        monkeypatch.setattr(
            plugins,
            "entry_points",
            lambda group: [
                EntryPoint(name=name, value=value, group=group)
                for name, value in entry_points_.items()
            ],
        )

        built = PluginCommandGroup(name="axolotl")

        @built.command("core")
        def _core():
            """core command"""
            click.echo("core ran")

        return built

    return _build


def test_builtin_command_listed_in_help(cli_runner):
    result = cli_runner.invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "lm-eval" in result.output


def test_help_does_not_import_plugin_module(cli_runner):
    """`--help` renders summaries from the registry, never importing the module."""
    sys.modules.pop("axolotl.integrations.lm_eval.cli", None)

    result = cli_runner.invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "axolotl.integrations.lm_eval.cli" not in sys.modules


def test_entry_point_command_is_listed_and_invoked(cli_runner, group):
    built = group({"dummy": DUMMY_TARGET})

    assert "dummy" in built.list_commands(None)

    result = cli_runner.invoke(built, ["dummy", "--rank-ratio", "0.5"])

    assert result.exit_code == 0
    assert "dummy ran rank_ratio=0.5" in result.output


def test_entry_point_command_owns_its_options(cli_runner, group):
    built = group({"dummy": DUMMY_TARGET})

    result = cli_runner.invoke(built, ["dummy", "--help"])

    assert result.exit_code == 0
    assert "--rank-ratio" in result.output


def test_plugin_cannot_shadow_declared_command(cli_runner, group):
    built = group({"core": DUMMY_TARGET})

    result = cli_runner.invoke(built, ["core"])

    assert result.exit_code == 0
    assert "core ran" in result.output


def test_builtin_takes_precedence_over_entry_point(group):
    builtin = PluginCommand(target=DUMMY_TARGET, short_help="builtin")
    built = group({"dummy": "other.module:command"}, builtins={"dummy": builtin})

    assert built.plugin_commands()["dummy"] == builtin


def test_unknown_command_errors(cli_runner, group):
    built = group({})

    result = cli_runner.invoke(built, ["nope"])

    assert result.exit_code != 0
    assert "No such command" in result.output


def test_broken_plugin_does_not_break_help(cli_runner, group):
    built = group({"broken": "axolotl.does_not_exist:command"})

    result = cli_runner.invoke(built, ["--help"])

    assert result.exit_code == 0
    assert "broken" in result.output


def test_non_command_target_is_rejected(group):
    built = group({"bogus": "tests.cli.test_cli_plugins:not_a_command"})

    with pytest.raises(TypeError):
        built.get_command(None, "bogus").resolve()


def test_target_without_attribute_is_rejected(group):
    built = group({"bogus": "tests.cli.test_cli_plugins"})

    with pytest.raises(ValueError):
        built.get_command(None, "bogus").resolve()
