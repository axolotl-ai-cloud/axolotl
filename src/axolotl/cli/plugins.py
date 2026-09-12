"""CLI commands contributed by axolotl integrations and third-party plugins."""

import importlib
from importlib.metadata import entry_points
from typing import NamedTuple

import click

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

ENTRY_POINT_GROUP = "axolotl.cli_commands"


class PluginCommand(NamedTuple):
    """A command to import from `target` (`<module>:<attribute>`) when it is used."""

    target: str
    # duplicated from the command's own docstring so that `axolotl --help` can render
    # the summary line without importing the module, which for an in-tree integration
    # pulls in torch via the package's `__init__`
    short_help: str = ""


BUILTIN_COMMANDS: dict[str, PluginCommand] = {
    "lm-eval": PluginCommand(
        target="axolotl.integrations.lm_eval.cli:lm_eval",
        short_help="use lm eval to evaluate a trained language model",
    ),
}


def _import_command(target: str) -> click.Command:
    module_name, separator, attribute = target.partition(":")
    if not separator:
        raise ValueError(f"expected '<module>:<attribute>', got '{target}'")

    command = getattr(importlib.import_module(module_name), attribute)
    if not isinstance(command, click.Command):
        raise TypeError(f"'{target}' is not a click command")

    return command


class LazyCommand(click.Command):
    """Stands in for a plugin command until it is actually used.

    Click resolves every command to render `axolotl --help`, so the real module is
    imported only once the command is invoked or its own `--help` is requested.
    """

    def __init__(self, name: str, spec: PluginCommand):
        super().__init__(name, short_help=spec.short_help)
        self.spec = spec
        self._command: click.Command | None = None

    def resolve(self) -> click.Command:
        if self._command is None:
            self._command = _import_command(self.spec.target)
        return self._command

    def get_short_help_str(self, limit: int = 45) -> str:
        if self.short_help:
            return self.short_help

        try:
            return self.resolve().get_short_help_str(limit)
        except Exception as exc:
            LOG.warning(
                "could not load plugin command '%s' from '%s': %s",
                self.name,
                self.spec.target,
                exc,
            )
            return ""

    def make_context(self, info_name, args, parent=None, **extra) -> click.Context:
        return self.resolve().make_context(info_name, args, parent=parent, **extra)

    def invoke(self, ctx: click.Context):
        return self.resolve().invoke(ctx)


class PluginCommandGroup(click.Group):
    """Click group that also serves commands contributed by plugins.

    Commands come from `BUILTIN_COMMANDS` and from the `axolotl.cli_commands` entry
    point group. Commands declared on the group take precedence, so plugins cannot
    shadow core commands.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._plugin_commands: dict[str, PluginCommand] | None = None

    def plugin_commands(self) -> dict[str, PluginCommand]:
        # cached because entry point discovery reads installed metadata off disk and
        # click resolves every command to render `--help`
        if self._plugin_commands is None:
            commands = dict(BUILTIN_COMMANDS)
            for entry_point in entry_points(group=ENTRY_POINT_GROUP):
                if entry_point.name in commands:
                    LOG.warning(
                        "ignoring plugin command '%s' from '%s': name is already taken",
                        entry_point.name,
                        entry_point.value,
                    )
                    continue
                commands[entry_point.name] = PluginCommand(target=entry_point.value)
            self._plugin_commands = commands

        return self._plugin_commands

    def list_commands(self, ctx: click.Context) -> list[str]:
        return sorted({*super().list_commands(ctx), *self.plugin_commands()})

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        command = super().get_command(ctx, cmd_name)
        if command is not None:
            return command

        spec = self.plugin_commands().get(cmd_name)
        return LazyCommand(cmd_name, spec) if spec is not None else None
