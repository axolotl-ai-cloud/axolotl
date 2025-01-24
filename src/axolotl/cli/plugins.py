"""Module for adding click CLI commands from axolotl plugins."""

import logging

import click

from axolotl.cli.utils import add_options_from_config, add_options_from_dataclass
from axolotl.logging_config import configure_logging
from axolotl.utils.config.models.input.v0_4_1 import AxolotlInputConfig

configure_logging()
LOG = logging.getLogger(__name__)


def setup_plugin_commands(cli: click.core.Group) -> None:
    """
    Setup CLI commands for available plugins.

    Args:
        cli: Click CLI object to add plugin CLI options to.
    """
    try:
        from axolotl_diff_transformer.convert_diff_transformer import do_cli
        from axolotl_diff_transformer.plugin.cli import ConvertDiffTransformerCliArgs

        @cli.command()
        @click.argument("config", type=click.Path(exists=True, path_type=str))
        @add_options_from_dataclass(ConvertDiffTransformerCliArgs)
        @add_options_from_config(AxolotlInputConfig)
        def convert_diff_transformer(config: str, **kwargs):
            """Convert model attention layers to differential attention layers."""
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            do_cli(config=config, **kwargs)

    except ImportError as exc:
        LOG.debug("axolotl-diff-transformer not found: %s", exc)
