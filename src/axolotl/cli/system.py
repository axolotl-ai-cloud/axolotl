"""
The Axolotl system module contains CLI tools relevant to debugging and system-level functions.
"""
import json
import logging
from typing import Any, Dict, Tuple

import click

from axolotl import cfg
from axolotl.cli.option_groups import all_option_group
from axolotl.utils.config import update_config

LOG = logging.getLogger(__name__)


@click.group(name="system", help=__doc__)
def system_group():
    """System command group"""


@system_group.command(name="config")
@click.option(
    "--pretty/--no-pretty",
    type=click.types.BOOL,
    help="Pretty print the JSON output",
    default=True,
    show_envvar=False,
    allow_from_autoenv=False,
)
@click.option(
    "--filter",
    "-f",
    "filter_strings",
    type=click.types.STRING,
    help="Output only JSON keys that match the provided filter expressions",
    multiple=True,
    show_envvar=False,
    allow_from_autoenv=False,
)
@all_option_group()
def config(pretty: bool, filter_strings: Tuple[str], **kwargs: Dict[str, Any]):
    """Applies override logic, performs validation, and outputs the derived configuration. Useful for developing configuration overrides and unit testing."""

    # Override default configuration
    update_config(overrides=kwargs)

    # Optionally apply json filter
    if len(filter_strings) != 0:
        derived_cfg = {
            key: value for key, value in cfg.items() if key in filter_strings
        }
    else:
        derived_cfg = cfg

    # Print effective config
    click.echo(json.dumps(derived_cfg, indent=2 if pretty else None, sort_keys=True))
