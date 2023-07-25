"""Axolotl CLI entrypoint"""
import importlib
import pkgutil
import threading
from pathlib import Path

import click

import axolotl
from axolotl.utils.config import load_config
from axolotl.utils.logging import configure_logging

threading.current_thread().name = "Main"


@click.group()
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=True, file_okay=True, readable=True),
    help="Path to configuration file, if set to a directory axolotl will prompt for the config file",
    default=Path("configs/"),
    required=False,
    envvar="AXOLOTL_CONFIG",
    allow_from_autoenv=True,
    show_envvar=True,
    show_default=True,
)
@click.option(
    "--log-level",
    type=click.Choice(
        ["DEBUG", "INFO", "WARN", "WARNING", "ERROR"], case_sensitive=False
    ),
    help="Sets the logging level",
    default="INFO",
    envvar="LOG_LEVEL",
    required=False,
    allow_from_autoenv=True,
    show_envvar=True,
    show_default=True,
)
@click.version_option(prog_name="axolotl", version=axolotl.__version__)
def cli(config: str, log_level: str):
    "Axolotl CLI"

    # Configure logging
    configure_logging(log_level)

    # Need to do an update here so we can don't lose the refernce to the cfg "singleton"
    loaded_cfg = load_config(config=Path(config))
    axolotl.cfg.update(loaded_cfg)


# Dynamically load all Click command groups under the axolotl.cli package
for module_info in pkgutil.walk_packages(
    axolotl.cli.__path__, axolotl.cli.__name__ + "."
):
    module = importlib.import_module(module_info.name)
    for item_name in dir(module):
        item = getattr(module, item_name)
        if isinstance(item, click.core.Group):
            cli.add_command(item)


if __name__ == "__main__":
    # By convention, we do not want to use auto envirnment variable names
    # pylint: disable=no-value-for-parameter
    cli(auto_envvar_prefix="CHANGEME", prog_name="axolotl")
