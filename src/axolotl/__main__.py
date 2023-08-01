"""Axolotl CLI entrypoint"""
import importlib
import logging
import pkgutil
import threading
from pathlib import Path

import click
from accelerate import Accelerator

import axolotl
from axolotl.cli import CTX_ACCELERATOR, CTX_CFG
from axolotl.utils.config import load_config
from axolotl.utils.logging import configure_logging

threading.current_thread().name = "Main"

LOG = logging.getLogger(__name__)


@click.group()
@click.option(
    "--config",
    "-c",
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
    "--log_level",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    help="Sets the logging level",
    default="INFO",
    envvar="LOG_LEVEL",
    required=False,
    allow_from_autoenv=True,
    show_envvar=True,
    show_default=True,
)
@click.option(
    "--log_main_only/--no_log_main_only",
    type=click.types.BOOL,
    help="When set, only logging from the main thread; useful when there are multiple Accelerate processes producing too many log entries to stdout",
    default=False,
    envvar="LOG_MAIN_ONLY",
    required=False,
    allow_from_autoenv=True,
    show_envvar=True,
    show_default=True,
)
@click.version_option(prog_name="axolotl", version=axolotl.__version__)
@click.pass_context
def cli(ctx: click.core.Context, config: str, log_level: str, log_main_only: bool):
    "Axolotl CLI"

    accelerator = Accelerator()

    # Configure logging
    if log_main_only and not accelerator.is_local_main_process:
        logging.disable(logging.CRITICAL)
    else:
        configure_logging(log_level)

    # To avoid weird behavior with multiple ways to initialize Accelerate, we will add the
    # reverence to the CLick context.
    ctx.meta[CTX_ACCELERATOR] = accelerator

    # Need to do an update here so we can don't lose the reference to the cfg "singleton". Decided
    # to keep cfg a singleton vs a Click context object to help maintain compatibility with
    # non-Click scripts in Axolotl.
    loaded_cfg = load_config(config=Path(config))
    axolotl.cfg.update(loaded_cfg)

    # For Click-aware applications, add cfg to the global context
    ctx.meta[CTX_CFG] = axolotl.cfg

    if axolotl.cfg.strict is not None:
        LOG.warning("The 'strict' configuration options has been deprecated")


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
    # By convention, we do not want to use auto envirnment variable names, setting to
    # CHANGEME will hopefully make it more obvious that developers need to set explicitly

    # pylint: disable=no-value-for-parameter
    cli(auto_envvar_prefix="CHANGEME", prog_name="axolotl")
