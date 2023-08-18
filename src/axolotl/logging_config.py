"""
Common logging module for axolotl
"""

import logging
import os
import sys
from logging import Formatter, _levelToName
from logging.config import dictConfig

from colorama import Fore, Style, init


class ColorfulFormatter(Formatter):
    """
    Formatter to add coloring to log messages by log type
    """

    COLORS = {
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.RED + Style.BRIGHT,
    }

    def format(self, record):
        log_message = super().format(record)
        return self.COLORS.get(record.levelname, "") + log_message + Fore.RESET


def configure_logging(log_level: str = os.getenv("LOG_LEVEL", "INFO")):
    """Configure with default logging"""

    # Set transformers log level,
    # see: https://huggingface.co/docs/transformers/main_classes/logging
    os.environ["TRANSFORMERS_VERBOSITY"] = log_level.lower()

    # Set accelerate log level,
    # see: https://huggingface.co/docs/accelerate/package_reference/logging
    os.environ["ACCELERATE_LOG_LEVEL"] = log_level

    init()  # Initialize colorama
    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "simple": {
                    "format": "[%(asctime)s] [%(levelname)s] [%(name)s.%(funcName)s:%(lineno)d] [PID:%(process)d] %(message)s",
                },
                "colorful": {
                    "()": ColorfulFormatter,
                    "format": "[%(asctime)s] [%(levelname)s] [%(name)s.%(funcName)s:%(lineno)d] [PID:%(process)d] %(message)s",
                },
            },
            "filters": {},
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "simple",
                    "filters": [],
                    "stream": sys.stdout,
                },
                "color_console": {
                    "class": "logging.StreamHandler",
                    "formatter": "colorful",
                    "filters": [],
                    "stream": sys.stdout,
                },
            },
            "loggers": {
                "": {
                    "handlers": ["color_console"],
                    "level": log_level,
                    "propagate": False,
                },
            },
        }
    )


def print_loggers():
    """Function to print the current logging hierarchy, helpful when debugging"""
    loggers_dict = logging.Logger.manager.loggerDict
    for _, logger in {
        "root": logging.Logger.manager.root,
        **loggers_dict,
    }.items():
        if isinstance(logger, logging.Logger):
            print(
                f"Logger: {logger.name} ({_levelToName[logger.level]}), Propagate: {logger.propagate}, Disabled: {logger.disabled}, Parent: {logger.parent.name if logger.parent is not None else ''}"
            )
