"""Logging configuration settings"""

import logging
import os
import sys
from logging import _levelToName
from logging.config import dictConfig


def configure_logging(log_level: str = "DEBUG"):
    """Configure with default logging"""

    # Set transformers log level,
    # see: https://huggingface.co/docs/transformers/main_classes/logging
    os.environ["TRANSFORMERS_VERBOSITY"] = log_level.lower()

    # Set accelerate log level,
    # see: https://huggingface.co/docs/accelerate/package_reference/logging
    os.environ["ACCELERATE_LOG_LEVEL"] = log_level

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "simple": {
                    "format": "[%(asctime)s] [%(levelname)s] [%(name)s.%(funcName)s:%(lineno)d] [PID:%(process)d] %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "simple",
                    "filters": [],
                    "stream": sys.stdout,
                },
            },
            "loggers": {
                "": {  # Root logger
                    "level": log_level,
                    "propagate": False,
                    "filters": [],
                    "handlers": ["console"],
                }
            },
        }
    )


def print_loggers():
    """Function to print the current logging hierarchy"""
    loggers_dict = logging.Logger.manager.loggerDict
    for _, logger in {
        "root": logging.Logger.manager.root,
        **loggers_dict,
    }.items():
        if isinstance(logger, logging.Logger):
            print(
                f"Logger: {logger.name} ({_levelToName[logger.level]}), Propagate: {logger.propagate}, Disabled: {logger.disabled}, Parent: {logger.parent.name if logger.parent is not None else ''}"
            )
