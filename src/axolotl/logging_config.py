"""
Common logging module for axolotl
"""

import logging
import os
import sys
from logging import Formatter, Logger, LogRecord
from logging.config import dictConfig
from typing import Any, Dict

from colorama import Fore, Style, init


class AxolotlOrWarnErrorFilter(logging.Filter):
    """
    Allows ANY WARNING+ or ERROR+
    Allows axolotl.* at INFO or higher
    Drops all other records (i.e. non-axolotl.INFO, DEBUG, etc.)
    """

    def filter(self, record: LogRecord) -> bool:
        # allow WARNING+ or ERROR+ from anywhere
        if record.levelno >= logging.WARNING:
            return True

        # else allow axolotl.* at INFO or higher
        return record.name.startswith("axolotl") and record.levelno >= logging.INFO


class AxolotlLogger(Logger):
    """A Logger that automatically rejects non-axolotl INFOs."""

    def __init__(self, name: str, level: int = logging.NOTSET):
        super().__init__(name, level)

        # set global filter on the logger itself
        self.addFilter(AxolotlOrWarnErrorFilter())


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
        record.rank = int(os.getenv("LOCAL_RANK", "0"))
        log_message = super().format(record)
        return self.COLORS.get(record.levelname, "") + log_message + Fore.RESET


DEFAULT_LOGGING_CONFIG: Dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {
        "simple": {
            "format": "[%(asctime)s] [%(levelname)s] [%(name)s.%(funcName)s:%(lineno)d] [PID:%(process)d] %(message)s",
        },
        "colorful": {
            "()": ColorfulFormatter,
            "format": "[%(asctime)s] [%(levelname)s] [%(name)s.%(funcName)s:%(lineno)d] [PID:%(process)d] [RANK:%(rank)d] %(message)s",
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
    # log level will be superseded by the AxolotlLogger
    "root": {"handlers": ["console"], "level": os.getenv("LOG_LEVEL", "WARNING")},
    "loggers": {
        "axolotl": {
            "handlers": ["color_console"],
            "level": os.getenv("LOG_LEVEL", "INFO"),
            "propagate": False,
        },
    },
}


def configure_logging():
    """Configure with default logging"""
    init()  # Initialize colorama
    dictConfig(DEFAULT_LOGGING_CONFIG)
    logging.setLoggerClass(AxolotlLogger)
