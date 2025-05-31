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

DEFAULT_AXOLOTL_LOG_LEVEL = "INFO"
DEFAULT_LOG_LEVEL = "WARNING"


class AxolotlOrWarnErrorFilter(logging.Filter):
    """
    Allows ANY WARNING or higher (unless overridden by LOG_LEVEL)
    Allows axolotl.* at INFO or higher (unless overridden by AXOLOTL_LOG_LEVEL)
    Drops all other records (i.e. non-axolotl.INFO, DEBUG, etc. by default)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.axolotl_level = logging.getLevelNamesMapping()[
            os.getenv("AXOLOTL_LOG_LEVEL", DEFAULT_AXOLOTL_LOG_LEVEL)
        ]
        self.other_level = logging.getLevelNamesMapping()[
            os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL)
        ]

    def filter(self, record: LogRecord) -> bool:
        # General filter
        if record.levelno >= self.other_level:
            return True

        # Axolotl filter
        return (
            record.name.startswith("axolotl") and record.levelno >= self.axolotl_level
        )


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
    "disable_existing_loggers": False,
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
    "root": {
        "handlers": ["console"],
        "level": os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL),
    },
    "loggers": {
        "axolotl": {
            "handlers": ["color_console"],
            "level": os.getenv("AXOLOTL_LOG_LEVEL", DEFAULT_AXOLOTL_LOG_LEVEL),
            "propagate": False,
        },
    },
}


def configure_logging():
    """Configure with default logging"""
    init()  # Initialize colorama
    dictConfig(DEFAULT_LOGGING_CONFIG)
    logging.setLoggerClass(AxolotlLogger)

    # set default `ACCELERATE_LOG_LEVEL` to `LOG_LEVEL` if available and not set
    if "ACCELERATE_LOG_LEVEL" not in os.environ:
        os.environ["ACCELERATE_LOG_LEVEL"] = os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL)
