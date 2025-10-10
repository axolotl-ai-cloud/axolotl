"""Common logging module for axolotl."""

import logging
import os
from logging import Formatter, Logger, LogRecord
from logging.config import dictConfig
from typing import Any, Dict

from colorama import Fore, Style, init

DEFAULT_AXOLOTL_LOG_LEVEL = "INFO"
DEFAULT_LOG_LEVEL = "WARNING"


class AxolotlOrWarnErrorFilter(logging.Filter):
    """
    Allows ANY WARNING or higher (unless overridden by LOG_LEVEL). Allows axolotl.* at
    INFO or higher (unless overridden by AXOLOTL_LOG_LEVEL). Drops all other records
    (i.e. non-axolotl.INFO, DEBUG, etc. by default).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        axolotl_log_level = os.getenv(
            "AXOLOTL_LOG_LEVEL", DEFAULT_AXOLOTL_LOG_LEVEL
        ).upper()
        other_log_level = os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()

        try:
            # py311+ only
            level_mapping = logging.getLevelNamesMapping()
            self.axolotl_level = level_mapping[axolotl_log_level]
            self.other_level = level_mapping[other_log_level]
        except AttributeError:
            # For py310, use getLevelName directly
            self.axolotl_level = logging.getLevelName(axolotl_log_level)
            self.other_level = logging.getLevelName(other_log_level)

    def filter(self, record: LogRecord) -> bool:
        # General filter
        if record.levelno >= self.other_level:
            return True

        # Axolotl filter
        return (
            record.name.startswith("axolotl") and record.levelno >= self.axolotl_level
        )


class AxolotlLogger(Logger):
    """Logger that applies filtering to non-axolotl loggers."""

    def __init__(self, name: str, level: int = logging.NOTSET):
        super().__init__(name, level)
        if not name.startswith("axolotl"):
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
        record.rank_fmt = f" [RANK:{record.rank}]" if record.rank != 0 else ""
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
            "format": "[%(asctime)s] [%(levelname)s] [%(name)s.%(funcName)s:%(lineno)d] [PID:%(process)d]%(rank_fmt)s %(message)s",
        },
        "concise": {
            "format": "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        },
        "concise_color": {
            "()": ColorfulFormatter,
            "format": "[%(asctime)s] [%(levelname)s] [%(name)s]%(rank_fmt)s %(message)s",
        },
    },
    "filters": {
        "ax_or_warn": {
            "()": "axolotl.logging_config.AxolotlOrWarnErrorFilter",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "concise",
            "filters": ["ax_or_warn"],
            "stream": "ext://sys.stdout",
        },
        "color_console": {
            "class": "logging.StreamHandler",
            "formatter": "concise_color",
            "filters": ["ax_or_warn"],
            "stream": "ext://sys.stdout",
        },
        "ax_file_only": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "simple",
            "stream": "ext://axolotl.utils.tee.file_only_stream",
        },
        "root_file_only": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "simple",
            "stream": "ext://axolotl.utils.tee.file_only_stream",
        },
    },
    "root": {
        "handlers": ["console", "root_file_only"],
        "level": os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper(),
    },
    "loggers": {
        "axolotl": {
            "handlers": ["color_console", "ax_file_only"],
            "level": os.getenv("AXOLOTL_LOG_LEVEL", DEFAULT_AXOLOTL_LOG_LEVEL).upper(),
            "propagate": False,
        },
    },
}


def configure_logging():
    """Configure with default logging"""
    init()  # Initialize colorama

    dictConfig(DEFAULT_LOGGING_CONFIG)
    logging.setLoggerClass(AxolotlLogger)

    # Route Python warnings through logging so they reach file handlers
    logging.captureWarnings(True)

    # Set default `ACCELERATE_LOG_LEVEL` to `LOG_LEVEL` if available and not set
    if "ACCELERATE_LOG_LEVEL" not in os.environ:
        os.environ["ACCELERATE_LOG_LEVEL"] = os.getenv(
            "LOG_LEVEL", DEFAULT_LOG_LEVEL
        ).upper()
