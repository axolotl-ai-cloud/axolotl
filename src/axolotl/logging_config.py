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

from axolotl.utils.tee import get_file_only_stream

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
    """A Logger class placeholder (no global filters added).

    Filtering is applied at handler level to allow separate console/file behavior.
    """

    def __init__(self, name: str, level: int = logging.NOTSET):
        super().__init__(name, level)


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
        # Concise (no callsite, no PID) variants
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
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            # formatter set in configure_logging() based on AXOLOTL_LOG_FORMAT
            "formatter": "concise",
            "filters": ["ax_or_warn"],
            "stream": "ext://sys.stdout",
        },
        "color_console": {
            "class": "logging.StreamHandler",
            # formatter set in configure_logging() based on AXOLOTL_LOG_FORMAT
            "formatter": "concise_color",
            "filters": ["ax_or_warn"],
            "stream": "ext://sys.stdout",
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

    # set default `ACCELERATE_LOG_LEVEL` to `LOG_LEVEL` if available and not set
    if "ACCELERATE_LOG_LEVEL" not in os.environ:
        os.environ["ACCELERATE_LOG_LEVEL"] = os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL)

    # Add a DEBUG-level file-only handler that writes all records below the console
    # threshold into the tee file, to avoid duplication while capturing full DEBUG logs.
    class BelowLevelFilter(logging.Filter):
        def __init__(self, threshold: int):
            super().__init__()
            self.threshold = threshold

        def filter(self, record: LogRecord) -> bool:
            return record.levelno < self.threshold

    # Determine thresholds
    level_names = (
        logging.getLevelNamesMapping()
        if hasattr(logging, "getLevelNamesMapping")
        else None
    )

    def to_level(name: str, default: int) -> int:
        if level_names:
            return level_names.get(name.upper(), default)
        return (
            logging.getLevelName(name.upper())
            if isinstance(logging.getLevelName(name.upper()), int)
            else default
        )

    root_console_level = to_level(
        os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL), logging.WARNING
    )
    ax_console_level = to_level(
        os.getenv("AXOLOTL_LOG_LEVEL", DEFAULT_AXOLOTL_LOG_LEVEL), logging.INFO
    )

    # Create a file-only stream handler for axolotl logs below console level
    ax_file_only_handler = logging.StreamHandler(get_file_only_stream())
    ax_file_only_handler.setLevel(logging.DEBUG)
    ax_file_only_handler.addFilter(BelowLevelFilter(ax_console_level))
    ax_file_only_handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")
    )
    logging.getLogger("axolotl").addHandler(ax_file_only_handler)

    # And another for non-axolotl logs below root console level
    root_file_only_handler = logging.StreamHandler(get_file_only_stream())
    root_file_only_handler.setLevel(logging.DEBUG)
    root_file_only_handler.addFilter(BelowLevelFilter(root_console_level))
    root_file_only_handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")
    )
    logging.getLogger().addHandler(root_file_only_handler)
