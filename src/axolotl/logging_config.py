import os
import sys
from logging.config import dictConfig
from typing import Any, Dict
from logging import Formatter
from colorama import Fore, init

class ColorfulFormatter(Formatter):
    COLORS = {
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Fore.BOLD,
    }

    def format(self, record):
        log_message = super().format(record)
        return self.COLORS.get(record.levelname, '') + log_message + Fore.RESET

DEFAULT_LOGGING_CONFIG: Dict[str, Any] = {
    "version": 1,
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
    "root": {"handlers": ["console"], "level": os.getenv("LOG_LEVEL", "INFO")},
    "loggers": {
        "axolotl": {"handlers": ["color_console"], "level": "DEBUG", "propagate": False},
    },
}

def configure_logging():
    """Configure with default logging"""
    init() # Initialize colorama
    dictConfig(DEFAULT_LOGGING_CONFIG)
