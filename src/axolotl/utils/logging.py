"""Logging configuration settings"""

import sys
from logging.config import dictConfig


def configure_logging(log_level: str):
    """Configure with default logging"""
    dictConfig(
        {
            "version": 1,
            "formatters": {
                "simple": {
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
            },
            "root": {"handlers": ["console"], "level": log_level},
            "loggers": {
                "axolotl": {
                    "handlers": ["console"],
                    "level": log_level,
                    "propagate": False,
                },
            },
        }
    )
