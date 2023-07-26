"""Logging configuration settings"""

import logging
import sys
from logging.config import dictConfig


def configure_logging(log_level: str = "DEBUG"):
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
        }
    )


def print_loggers():
    """Function to print the current logging hierarchy"""
    loggers_dict = logging.Logger.manager.loggerDict
    for logger_name, logger in loggers_dict.items():
        if isinstance(logger, logging.Logger):
            print(
                f"Logger name: {logger_name}, Logger level: {logger.level}, Propagate: {logger.propagate}"
            )
