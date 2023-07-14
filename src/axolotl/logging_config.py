import sys
from logging.config import dictConfig
from typing import Any, Dict

DEFAULT_LOGGING_CONFIG: Dict[str, Any] = {
    "version": 1,
    "formatters": {
        "simple": {
            "format": "[%(asctime)s] [%(levelname)s] [PID:%(process)d] [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
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
    "root": {"handlers": ["console"], "level": "INFO"},
}


def configure_logging():
    """Configure with default logging"""
    dictConfig(DEFAULT_LOGGING_CONFIG)
