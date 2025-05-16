"""
logging helpers to only log on main process
"""

import os
from axolotl.utils.distributed import is_main_process
import logging

# Adapted from Accelerate
# https://github.com/huggingface/accelerate/blob/main/src/accelerate/logging.py
class MultiProcessAdapter(logging.LoggerAdapter):
    """
    logger adapter for distributed logging, specifically to only log on main process
    """

    @staticmethod
    def _should_log(main_process_only):
        return not main_process_only or (
            main_process_only and is_main_process(use_environ=True)
        )

    def log(self, level, msg, *args, **kwargs):
        main_process_only = kwargs.pop("main_process_only", True)
        kwargs.setdefault("stacklevel", 2)

        if self.isEnabledFor(level):
            if self._should_log(main_process_only):
                msg, kwargs = self.process(msg, kwargs)
                self.logger.log(level, msg, *args, **kwargs)


def get_logger(name: str, log_level: str | None = None):
    if log_level is None:
        log_level = os.environ.get("AXOLOTL_LOG_LEVEL", None)
    logger = logging.getLogger(name)
    if log_level is not None:
        logger.setLevel(log_level.upper())
        logger.root.setLevel(log_level.upper())
    return MultiProcessAdapter(logger, {})
