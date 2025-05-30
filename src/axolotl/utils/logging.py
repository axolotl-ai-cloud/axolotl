"""
logging helpers to only log on main process
"""

import functools
import logging
import os

from axolotl.utils.distributed import is_main_process

# Adapted from Accelerate
# https://github.com/huggingface/accelerate/blob/main/src/accelerate/logging.py


class MultiProcessAdapter(logging.LoggerAdapter):
    """
    logger adapter for distributed logging, specifically to only log on main process
    """

    def __init__(self, logger, use_environ=False, extra=None):
        super().__init__(logger, extra)
        self.use_environ = use_environ

    @staticmethod
    def _should_log(main_process_only, use_environ=False):
        return not main_process_only or (
            main_process_only and is_main_process(use_environ=use_environ)
        )

    def log(self, level, msg, *args, **kwargs):
        use_environ = kwargs.pop("use_environ", self.use_environ)
        main_process_only = kwargs.pop("main_process_only", True)
        kwargs.setdefault("stacklevel", 2)

        if self.isEnabledFor(level) and self._should_log(
            main_process_only, use_environ=use_environ
        ):
            msg, kwargs = self.process(msg, kwargs)
            self.logger.log(level, msg, *args, **kwargs)

    @functools.lru_cache(maxsize=10)
    def warning_once(self, *args, **kwargs):
        """
        This method is identical to `logger.warning()`, but will emit the warning with the same message only once

        Note: The cache is for the function arguments, so 2 different callers using the same arguments will hit the
        cache. The assumption here is that all warning messages are unique across the code. If they aren't then need to
        switch to another type of cache that includes the caller frame information in the hashing function.
        """
        self.warning(*args, **kwargs)


def get_logger(
    name: str, log_level: str | None = None, use_environ: bool = False
) -> MultiProcessAdapter:
    if log_level is None:
        log_level = os.environ.get("AXOLOTL_LOG_LEVEL", None)
    logger = logging.getLogger(name)
    if log_level is not None:
        logger.setLevel(log_level.upper())
        logger.root.setLevel(log_level.upper())
    return MultiProcessAdapter(logger, use_environ=use_environ, extra={})
