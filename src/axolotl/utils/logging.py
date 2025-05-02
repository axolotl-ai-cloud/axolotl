"""
logging helpers to only log on main process
"""

import logging
from functools import partial

from axolotl.utils.distributed import is_main_process


def log_rank_zero(log: logging.Logger, message: str, level: str = "info"):
    if is_main_process(use_environ=True):
        getattr(log, level.lower())(message)


log_info_rank_zero = partial(log_rank_zero, level="info")
log_debug_rank_zero = partial(log_rank_zero, level="debug")
log_warning_rank_zero = partial(log_rank_zero, level="warning")
log_error_rank_zero = partial(log_rank_zero, level="error")
