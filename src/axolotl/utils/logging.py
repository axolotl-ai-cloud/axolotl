import logging

from axolotl.utils.distributed import is_main_process


def log_info_rank_zero(log: logging.Logger, message: str):
    if is_main_process():
        log.info(message)


def log_debug_rank_zero(log: logging.Logger, message: str):
    if is_main_process():
        log.debug(message)


def log_warning_rank_zero(log: logging.Logger, message: str):
    if is_main_process():
        log.warning(message)
