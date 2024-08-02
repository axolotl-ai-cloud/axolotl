"""
module for base dataset transform strategies
"""

import importlib
import logging

LOG = logging.getLogger("axolotl")


def load(strategy, cfg, module_base=None, **kwargs):
    try:
        load_fn = strategy.split(".")[-1]
        strategy = ".".join(strategy.split(".")[:-1])
        mod = importlib.import_module(f".{strategy}", module_base)
        func = getattr(mod, load_fn)
        return func(cfg, **kwargs)
    except Exception:  # pylint: disable=broad-exception-caught
        LOG.warning(f"unable to load strategy {strategy}")
        return None
