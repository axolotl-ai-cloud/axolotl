"""
module for base dataset transform strategies
"""

import importlib

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def load(strategy, cfg, module_base=None, **kwargs):
    try:
        if len(strategy.split(".")) == 1:
            strategy = strategy + ".default"
        load_fn = strategy.split(".")[-1]
        if len(strategy.split(".")) > 1:
            try:
                importlib.import_module(
                    strategy.split(".")[-2],
                    ".".join(strategy.split(".")[:-2]),
                )
                module_base = ".".join(strategy.split(".")[:-2])
                strategy = strategy.split(".")[-2]
            except ModuleNotFoundError:
                strategy = "." + ".".join(strategy.split(".")[:-1])
        else:
            strategy = "." + ".".join(strategy.split(".")[:-1])
        mod = importlib.import_module(strategy, module_base)
        func = getattr(mod, load_fn)
        return func(cfg, **kwargs)
    except Exception:
        LOG.warning(f"unable to load strategy {strategy}")
        return None
