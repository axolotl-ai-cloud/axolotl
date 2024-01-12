"""
module for DPO style dataset transform strategies
"""

import importlib


def load(strategy, cfg):
    try:
        load_fn = strategy.split(".")[-1]
        strategy = ".".join(strategy.split(".")[:-1])
        mod = importlib.import_module(f".{strategy}", "axolotl.prompt_strategies.dpo")
        func = getattr(mod, load_fn)
        load_kwargs = {}
        return func(cfg, **load_kwargs)
    except Exception:  # pylint: disable=broad-exception-caught
        return None
