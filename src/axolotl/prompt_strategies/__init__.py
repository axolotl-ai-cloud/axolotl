"""Module to load prompt strategies."""

import importlib


def load(strategy, tokenizer, cfg):
    try:
        load_fn = "load"
        if strategy.split(".")[-1].startswith("load_"):
            load_fn = strategy.split(".")[-1]
            strategy = ".".join(strategy.split(".")[:-1])
        mod = importlib.import_module(f".{strategy}", "axolotl.prompt_strategies")
        func = getattr(mod, load_fn)
        return func(tokenizer, cfg)
    except Exception:  # pylint: disable=broad-exception-caught
        return None
