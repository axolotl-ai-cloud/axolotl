"""Module to load prompt strategies."""

import importlib


def load(strategy, tokenizer, cfg, ds_cfg):
    try:
        load_fn = "load"
        if strategy.split(".")[-1].startswith("load_"):
            load_fn = strategy.split(".")[-1]
            strategy = ".".join(strategy.split(".")[:-1])
        mod = importlib.import_module(f".{strategy}", "axolotl.prompt_strategies")
        func = getattr(mod, load_fn)
        load_kwargs = {}
        if strategy == "user_defined":
            load_kwargs["ds_cfg"] = ds_cfg
        return func(tokenizer, cfg, **load_kwargs)
    except Exception:  # pylint: disable=broad-exception-caught
        return None
