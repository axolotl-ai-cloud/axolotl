"""Module to load prompt strategies."""

import importlib
import inspect
import logging

from axolotl.prompt_strategies.user_defined import UserDefinedDatasetConfig

LOG = logging.getLogger("axolotl.prompt_strategies")


def load(strategy, tokenizer, cfg, ds_cfg, processor=None):
    try:
        if strategy == "messages":
            from .messages import load as messages_load

            return messages_load(tokenizer, cfg, ds_cfg, processor=processor)
        load_fn = "load"
        if strategy.split(".")[-1].startswith("load_"):
            load_fn = strategy.split(".")[-1]
            strategy = ".".join(strategy.split(".")[:-1])
        mod = importlib.import_module(f".{strategy}", "axolotl.prompt_strategies")
        func = getattr(mod, load_fn)
        load_kwargs = {}
        if strategy == "user_defined":
            load_kwargs["ds_cfg"] = UserDefinedDatasetConfig(**ds_cfg)
        else:
            sig = inspect.signature(func)
            if "ds_cfg" in sig.parameters:
                load_kwargs["ds_cfg"] = ds_cfg
            if "processor" in sig.parameters:
                load_kwargs["processor"] = processor
        return func(tokenizer, cfg, **load_kwargs)
    except ModuleNotFoundError:
        return None
    except Exception as exc:  # pylint: disable=broad-exception-caught
        LOG.error(f"Failed to load prompt strategy `{strategy}`: {str(exc)}")
        raise exc
    return None
