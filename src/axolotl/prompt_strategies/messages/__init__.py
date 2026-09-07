"""Module to load message prompt strategies."""

import importlib
import inspect

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def load(tokenizer, cfg, ds_cfg, processor=None):
    try:
        strategy = ds_cfg.get("input_transform", "chat")

        load_fn = "load"
        if strategy.split(".")[-1].startswith("load_"):
            load_fn = strategy.split(".")[-1]
            strategy = ".".join(strategy.split(".")[:-1])
        mod = importlib.import_module(
            f".{strategy}", "axolotl.prompt_strategies.messages"
        )
        func = getattr(mod, load_fn)
        load_kwargs = {}
        sig = inspect.signature(func)
        if "ds_cfg" in sig.parameters:
            load_kwargs["ds_cfg"] = ds_cfg
        if "processor" in sig.parameters:
            load_kwargs["processor"] = processor
        return func(tokenizer, cfg, **load_kwargs)
    except ModuleNotFoundError:
        return None
    except Exception as exc:
        LOG.error(f"Failed to load prompt strategy `{strategy}`: {str(exc)}")
        raise exc
