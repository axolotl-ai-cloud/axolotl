"""
module for base dataset transform strategies
"""

import importlib
import logging
import sys

LOG = logging.getLogger("axolotl")


def import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Could not create module spec for: {file_path}")
    module = importlib.util.module_from_spec(spec)

    sys.modules[module_name] = module
    loader = importlib.machinery.SourceFileLoader(module_name, file_path)
    spec.loader = loader
    loader.exec_module(module)
    return module


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
    except Exception:  # pylint: disable=broad-exception-caught
        LOG.warning(f"unable to load strategy {strategy}")
        return None
