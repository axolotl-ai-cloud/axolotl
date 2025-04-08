"""
module for base dataset transform strategies
"""

import importlib
import logging
import sys

LOG = logging.getLogger("axolotl")


def import_from_path(module_name: str, file_path: str):
    """
    Import a module from a file path.
    Args:
        module_name (str): Name of the module.
        file_path (str): Path to the file.
    Feturns:
        module: The imported module.
    """
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
    if len(strategy.split(".")) == 1:
        strategy = strategy + ".default"
    load_fn = strategy.split(".")[-1]
    func = None
    if len(strategy.split(".")) > 1:
        try:
            mod = importlib.import_module(
                strategy.split(".")[-2],
                ".".join(strategy.split(".")[:-2]),
            )
            func = getattr(mod, load_fn)
            return func(cfg, **kwargs)
        except ModuleNotFoundError:
            pass

        try:
            mod = importlib.import_module(
                "." + ".".join(strategy.split(".")[:-1]), module_base
            )
            func = getattr(mod, load_fn)
            return func(cfg, **kwargs)
        except ModuleNotFoundError:
            pass

        try:
            file_path = "/".join(strategy.split(".")[:-1]) + ".py"
            module_name = strategy.split(".")[-2]
            mod = import_from_path(module_name, file_path)
            func = getattr(mod, load_fn)
            if func is not None:
                return func(cfg, **kwargs)
        except FileNotFoundError:
            pass
    else:
        strategy = "." + ".".join(strategy.split(".")[:-1])
        mod = importlib.import_module(strategy, module_base)
        func = getattr(mod, load_fn)
        return func(cfg, **kwargs)

    LOG.warning(f"unable to load strategy {strategy}")
    return func
