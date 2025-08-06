"""
Helper for importing modules from strings
"""

import importlib


def get_cls_from_module_str(module_str: str):
    # use importlib to dynamically load the reward function from the module
    module_name = module_str.split(".")[-1]
    mod = importlib.import_module(".".join(module_str.split(".")[:-1]))
    mod_cls = getattr(mod, module_name)
    return mod_cls
