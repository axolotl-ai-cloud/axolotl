"""Patch transformers.dynamic_module_utils.get_class_in_module to avoid reloading models from disk"""

import importlib
import os
import sys
import typing
from pathlib import Path

from transformers.file_utils import HF_MODULES_CACHE


def _patched_get_class_in_module(
    class_name: str, module_path: typing.Union[str, os.PathLike]
) -> typing.Type:
    """
    Import a module on the cache directory for modules and extract a class from it.

    Args:
        class_name (`str`): The name of the class to import.
        module_path (`str` or `os.PathLike`): The path to the module to import.

    Returns:
        `typing.Type`: The class looked for.
    """
    name = os.path.normpath(module_path)
    if name.endswith(".py"):
        name = name[:-3]
    name = name.replace(os.path.sep, ".")
    module_spec = importlib.util.spec_from_file_location(
        name, location=Path(HF_MODULES_CACHE) / module_path
    )
    module = sys.modules.get(name)
    if module is None:
        module = importlib.util.module_from_spec(module_spec)
        # insert it into sys.modules before any loading begins
        sys.modules[name] = module
        # load in initial case only
        module_spec.loader.exec_module(module)
    return getattr(module, class_name)


def patch_transformers_dynamic_module_utils():
    """
    Recently, transformers started reloading modeling code from disk for models marked trust_remote_code=True.
    This causes monkey-patches for multipack and liger to be removed.
    We replace the original function with a version that does not reload the module from disk.
    See https://github.com/huggingface/transformers/pull/30370#pullrequestreview-2264361581
    """
    import transformers

    transformers.dynamic_module_utils.get_class_in_module = _patched_get_class_in_module
