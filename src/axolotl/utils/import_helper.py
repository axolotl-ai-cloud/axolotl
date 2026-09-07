"""
Helper for importing modules from strings
"""

import importlib


def get_cls_from_module_str(module_str: str):
    # use importlib to dynamically load the reward function from the module
    if not isinstance(module_str, str) or not module_str.strip():
        raise ValueError("module_str must be a non-empty string")

    parts = module_str.split(".")
    if len(parts) < 2:
        raise ValueError(f"Invalid module string format: {module_str}")

    try:
        cls_name = parts[-1]
        module_path = ".".join(parts[:-1])
        mod = importlib.import_module(module_path)
        mod_cls = getattr(mod, cls_name)
        return mod_cls
    except ImportError as e:
        raise ImportError(f"Failed to import module '{module_path}': {e}") from e
    except AttributeError as e:
        raise AttributeError(
            f"Class '{cls_name}' not found in module '{module_path}': {e}"
        ) from e
