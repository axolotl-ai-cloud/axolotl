"""
Basic utils for Axolotl
"""

import importlib.util
import os
import re

import torch


def make_lazy_getattr(
    lazy_imports: dict[str, str], module_name: str, module_globals: dict
):
    """Create a module-level ``__getattr__`` that lazily imports symbols.

    Args:
        lazy_imports: Mapping of attribute name to relative module path,
            e.g. ``{"AxolotlDPOTrainer": ".dpo.trainer"}``.
        module_name: The ``__name__`` of the calling module (used as the
            anchor for relative imports).
        module_globals: The ``globals()`` dict of the calling module,
            used to cache resolved attributes so ``__getattr__`` is only
            invoked once per name.

    Returns:
        A ``__getattr__`` function suitable for assignment at module scope.
    """
    import importlib

    def __getattr__(name: str):
        if name in lazy_imports:
            module = importlib.import_module(lazy_imports[name], module_name)
            attr = getattr(module, name)
            module_globals[name] = attr
            return attr
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")

    return __getattr__


def is_mlflow_available():
    return importlib.util.find_spec("mlflow") is not None


def is_comet_available():
    return importlib.util.find_spec("comet_ml") is not None


def is_opentelemetry_available():
    return (
        importlib.util.find_spec("opentelemetry") is not None
        and importlib.util.find_spec("prometheus_client") is not None
    )


def is_trackio_available():
    return importlib.util.find_spec("trackio") is not None


def get_pytorch_version() -> tuple[int, int, int]:
    """
    Get Pytorch version as a tuple of (major, minor, patch).
    """
    torch_version = torch.__version__
    version_match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", torch_version)

    if not version_match:
        raise ValueError("Invalid version format")

    major, minor, patch = version_match.groups()
    major, minor = int(major), int(minor)
    patch = int(patch) if patch is not None else 0  # Default patch to 0 if not present
    return major, minor, patch


def set_pytorch_cuda_alloc_conf():
    """Set up CUDA allocation config"""
    torch_version = torch.__version__.split(".")
    torch_major, torch_minor = int(torch_version[0]), int(torch_version[1])
    config_value = "expandable_segments:True"
    config_older_suffix = ",roundup_power2_divisions:16"
    if (
        torch_major == 2
        and torch_minor >= 9
        and os.getenv("PYTORCH_ALLOC_CONF") is None
    ):
        os.environ["PYTORCH_ALLOC_CONF"] = config_value
    elif (
        torch_major == 2
        and torch_minor >= 2
        and os.getenv("PYTORCH_CUDA_ALLOC_CONF") is None
    ):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = config_value + config_older_suffix


def set_misc_env():
    if os.getenv("XFORMERS_IGNORE_FLASH_VERSION_CHECK") is None:
        os.environ["XFORMERS_IGNORE_FLASH_VERSION_CHECK"] = "1"


def get_not_null(value, default=None):
    """
    return the value if it's not None, otherwise return the default value
    """
    return value if value is not None else default
