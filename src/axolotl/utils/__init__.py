"""
Basic utils for Axolotl
"""

import importlib.util
import os
import re

import torch


def is_mlflow_available():
    return importlib.util.find_spec("mlflow") is not None


def is_comet_available():
    return importlib.util.find_spec("comet_ml") is not None


def is_opentelemetry_available():
    return (
        importlib.util.find_spec("opentelemetry") is not None
        and importlib.util.find_spec("prometheus_client") is not None
    )


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
    # Set both PYTORCH_ALLOC_CONF and PYTORCH_CUDA_ALLOC_CONF for compatibility
    if (
        os.getenv("PYTORCH_ALLOC_CONF") is None
        and os.getenv("PYTORCH_CUDA_ALLOC_CONF") is None
    ):
        config_value = "expandable_segments:True,roundup_power2_divisions:16"
        os.environ["PYTORCH_ALLOC_CONF"] = config_value
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = config_value


def get_not_null(value, default=None):
    """
    return the value if it's not None, otherwise return the default value
    """
    return value if value is not None else default
