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


# pylint: disable=duplicate-code
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
    """Set up CUDA allocation config if using PyTorch >= 2.2"""
    torch_version = torch.__version__.split(".")
    torch_major, torch_minor = int(torch_version[0]), int(torch_version[1])
    if torch_major == 2 and torch_minor >= 2:
        if os.getenv("PYTORCH_CUDA_ALLOC_CONF") is None:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
                "expandable_segments:True,roundup_power2_divisions:16"
            )
