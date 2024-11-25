"""
Basic utils for Axolotl
"""

import importlib.util
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


# pylint: enable=duplicate-code
