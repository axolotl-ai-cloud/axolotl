"""
Basic utils for Axolotl
"""
import importlib.util

import torch
from packaging import version


def is_mlflow_available():
    return importlib.util.find_spec("mlflow") is not None


def is_comet_available():
    return importlib.util.find_spec("comet_ml") is not None


def is_torch_min(version_string: str):
    torch_version = version.parse(torch.__version__)
    return torch_version >= version.parse(version_string)
