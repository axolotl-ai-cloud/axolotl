"""
Basic utils for Axolotl
"""
import importlib


def is_mlflow_available():
    return importlib.util.find_spec("mlflow") is not None
