"""
helper utils for tests
"""
import os
import shutil
import tempfile
import unittest
from functools import wraps
from importlib.metadata import version
from pathlib import Path


def with_temp_dir(test_func):
    @wraps(test_func)
    def wrapper(*args, **kwargs):
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        try:
            # Pass the temporary directory to the test function
            test_func(*args, temp_dir=temp_dir, **kwargs)
        finally:
            # Clean up the directory after the test
            shutil.rmtree(temp_dir)

    return wrapper


def most_recent_subdir(path):
    base_path = Path(path)
    subdirectories = [d for d in base_path.iterdir() if d.is_dir()]
    if not subdirectories:
        return None
    subdir = max(subdirectories, key=os.path.getctime)

    return subdir


def require_torch_2_1_1(test_case):
    """
    Decorator marking a test that requires torch >= 2.1.1
    """

    def is_min_2_1_1():
        torch_version = version("torch")
        return torch_version >= "2.1.1"

    return unittest.skipUnless(is_min_2_1_1(), "test torch 2.1.1")(test_case)
