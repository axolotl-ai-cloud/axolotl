"""
helper utils for tests
"""
import os
import shutil
import tempfile
from functools import wraps
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
