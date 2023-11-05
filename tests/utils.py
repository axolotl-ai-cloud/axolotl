"""
helper utils for tests
"""

import shutil
import tempfile
from functools import wraps


def with_temp_dir(test_func):
    @wraps(test_func)
    def wrapper(*args, **kwargs):
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        try:
            # Pass the temporary directory to the test function
            test_func(temp_dir, *args, **kwargs)
        finally:
            # Clean up the directory after the test
            shutil.rmtree(temp_dir)

    return wrapper
