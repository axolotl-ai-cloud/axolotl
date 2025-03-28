"""
test utils for helpers and decorators
"""

import os
from functools import wraps


def with_hf_offline(test_func):
    """
    test decorator that sets HF_OFFLINE environment variable to True and restores it after the test even if the test fails.
    :param test_func:
    :return:
    """

    @wraps(test_func)
    def wrapper(*args, **kwargs):
        # Save the original value of HF_OFFLINE environment variable
        original_hf_offline = os.getenv("HF_OFFLINE")

        # Set HF_OFFLINE environment variable to True
        os.environ["HF_OFFLINE"] = "1"

        try:
            # Run the test function
            test_func(*args, **kwargs)
        finally:
            # Restore the original value of HF_OFFLINE environment variable
            if original_hf_offline is not None:
                os.environ["HF_OFFLINE"] = original_hf_offline
            else:
                del os.environ["HF_OFFLINE"]

    return wrapper
