"""
test utils for helpers and decorators
"""

import os
from functools import wraps


def with_hf_offline(test_func):
    """
    test decorator that sets HF_HUB_OFFLINE environment variable to True and restores it after the test even if the test fails.
    :param test_func:
    :return:
    """

    def reload_modules():
        # Force reload of the modules that check this variable
        import importlib

        import datasets
        import huggingface_hub.constants
        import transformers

        # Reload the constants module first, as others depend on it
        importlib.reload(huggingface_hub.constants)
        importlib.reload(transformers)
        importlib.reload(datasets)

    @wraps(test_func)
    def wrapper(*args, **kwargs):
        # Save the original value of HF_HUB_OFFLINE environment variable
        original_hf_offline = os.getenv("HF_HUB_OFFLINE")

        # Set HF_OFFLINE environment variable to True
        os.environ["HF_HUB_OFFLINE"] = "1"

        reload_modules()
        try:
            # Run the test function
            return test_func(*args, **kwargs)
        finally:
            # Restore the original value of HF_HUB_OFFLINE environment variable
            if original_hf_offline is not None:
                os.environ["HF_HUB_OFFLINE"] = original_hf_offline
            else:
                del os.environ["HF_HUB_OFFLINE"]
            reload_modules()

    return wrapper
