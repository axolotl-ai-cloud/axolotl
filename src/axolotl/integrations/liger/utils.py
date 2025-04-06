"""
utils to patch liger kernel ops to disable torch.compile
"""

from functools import wraps

import torch


def patch_with_compile_disable(module, function_name):
    """
    Patch a function in a module by wrapping it with torch.compile.disable

    Args:
        module: The module containing the function to patch
        function_name: The name of the function to patch
    """
    original_function = getattr(module, function_name)

    @wraps(original_function)
    @torch.compiler.disable
    def wrapped_function(*args, **kwargs):
        return original_function(*args, **kwargs)

    # Replace the original function with the wrapped one
    setattr(module, function_name, wrapped_function)

    # Return the original function in case you need to restore it later
    return original_function
