"""Flex attention monkey patch"""

import torch
import transformers


def is_transformers_version_less_than_or_equal_4_50():
    version_parts = transformers.__version__.split(".")

    # Check if the version is less than or equal to 4.50
    if int(version_parts[0]) < 4:
        return True
    if int(version_parts[0]) == 4:
        if int(version_parts[1]) < 50:
            return True
        if int(version_parts[1]) == 50 and int(version_parts[2]) <= 99:
            return True
    return False


def patch_flex():
    is_torch_2_6 = torch.__version__.startswith("2.6")
    is_transformers_4_50_or_below = is_transformers_version_less_than_or_equal_4_50()

    if is_torch_2_6 and is_transformers_4_50_or_below:
        from torch.nn.attention.flex_attention import flex_attention

        class WrappedFlexAttention:
            """
            We are doing a singleton class so that flex attention is compiled once when it's first called.
            """

            _instance = None
            _is_flex_compiled = False
            _compiled_flex_attention = None

            def __new__(cls, *args, **kwargs):
                if cls._instance is None:
                    # Create a new instance if one doesn't already exist
                    cls._instance = super().__new__(cls)
                return cls._instance

            @torch.compiler.disable(recursive=False)
            def __init__(self):
                """
                Initialize or update the singleton instance.
                """
                if not self._is_flex_compiled:
                    self._compiled_flex_attention = torch.compile(
                        flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs"
                    )
                    self._is_flex_compiled = True

            def __call__(self):
                return self._compiled_flex_attention

        transformers.integrations.flex_attention.WrappedFlexAttention = (
            WrappedFlexAttention
        )
