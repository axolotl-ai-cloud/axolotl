"""Flex attention monkey patch"""

import sys

import torch
import transformers
from packaging import version
from transformers.utils.import_utils import _torch_version, is_torch_less_or_equal

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def patch_flex_wrapper(**flex_attn_compile_kwargs):
    # TODO remove this patch when transformers#37285 is merged and in a release
    is_torch_2_6 = torch.__version__.startswith("2.6")

    if not is_torch_2_6:
        return

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

        @classmethod
        def del_singleton(cls):
            cls._instance = None

        @torch.compiler.disable(recursive=False)
        def __init__(self, training):
            """
            Initialize or update the singleton instance.
            """
            self.training = None
            if not self._is_flex_compiled or training != self.training:
                self.training = training
                if is_torch_less_or_equal("2.5.1"):
                    self._compiled_flex_attention = torch.compile(
                        flex_attention, dynamic=False
                    )
                # In PyTorch 2.6.0, there's a known issue with flex attention compilation which may
                # cause errors. The suggested fix is to compile with "max-autotune-no-cudagraphs"
                # see https://github.com/pytorch/pytorch/issues/146260 for training
                elif version.parse(_torch_version).base_version == "2.6.0" and training:
                    self._compiled_flex_attention = torch.compile(
                        flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs"
                    )
                # Fallback, usually the most recent torch 2.7.x+ versions
                else:
                    LOG.info(
                        "Compiling flex attention with kwargs: %s. This may take a while...",
                        flex_attn_compile_kwargs,
                        main_process_only=True,
                    )
                    self._compiled_flex_attention = torch.compile(
                        flex_attention,
                        **flex_attn_compile_kwargs,
                    )
                    LOG.info(
                        "Flex attention compiled successfully.", main_process_only=True
                    )

                self._is_flex_compiled = True

        def __call__(self):
            return self._compiled_flex_attention

    transformers.integrations.flex_attention.WrappedFlexAttention = WrappedFlexAttention
    sys.modules[
        "transformers.integrations.flex_attention"
    ].WrappedFlexAttention = WrappedFlexAttention
