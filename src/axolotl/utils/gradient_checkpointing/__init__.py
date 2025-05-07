"""custom checkpointing utils"""

import importlib
from functools import partial

from packaging import version

from axolotl.utils.gradient_checkpointing.unsloth import (
    Unsloth_Offloaded_Gradient_Checkpointer,
)

transformers_version = version.parse(importlib.metadata.version("transformers"))
if transformers_version > version.parse("4.51.3"):
    from transformers.modeling_layers import GradientCheckpointingLayer

    def uses_gc_layers(decoder_layer):
        return isinstance(decoder_layer.func.__self__, GradientCheckpointingLayer)

else:

    def uses_gc_layers(_):
        return False


def hf_grad_checkpoint_offload_wrapper(
    decoder_layer, *args, use_reentrant=None
):  # pylint: disable=unused-argument
    if uses_gc_layers(decoder_layer):
        return Unsloth_Offloaded_Gradient_Checkpointer.apply(
            decoder_layer,
            *args,
        )

    return Unsloth_Offloaded_Gradient_Checkpointer.apply(
        (
            decoder_layer.func.__self__
            if isinstance(decoder_layer, partial)
            else decoder_layer.__self__
        ),
        *args,
    )
