"""custom checkpointing utils"""

import importlib
from functools import partial

from packaging import version

from axolotl.monkeypatch.gradient_checkpointing.offload_cpu import (  # noqa: F401
    CPU_Offloaded_Gradient_Checkpointer,
)
from axolotl.monkeypatch.gradient_checkpointing.offload_disk import (
    Disco,
)

transformers_version = version.parse(importlib.metadata.version("transformers"))
if transformers_version > version.parse("4.51.3"):
    from transformers.modeling_layers import GradientCheckpointingLayer

    def uses_gc_layers(decoder_layer):
        return isinstance(decoder_layer.func.__self__, GradientCheckpointingLayer)

else:

    def uses_gc_layers(_):
        return False


def hf_grad_checkpoint_offload_wrapper(decoder_layer, *args, use_reentrant=None):
    if uses_gc_layers(decoder_layer):
        return CPU_Offloaded_Gradient_Checkpointer.apply(
            decoder_layer,
            *args,
        )

    return CPU_Offloaded_Gradient_Checkpointer.apply(
        (
            decoder_layer.func.__self__
            if isinstance(decoder_layer, partial)
            else decoder_layer.__self__
        ),
        *args,
    )


def hf_grad_checkpoint_disk_offload_wrapper(decoder_layer, *args, use_reentrant=None):
    if uses_gc_layers(decoder_layer):
        return Disco.apply(
            decoder_layer,
            *args,
        )

    return Disco.apply(
        (
            decoder_layer.func.__self__
            if isinstance(decoder_layer, partial)
            else decoder_layer.__self__
        ),
        *args,
    )
