"""custom checkpointing utils"""
from axolotl.utils.gradient_checkpointing.unsloth import (
    Unsloth_Offloaded_Gradient_Checkpointer,
)


def hf_grad_checkpoint_unsloth_wrapper(decoder_layer, *args, **kwargs):
    return Unsloth_Offloaded_Gradient_Checkpointer.apply(
        decoder_layer.__call__,
        *args,
        **kwargs,
    )
