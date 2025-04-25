"""custom checkpointing utils"""

from functools import partial

from axolotl.utils.gradient_checkpointing.unsloth import (
    Unsloth_Offloaded_Gradient_Checkpointer,
)


def hf_grad_checkpoint_offload_wrapper(
    decoder_layer, *args, use_reentrant=None
):  # pylint: disable=unused-argument
    return Unsloth_Offloaded_Gradient_Checkpointer.apply(
        (
            decoder_layer.func.__self__
            if isinstance(decoder_layer, partial)
            else decoder_layer.__self__
        ),
        *args,
    )
