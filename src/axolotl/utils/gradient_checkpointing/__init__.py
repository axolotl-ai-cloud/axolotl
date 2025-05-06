"""custom checkpointing utils"""

from functools import partial

import torch
from torch.utils.checkpoint import (
    CheckpointPolicy,
    checkpoint,
    create_selective_checkpoint_contexts,
)

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


aten = torch.ops.aten
compute_intensive_ops = [
    aten.mm,
    aten.convolution,
    aten.convolution_backward,
    aten.bmm,
    aten.addmm,
    aten._scaled_dot_product_flash_attention,
    aten._scaled_dot_product_efficient_attention,
    aten._flash_attention_forward,
    aten._efficient_attention_forward,
    aten.upsample_bilinear2d,
    aten._scaled_mm,
]


def policy_fn(ctx, op, *args, **kwargs):
    if op in compute_intensive_ops:
        return CheckpointPolicy.MUST_SAVE
    else:
        return CheckpointPolicy.PREFER_RECOMPUTE


context_fn = partial(create_selective_checkpoint_contexts, policy_fn)


def checkpoint_w_policy(
    decoder_layer, *args, use_reentrant=None
):  # pylint: disable=unused-argument
    return checkpoint(
        decoder_layer,
        *args,
        use_reentrant=use_reentrant,
        context_fn=context_fn,
    )
