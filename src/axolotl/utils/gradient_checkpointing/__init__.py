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
    aten.mm.default,
    aten.bmm.default,
    aten.addmm.default,
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
