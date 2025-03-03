"""
Utilities for sequence parallelism implementation.

Modified from:
https://github.com/Qihoo360/360-LLaMA-Factory/blob/f295a5760cceebe069fb5b975813d2c945598acb/src/llamafactory/model/model_utils/sequence_parallel.py
"""

from functools import partial

import torch.distributed as dist
import transformers
import transformers.modeling_attn_mask_utils
from ring_flash_attn import (
    ring_flash_attn_func,
    stripe_flash_attn_func,
    zigzag_ring_flash_attn_func,
)


def ring_flash_attn_forward(
    query_states,
    key_states,
    value_states,
    attention_mask,
    q_len,
    dropout=0,
    sliding_window=None,
    is_causal=True,
    group=None,
    **kwargs,
):
    attn_output = ring_flash_attn_func(
        query_states, key_states, value_states, dropout, causal=is_causal, group=group
    )

    return attn_output


def zigzag_flash_attn_forward(
    query_states,
    key_states,
    value_states,
    attention_mask,
    q_len,
    dropout=0,
    sliding_window=None,
    is_causal=True,
    group=None,
    **kwargs,
):
    attn_output = zigzag_ring_flash_attn_func(
        query_states, key_states, value_states, dropout, causal=is_causal, group=group
    )

    return attn_output


def stripe_flash_attn_forward(
    query_states,
    key_states,
    value_states,
    attention_mask,
    q_len,
    dropout=0,
    sliding_window=None,
    is_causal=True,
    group=None,
    **kwargs,
):
    attn_output = stripe_flash_attn_func(
        query_states, key_states, value_states, dropout, causal=is_causal, group=group
    )

    return attn_output


def init_sp_group(sp_size):
    assert dist.is_initialized()
    world_size = dist.get_world_size()
    assert (
        world_size % sp_size == 0
    ), "Total number of GPUs must be a multiple of sequence_parallel_size."

    sp_group_num = world_size // sp_size
    sp_ranks_list = [
        list(range(i * sp_size, i * sp_size + sp_size)) for i in range(sp_group_num)
    ]

    sp_groups = [dist.new_group(sp_ranks_this) for sp_ranks_this in sp_ranks_list]

    global_rank_this = dist.get_rank()
    sp_idx = global_rank_this // sp_size
    return sp_groups[sp_idx]


def apply_sequence_parallel(cfg):
    if cfg.sequence_parallel_size == 1:
        return None  # no sequence parallelism

    # init sequence-parallel groups here
    group_this = init_sp_group(cfg.sequence_parallel_size)

    if cfg.sequence_parallel_mode == "ring":
        new_flash_attention_forward = partial(ring_flash_attn_forward, group=group_this)
    elif cfg.sequence_parallel_mode == "zigzag-ring":
        new_flash_attention_forward = partial(
            zigzag_flash_attn_forward, group=group_this
        )
    elif cfg.sequence_parallel_mode == "stripe":
        new_flash_attention_forward = partial(
            stripe_flash_attn_forward, group=group_this
        )
    else:
        raise NotImplementedError(
            "Other sequence parallel modes are to be implemented."
        )

    # monkey patching
    transformers.modeling_flash_attention_utils._flash_attention_forward = (
        new_flash_attention_forward
    )

    return group_this
