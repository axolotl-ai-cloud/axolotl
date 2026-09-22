"""Route transformers' flash attention calls through ring attention.

Modelled on ring-flash-attention's `adapters/hf_adapter.py` (MIT License, Copyright
2024 Zilin Zhu), trimmed to the transformers versions axolotl supports.
"""

import os

import torch
import torch.distributed as dist
import transformers.integrations.flash_attention
import transformers.modeling_flash_attention_utils
from transformers.modeling_flash_attention_utils import fa_peft_integration_check

from axolotl.monkeypatch.ring_attn.backends import NO_WINDOW, FlashAttnBackend
from axolotl.monkeypatch.ring_attn.batch_ring import ring_flash_attn_func
from axolotl.monkeypatch.ring_attn.llama3_varlen import (
    llama3_flash_attn_prepare_cu_seqlens,
    llama3_flash_attn_varlen_func,
)
from axolotl.utils.schemas.enums import RingAttnFunc

# This rank's varlen slices for the current forward pass; refreshed every step.
DATA_PARAMS: dict = {}


def update_ring_flash_attn_params(
    cu_seqlens: torch.Tensor, process_group: dist.ProcessGroup
):
    world_size = dist.get_world_size(group=process_group)
    rank = dist.get_rank(group=process_group)
    (
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        local_k_slice,
    ) = llama3_flash_attn_prepare_cu_seqlens(cu_seqlens, True, rank, world_size)
    DATA_PARAMS.update(
        {
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": cu_seqlens_k,
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_k": max_seqlen_k,
            "local_k_slice": local_k_slice,
        }
    )


def create_ring_flash_attention_forward(
    backend: FlashAttnBackend,
    process_group: dist.ProcessGroup,
    ring_attn_func: RingAttnFunc,
    heads_k_stride: int,
):
    """Build a drop-in for `transformers.modeling_flash_attention_utils._flash_attention_forward`."""

    def _flash_attention_forward(  # pylint: disable=too-many-arguments
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        query_length: int,
        is_causal: bool,
        dropout: float = 0.0,
        position_ids: torch.Tensor | None = None,
        softmax_scale: float | None = None,
        sliding_window: int | None = None,
        use_top_left_mask: bool = False,
        softcap: float | None = None,
        deterministic: bool | None = None,
        cu_seq_lens_q: torch.LongTensor | None = None,
        cu_seq_lens_k: torch.LongTensor | None = None,
        max_length_q: int | None = None,
        max_length_k: int | None = None,
        target_dtype: torch.dtype | None = None,
        attn_implementation: str | None = None,
        **kwargs,
    ):
        # flash_attn < 2.1 emitted top-left aligned causal masks (see transformers).
        if use_top_left_mask:
            causal = is_causal and query_length != 1
        else:
            causal = is_causal

        if dropout:
            raise ValueError(
                "Ring attention does not support attention dropout: the backward pass "
                "cannot replay the per-block dropout masks. Set the model's attention "
                "dropout to 0 when context_parallel_size > 1."
            )

        use_sliding_window = (
            sliding_window is not None and key_states.shape[1] > sliding_window
        )
        window_size = (
            (sliding_window, sliding_window) if use_sliding_window else NO_WINDOW
        )
        if deterministic is None:
            deterministic = os.environ.get("FLASH_ATTENTION_DETERMINISTIC", "0") == "1"

        query_states, key_states, value_states = fa_peft_integration_check(
            query_states, key_states, value_states, target_dtype
        )

        if ring_attn_func is RingAttnFunc.VARLEN_LLAMA3:
            if not causal:
                raise ValueError(
                    "varlen_llama3 ring attention only supports causal attention"
                )
            if query_states.size(0) != 1:
                raise ValueError(
                    "varlen_llama3 ring attention expects one packed row per micro "
                    f"batch, got batch size {query_states.size(0)}"
                )
            out = llama3_flash_attn_varlen_func(
                query_states.squeeze(0),
                key_states.squeeze(0),
                value_states.squeeze(0),
                cu_seqlens_q=DATA_PARAMS["cu_seqlens_q"],
                cu_seqlens_k=DATA_PARAMS["cu_seqlens_k"],
                max_seqlen_q=DATA_PARAMS["max_seqlen_q"],
                max_seqlen_k=DATA_PARAMS["max_seqlen_k"],
                heads_k_stride=heads_k_stride,
                local_k_slice=DATA_PARAMS["local_k_slice"],
                backend=backend,
                group=process_group,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                softcap=softcap or 0.0,
                deterministic=deterministic,
            )
            return out.unsqueeze(0)

        return ring_flash_attn_func(
            query_states,
            key_states,
            value_states,
            backend=backend,
            group=process_group,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap or 0.0,
            deterministic=deterministic,
        )

    return _flash_attention_forward


def substitute_hf_flash_attn(
    backend: FlashAttnBackend,
    process_group: dist.ProcessGroup,
    ring_attn_func: RingAttnFunc,
    heads_k_stride: int,
):
    """Point transformers' flash attention entry point at ring attention.

    Every flash flavour transformers knows (`flash_attention_2/3/4` and the hub
    kernels) dispatches through `integrations.flash_attention.flash_attention_forward`,
    which bound `_flash_attention_forward` at import time, so both module attributes
    are replaced.
    """
    ring_forward = create_ring_flash_attention_forward(
        backend=backend,
        process_group=process_group,
        ring_attn_func=ring_attn_func,
        heads_k_stride=heads_k_stride,
    )
    transformers.modeling_flash_attention_utils._flash_attention_forward = ring_forward
    transformers.integrations.flash_attention._flash_attention_forward = ring_forward
