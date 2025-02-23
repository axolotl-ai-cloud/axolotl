from enum import Enum
from typing import Optional, Tuple, Callable

import torch
from yunchang import LongContextAttention, set_seq_parallel_pg

from axolotl.monkeypatch.attention.sequence_parallel import USPRingAttnType


def build_usp_fa_forward(ring_impl_type: USPRingAttnType) -> Callable:
    usp_attn = LongContextAttention(ring_impl_type.value)

    def flash_attention_forward(
            module: torch.nn.Module,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attention_mask: Optional[torch.Tensor],
            dropout: float = 0.0,
            scaling: Optional[float] = None,
            sliding_window: Optional[int] = None,
            softcap: Optional[float] = None,
            **kwargs,
    ) -> Tuple[torch.Tensor, None]:
        attn_output = usp_attn(
            query,
            key,
            value,
            dropout_p=dropout,
            softmax_scale=scaling,
            causal=True,
            softcap=softcap,
        )
        return attn_output, None

    return flash_attention_forward
