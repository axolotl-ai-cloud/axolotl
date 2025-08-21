"""
HuggingFace flash attention adapter for basic ring attention (batch API).

Inspired by
https://github.com/zhuzilin/ring-flash-attention/blob/ce9fd3935ca0e5f0592bb0826cbed18ec69da729/ring_flash_attn/adapters/hf_adapter.py.
Our implementation closely follows the structure of that module, but we've minified it
somewhat to support only the latest versions of transformers.
"""

import os
from typing import Callable

import torch
import torch.distributed as dist
import transformers
import transformers.modeling_flash_attention_utils
from ring_flash_attn import ring_flash_attn_func
from ring_flash_attn.adapters.hf_adapter import check_params
from transformers.modeling_flash_attention_utils import is_flash_attn_greater_or_equal

try:
    from transformers.modeling_flash_attention_utils import _flash_supports_window
except ImportError:
    try:
        from transformers.modeling_flash_attention_utils import (
            _flash_supports_window_size as _flash_supports_window,
        )
    except ImportError:
        _flash_supports_window = True

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from axolotl.utils.schemas.enums import RingAttnFunc

RING_ATTN_FUNC_MAPPING = {
    RingAttnFunc.BATCH_RING: torch.compile(ring_flash_attn_func),
    # RingAttnFunc.BATCH_ZIGZAG: torch.compile(zigzag_ring_flash_attn_func),
    # RingAttnFunc.BATCH_STRIPE: torch.compile(stripe_flash_attn_func),
}


def create_flash_attn_forward_varlen_llama3(
    process_group: dist.ProcessGroup, ring_attn_func: RingAttnFunc
) -> Callable:
    """
    Create a ring flash attention forward function compatible with HuggingFace's
    interface.

    Args:
        process_group: A PyTorch distributed process group.
        ring_attn_func: Function from `ring_flash_attention` to replace HF flash
            attention with.

    Returns:
        A function that implements the ring flash attention forward pass with the
            signature expected by HuggingFace Transformers.
    """

    # transformers 4.48+

    def _flash_attention_forward(
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor,
        query_length: int,
        is_causal: bool,
        dropout: float = 0.0,
        position_ids: torch.Tensor | None = None,
        softmax_scale: float | None = None,
        sliding_window: int | None = None,
        use_top_left_mask: bool = False,
        softcap: float | None = None,
        deterministic: bool = None,
        cu_seq_lens_q: torch.LongTensor | None = None,
        cu_seq_lens_k: torch.LongTensor | None = None,
        max_length_q: int | None = None,
        max_length_k: int | None = None,
        target_dtype: torch.dtype | None = None,
        attn_implementation: str | None = None,
        **kwargs,
    ):
        """
        Calls the forward method of Ring Flash Attention.

        Args:
            query_states: Tensor containing the query vectors.
            key_states: Tensor containing the key vectors.
            value_states: Tensor containing the value vectors.
            attention_mask: Not used in this implementation.
            query_length: Integer representing the length of the query sequence.
            is_causal: Boolean indicating whether to apply a causal mask to the attention.
            dropout: Float representing the dropout probability. Default is 0.0.
            position_ids: Not used in this implementation.
            softmax_scale: Optional float value for the softmax scaling factor. Default is None.
            sliding_window: Optional integer defining the size of the sliding attention window.
                Default is None.
            use_top_left_mask: Boolean indicating whether to use a top-left mask for the attention.
                Default is False.
            softcap: Not used in this implementation.
            deterministic: Optional boolean to enforce deterministic computation. Default is None.
            cu_seq_lens_q: Not used in this implementation.
            cu_seq_lens_k: Not used in this implementation.
            max_length_q: Not used in this implementation.
            max_length_k: Not used in this implementation.
            target_dtype: Not used in this implementation.
            attn_implementation: Not used in this implementation.
            **kwargs: Additional keyword arguments. Not used in this implementation.

        Returns:
            torch.Tensor: The output of the attention mechanism, with shape
                `[batch_size, query_length, num_heads, head_dim]`.
        """
        if not use_top_left_mask:
            causal = is_causal
        else:
            causal = is_causal and query_length != 1

        # Handle sliding window
        use_sliding_windows = (
            _flash_supports_window
            and sliding_window is not None
            and key_states.shape[1] > sliding_window
        )
        window_size = (
            (sliding_window, sliding_window) if use_sliding_windows else (-1, -1)
        )

        # Handle deterministic mode
        if is_flash_attn_greater_or_equal("2.4.1"):
            if deterministic is None:
                deterministic = (
                    os.environ.get("FLASH_ATTENTION_DETERMINISTIC", "0") == "1"
                )

        # Call ring flash attention function
        attn_output = RING_ATTN_FUNC_MAPPING[ring_attn_func](
            query_states,
            key_states,
            value_states,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            alibi_slopes=None,
            deterministic=deterministic,
            return_attn_probs=False,
            group=process_group,
        )

        return attn_output

    return _flash_attention_forward


def substitute_hf_flash_attn(
    process_group: dist.ProcessGroup, ring_attn_func: RingAttnFunc
):
    """
    Substitute HuggingFace's flash attention implementation with ring-based implementation.

    Args:
        process_group: PyTorch distributed process group for communication.
        ring_attn_func: Function from `ring_flash_attention` to replace HF flash
            attention with.
    """
    try:
        # Substitute flash attention
        old_flash_attention_forward = (
            transformers.modeling_flash_attention_utils._flash_attention_forward
        )
        new_flash_attention_forward = create_flash_attn_forward_varlen_llama3(
            process_group=process_group, ring_attn_func=ring_attn_func
        )

        if check_params(old_flash_attention_forward, new_flash_attention_forward):
            transformers.modeling_flash_attention_utils._flash_attention_forward = (
                new_flash_attention_forward
            )
        else:
            raise ValueError(
                "The signature of the new flash attention forward function does not match the old one."
            )
    except Exception as exception:
        raise ValueError(
            f"The current transformer version {transformers.__version__} is not supported. "
            "Please use pip install -U transformers to upgrade to the latest version. "
            "If the code failed with the latest version, "
            f"please file an issue."
        ) from exception

    # Register with ALL_ATTENTION_FUNCTIONS if available
    if ALL_ATTENTION_FUNCTIONS is not None:
        from ring_flash_attn.adapters.hf_adapter import flash_attention_forward

        ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = flash_attention_forward
