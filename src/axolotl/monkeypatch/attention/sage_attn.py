"""
Monkeypatch for SageAttention for use with transformers.

https://github.com/thu-ml/SageAttention/
"""

import torch
from transformers.integrations.sdpa_attention import repeat_kv

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

sageattn = None  # pylint: disable=invalid-name
sageattn_varlen = None  # pylint: disable=invalid-name


def _is_sageattn_available():
    """Determine if SageAttention is available"""
    try:
        import sageattention  # noqa: F401 # pylint: disable=unused-import

        return True
    except ImportError:
        return False


if _is_sageattn_available():
    # import sageattn here if available
    from sageattention import sageattn, sageattn_varlen


def _check_sageattn_imported():
    """Check if SageAttention is imported. Raises an ImportError if not."""
    if sageattn is None:
        raise ImportError(
            "SageAttention is not installed. Please install it from source: "
            "`pip install git+https://github.com/thu-ml/SageAttention.git@1718ddc06dbc694bcf3c6b49ac28c1921aa2d8bd`"
        )


def sage_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """
    Forward pass for SageAttention compatible with transformers attention interfaces.

    https://github.com/thu-ml/SageAttention/
    """

    _check_sageattn_imported()

    if kwargs.get("output_attentions", False) or kwargs.get("head_mask") is not None:
        raise NotImplementedError(
            "SageAttention does not support `output_attentions=True` or `head_mask`."
        )

    # The base sageattn API does not support dropout.
    if dropout > 0.0:
        raise NotImplementedError("SageAttention does not support dropout.")

    # Handle Grouped-Query Attention (GQA) and Multi-Query Attention (MQA)
    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)

    # Calculate is_causal following transformers
    if is_causal is None:
        is_causal = (
            query.shape[2] > 1
            and attention_mask is None
            and getattr(module, "is_causal", True)
        )

    position_ids = kwargs.get("position_ids", None)
    query_length = query.shape[2]

    cu_seqlens_q = kwargs.get("cu_seqlens_q", None)
    cu_seqlens_k = kwargs.get("cu_seqlens_k", None)
    max_length_q = kwargs.get("max_length_q", None)
    max_length_k = kwargs.get("max_length_k", None)

    # Sample packing uses position_ids, so we check for it first
    if position_ids is not None and (
        max_length_q is not None
        or (query_length != 1 and not (torch.diff(position_ids, dim=-1) >= 0).all())
    ):
        # transpose inputs to NHD layout for use with FA2 utils
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        batch_size = query.size(0)

        from transformers.modeling_flash_attention_utils import (
            prepare_fa2_from_position_ids,
        )

        if cu_seqlens_q is None or cu_seqlens_k is None:
            query, key, value, indices_q, cu_seq_lens, max_seq_lens = (
                prepare_fa2_from_position_ids(query, key, value, position_ids)
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_length_q, max_length_k = max_seq_lens

        else:
            query = query.reshape(-1, query.size(-2), query.size(-1))
            key = key.reshape(-1, key.size(-2), key.size(-1))
            value = value.reshape(-1, value.size(-2), value.size(-1))

        attn_output_unpad = sageattn_varlen(
            q=query,
            k=key,
            v=value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_length_q,
            max_seqlen_k=max_length_k,
            is_causal=is_causal,
            sm_scale=scaling,
            tensor_layout="NHD",
        )

        attn_output = attn_output_unpad.view(
            batch_size, -1, attn_output_unpad.size(-2), attn_output_unpad.size(-1)
        )

    elif attention_mask is not None:
        assert attention_mask.ndim == 2, "Attention mask must be 2D"

        from transformers.modeling_flash_attention_utils import (
            _upad_input,
        )

        # transpose inputs to NHD layout for use with FA2 utils
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        batch_size = query.shape[0]

        query, key, value, indices_q, cu_seq_lens, max_seq_lens = _upad_input(
            query, key, value, attention_mask, query_length
        )
        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_q, max_seqlen_k = max_seq_lens

        attn_output_unpad = sageattn_varlen(
            q=query,
            k=key,
            v=value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            is_causal=is_causal,
            sm_scale=scaling,
            tensor_layout="NHD",
        )

        from flash_attn.bert_padding import pad_input

        attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)

    else:
        # Use standard sageattn
        # The input layout for transformers models is (batch_size, num_heads, seq_len, head_dim),
        # which corresponds to SageAttention's "HND" layout.
        attn_output = sageattn(
            q=query,
            k=key,
            v=value,
            tensor_layout="HND",
            is_causal=is_causal,
            sm_scale=scaling,
        )

        # SageAttention with "HND" returns (batch, heads, seq_len, head_dim)
        # Transformers expects (batch, seq_len, heads, head_dim) for the output
        # So we need to transpose dimensions 1 and 2
        attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None


def patch_sageattn():
    """Patch SageAttention for use with transformers."""

    _check_sageattn_imported()

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    # Replace flash attention with sage attention
    ALL_ATTENTION_FUNCTIONS.register("flash_attention_2", sage_attention_forward)

    # Note: New method after transformers refactor to use ALL_MASK_ATTENTION_FUNCTIONS
    # Register sage_attention with the global attention interface
    # ALL_ATTENTION_FUNCTIONS.register("sage_attention", sage_attention_forward)

    # from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, flash_attention_mask

    # ALL_MASK_ATTENTION_FUNCTIONS.register("sage_attention", flash_attention_mask)

    LOG.info("SageAttention patched successfully")
