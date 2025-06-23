"""
Monkeypatch for SageAttention for use with transformers.

https://github.com/thu-ml/SageAttention/
"""

import torch
from transformers.integrations.sdpa_attention import repeat_kv

from axolotl.utils.logging import get_logger

logger = get_logger(__name__)

sageattn = None  # pylint: disable=invalid-name


def _is_sageattn_available():
    """Determine if SageAttention is available"""
    try:
        import sageattention  # noqa: F401 # pylint: disable=unused-import

        return True
    except ImportError:
        return False


if _is_sageattn_available():
    # import sageattn here if available
    from sageattention import sageattn


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

    # The base sageattn API does not take an explicit attention mask, requires `sageattn_varlen` handling.
    if attention_mask is not None:
        raise NotImplementedError(
            "Currently, the SageAttention integration does not support custom `attention_mask`. "
            "The integration requires updating to use `sageattn_varlen`. Please create a feature request."
        )

    if kwargs.get("position_ids") is not None:
        raise NotImplementedError(
            "Currently, the SageAttention integration does not support `position_ids`. "
            "The integration requires updating to use `sageattn_varlen`. Please create a feature request."
        )

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

    # Register sage_attention with the global attention interface
    ALL_ATTENTION_FUNCTIONS.register("sage_attention", sage_attention_forward)
