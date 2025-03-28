"""
Monkeypatch for Vision Llama for FA2 support
"""

# pylint: disable=duplicate-code

from typing import Optional, Tuple

import torch
from flash_attn.flash_attn_interface import flash_attn_func
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.models.mllama.configuration_mllama import MllamaTextConfig
from transformers.models.mllama.modeling_mllama import (
    MllamaTextCrossAttention,
    MllamaTextSelfAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.utils import is_flash_attn_greater_or_equal_2_10


class MllamaTextCrossFlashAttention2(MllamaTextCrossAttention):
    """
    Mllama flash cross-attention module. This module inherits from `MllamaTextCrossAttention` and
    implements the forward pass using Flash Attention for improved performance.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Check if flash attention version is greater or equal to 2.1
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        attention_mask: Optional[  # pylint: disable=unused-argument
            torch.Tensor
        ] = None,
        output_attentions: bool = False,
        use_cache: bool = False,  # pylint: disable=unused-argument
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        query_states = self.q_norm(query_states)

        if cross_attention_states is not None:
            key_states = self.k_proj(cross_attention_states)
            value_states = self.v_proj(cross_attention_states)
            key_states = key_states.view(
                bsz, -1, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2)
            value_states = value_states.view(
                bsz, -1, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2)
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            key_states = self.k_norm(key_states)
            if past_key_value is not None:
                key_states, value_states = past_key_value.update(
                    key_states,
                    value_states,
                    self.layer_idx,
                    {"cache_position": cache_position},
                )
        elif cache_position[0] != 0:
            key_states, value_states = (
                past_key_value.key_cache[self.layer_idx],
                past_key_value.value_cache[self.layer_idx],
            )
        else:
            raise ValueError(
                "Cross attention layer can't find neither `cross_attn_states` nor cached values for key/values!"
            )

        # Transpose to get the expected layout for flash attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Apply Flash Attention
        dropout_rate = self.dropout if self.training else 0.0
        output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            dropout_p=dropout_rate,
            softmax_scale=None,
            causal=False,
            return_attn_probs=output_attentions,
        )

        attn_output = output.contiguous().view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class MllamaTextSelfFlashAttention2(MllamaTextSelfAttention):
    """
    Mllama flash self-attention module. This module inherits from `MllamaTextSelfAttention` and
    implements the forward pass using Flash Attention for improved performance.
    """

    def __init__(self, config: MllamaTextConfig, layer_idx: int, *args, **kwargs):
        super().__init__(config, layer_idx, *args, **kwargs)

        # Check if flash attention version is greater or equal to 2.1
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,  # pylint: disable=unused-argument
        past_key_value=None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,  # pylint: disable=unused-argument
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x num_heads x head_dim
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Transpose to get the expected layout for flash attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.dropout if self.training else 0.0

        # Handle potential silent casting to float32
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = (
                    self.config._pre_quantization_dtype  # pylint: disable=protected-access
                )
            else:
                target_dtype = self.q_proj.weight.dtype

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=True,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


def patch_mllama():
    from transformers.models.mllama.modeling_mllama import (
        MLLAMA_TEXT_ATTENTION_CLASSES,
        MLLAMA_TEXT_CROSS_ATTENTION_CLASSES,
        MLLAMA_VISION_ATTENTION_CLASSES,
        MllamaPreTrainedModel,
    )

    MllamaPreTrainedModel._supports_flash_attn_2 = (  # pylint: disable=protected-access
        True
    )
    MLLAMA_TEXT_ATTENTION_CLASSES["flash_attention_2"] = MllamaTextSelfFlashAttention2
    MLLAMA_TEXT_CROSS_ATTENTION_CLASSES["flash_attention_2"] = (
        MllamaTextCrossFlashAttention2
    )
    # fallback to SDPA
    MLLAMA_VISION_ATTENTION_CLASSES["flash_attention_2"] = (
        MLLAMA_VISION_ATTENTION_CLASSES["sdpa"]
    )
