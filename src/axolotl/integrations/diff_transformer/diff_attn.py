"""Re-implemention of differential attention."""
# pylint: disable=invalid-name

import logging
import math
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

try:
    from flash_attn.flash_attn_interface import flash_attn_func

    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(batch_size, n_kv_heads, n_rep, slen, head_dim)
        .reshape(batch_size, n_kv_heads * n_rep, slen, head_dim)
    )


def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class LlamaDifferentialAttentionBase(nn.Module):
    """Base class for differential attention implementations."""

    def __init__(self, config: Any, layer_idx: int):
        super().__init__()
        self.config = config
        self._init_config(config, layer_idx)
        self._init_projections()
        self._init_differential_params()
        self._init_normalization(config)

    def _init_config(self, config: Any, layer_idx: int):
        """Initialize configuration parameters."""
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.base_num_heads = config.num_attention_heads
        self.base_num_kv_heads = config.num_key_value_heads
        self.layer_idx = layer_idx
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.split_heads = config.split_heads

        if config.split_heads:
            # Split heads mode - single projections
            self.head_dim = config.hidden_size // config.num_attention_heads
            # NOTE: This rounds down `base_num_heads / 2` as opposed to the original
            # implementation, which asserts `self.base_num_heads` is even
            self.heads_per_component = self.base_num_heads // 2
            self.value_head_dim = 2 * self.head_dim
        else:
            # Double projection mode
            self.head_dim = config.hidden_size // config.num_attention_heads
            self.heads_per_component = self.base_num_heads
            self.value_head_dim = self.head_dim

    def _init_projections(self):
        """Initialize Q, K, V projections."""
        if self.split_heads:
            # Split heads mode - single projections
            q_out_dim = self.hidden_size
            k_out_dim = self.hidden_size // self.base_num_heads * self.base_num_kv_heads
        else:
            # Double projection mode
            q_out_dim = self.hidden_size * 2
            k_out_dim = (
                self.hidden_size // self.base_num_heads * self.base_num_kv_heads * 2
            )

        self.q_proj = nn.Linear(self.hidden_size, q_out_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, k_out_dim, bias=False)
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.hidden_size // self.base_num_heads * self.base_num_kv_heads,
            bias=False,
        )
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def _init_differential_params(self):
        """Initialize differential attention parameters."""
        self.lambda_init = nn.Parameter(
            torch.full((), lambda_init_fn(self.layer_idx)),
            requires_grad=False,
        )
        self.lambda_q1 = nn.Parameter(
            torch.zeros(self.head_dim).normal_(mean=0, std=0.1)
        )
        self.lambda_k1 = nn.Parameter(
            torch.zeros(self.head_dim).normal_(mean=0, std=0.1)
        )
        self.lambda_q2 = nn.Parameter(
            torch.zeros(self.head_dim).normal_(mean=0, std=0.1)
        )
        self.lambda_k2 = nn.Parameter(
            torch.zeros(self.head_dim).normal_(mean=0, std=0.1)
        )
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

    def _init_normalization(self, config):
        """Initialize normalization layers."""
        sublayer_norm = getattr(config, "sublayer_norm", True)
        if sublayer_norm:
            self.subln = LlamaRMSNorm(self.value_head_dim, eps=config.rms_norm_eps)
        else:
            self.subln = nn.Identity()

    def _prepare_attention_inputs(self, hidden_states: torch.Tensor):
        """Prepare inputs for attention computation."""
        bsz, q_len, _ = hidden_states.size()

        # Project and split
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        q1, q2 = q.chunk(2, dim=-1)
        k1, k2 = k.chunk(2, dim=-1)

        # Reshape
        q1 = q1.view(bsz, q_len, self.heads_per_component, self.head_dim).transpose(
            1, 2
        )
        q2 = q2.view(bsz, q_len, self.heads_per_component, self.head_dim).transpose(
            1, 2
        )
        k1 = k1.view(bsz, q_len, self.heads_per_component, self.head_dim).transpose(
            1, 2
        )
        k2 = k2.view(bsz, q_len, self.heads_per_component, self.head_dim).transpose(
            1, 2
        )
        v = v.view(bsz, q_len, self.heads_per_component, self.value_head_dim).transpose(
            1, 2
        )

        return q1, q2, k1, k2, v

    def _apply_rotary_embeddings(
        self, q1, q2, k1, k2, position_ids, position_embeddings
    ):
        """Apply rotary embeddings to queries and keys."""
        if position_embeddings is None:
            LOG.warning(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(q1, position_ids)
        else:
            cos, sin = position_embeddings

        q1, k1 = apply_rotary_pos_emb(q1, k1, cos, sin)
        q2, k2 = apply_rotary_pos_emb(q2, k2, cos, sin)

        return q1, q2, k1, k2, cos, sin

    def _handle_cache(self, k1, k2, v, past_key_value, cache_kwargs):
        """Handle caching for autoregressive generation."""
        if past_key_value is not None:
            k = torch.stack([k1, k2], dim=1)
            k, v = past_key_value.update(k, v, self.layer_idx, cache_kwargs)
            k1, k2 = k.unbind(dim=1)

        # Repeat KV heads
        k1 = repeat_kv(k1, self.base_num_heads // self.base_num_kv_heads)
        k2 = repeat_kv(k2, self.base_num_heads // self.base_num_kv_heads)
        v = repeat_kv(v, self.base_num_heads // self.base_num_kv_heads)

        return k1, k2, v

    def _compute_lambda(self, q1):
        """Compute lambda values for differential attention."""
        lambda_1 = torch.exp(
            torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()
        ).type_as(q1)
        lambda_2 = torch.exp(
            torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()
        ).type_as(q1)
        return lambda_1 - lambda_2 + self.lambda_init

    def _process_attention_output(self, attn, bsz, q_len):
        """Process and project attention output."""
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
        return self.o_proj(attn)


class LlamaDifferentialAttention(LlamaDifferentialAttentionBase):
    """Standard implementation of differential attention."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,  # pylint: disable=unused-argument
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,  # pylint: disable=unused-argument
    ):
        bsz, q_len, _ = hidden_states.size()
        q1, q2, k1, k2, v = self._prepare_attention_inputs(hidden_states)
        q1, q2, k1, k2, cos, sin = self._apply_rotary_embeddings(
            q1, q2, k1, k2, position_ids, position_embeddings
        )

        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        k1, k2, v = self._handle_cache(k1, k2, v, past_key_value, cache_kwargs)

        # Standard attention computation
        attn1 = torch.matmul(q1, k1.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn2 = torch.matmul(q2, k2.transpose(-1, -2)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : k1.shape[-2]]
            attn1 = attn1 + causal_mask
            attn2 = attn2 + causal_mask

        attn1 = F.softmax(attn1, dim=-1, dtype=torch.float32).type_as(attn1)
        attn2 = F.softmax(attn2, dim=-1, dtype=torch.float32).type_as(attn2)

        dropout_p = self.attention_dropout if self.training else 0.0
        attn1 = F.dropout(attn1, p=dropout_p, training=self.training)
        attn2 = F.dropout(attn2, p=dropout_p, training=self.training)

        lambda_full = self._compute_lambda(q1)
        attn = torch.matmul(attn1, v) - lambda_full * torch.matmul(attn2, v)
        attn = self._process_attention_output(attn, bsz, q_len)

        if output_attentions:
            attn_weights = attn1 - lambda_full * attn2
            attn_weights = attn_weights.view(bsz, self.heads_per_component, q_len, -1)
            return attn, attn_weights, past_key_value
        return attn, None, past_key_value


class LlamaDifferentialSdpaAttention(LlamaDifferentialAttentionBase):
    """SDPA-based implementation of differential attention."""

    # pylint: disable=duplicate-code
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,  # pylint: disable=unused-argument
    ):
        if output_attentions:
            LOG.warning(
                "LlamaDifferentialModel is using LlamaDifferentialSdpaAttention, but "
                + "`torch.nn.functional.scaled_dot_product_attention` does not support "
                + "`output_attentions=True`. Falling back to the eager attention implementation."
            )
            return LlamaDifferentialAttention.forward(
                self,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
            )

        bsz, q_len, _ = hidden_states.size()
        q1, q2, k1, k2, v = self._prepare_attention_inputs(hidden_states)
        q1, q2, k1, k2, cos, sin = self._apply_rotary_embeddings(
            q1, q2, k1, k2, position_ids, position_embeddings
        )

        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        k1, k2, v = self._handle_cache(k1, k2, v, past_key_value, cache_kwargs)

        # SDPA-specific attention computation
        causal_mask = (
            None if attention_mask is None else attention_mask[:, :, :, : k1.shape[-2]]
        )
        is_causal = attention_mask is None and q_len > 1
        dropout_p = self.attention_dropout if self.training else 0.0

        if q1.device.type == "cuda" and causal_mask is not None:
            q1, q2 = q1.contiguous(), q2.contiguous()
            k1, k2 = k1.contiguous(), k2.contiguous()
            v = v.contiguous()

        attn1 = F.scaled_dot_product_attention(
            q1, k1, v, attn_mask=causal_mask, dropout_p=dropout_p, is_causal=is_causal
        )
        attn2 = F.scaled_dot_product_attention(
            q2, k2, v, attn_mask=causal_mask, dropout_p=dropout_p, is_causal=is_causal
        )

        lambda_full = self._compute_lambda(q1)
        attn = attn1 - lambda_full * attn2

        attn = self._process_attention_output(attn, bsz, q_len)
        return attn, None, past_key_value


class LlamaDifferentialFlashAttention2(LlamaDifferentialAttentionBase):
    """Flash Attention 2-based implementation of differential attention."""

    def __init__(self, *args, **kwargs):
        if not FLASH_ATTENTION_AVAILABLE:
            raise ImportError(
                "LlamaDifferentialFlashAttention2 requires flash-attn library. "
                "Please install with `pip install flash-attn --no-build-isolation`"
            )

        super().__init__(*args, **kwargs)

    # pylint: disable=duplicate-code
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,  # pylint: disable=unused-argument
    ):
        if output_attentions:
            LOG.warning(
                "LlamaDifferentialModel is using LlamaDifferentialFlashAttention2, but "
                + "flash attenion does not support `output_attentions=True`. Falling back "
                + "to the eager attention implementation."
            )
            return LlamaDifferentialAttention.forward(
                self,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
            )

        bsz, q_len, _ = hidden_states.size()
        q1, q2, k1, k2, v = self._prepare_attention_inputs(hidden_states)
        q1, q2, k1, k2, cos, sin = self._apply_rotary_embeddings(
            q1, q2, k1, k2, position_ids, position_embeddings
        )

        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        k1, k2, v = self._handle_cache(k1, k2, v, past_key_value, cache_kwargs)

        # Flash Attention specific processing
        q1, q2 = q1.transpose(1, 2), q2.transpose(1, 2)
        k1, k2 = k1.transpose(1, 2), k2.transpose(1, 2)
        v = v.transpose(1, 2)

        dropout_p = self.attention_dropout if self.training else 0.0

        if self.split_heads:
            v1, v2 = v.chunk(2, dim=-1)
            attn11 = flash_attn_func(q1, k1, v1, dropout_p=dropout_p, causal=True)
            attn12 = flash_attn_func(q1, k1, v2, dropout_p=dropout_p, causal=True)
            attn1 = torch.cat([attn11, attn12], dim=-1)

            attn21 = flash_attn_func(q2, k2, v1, dropout_p=dropout_p, causal=True)
            attn22 = flash_attn_func(q2, k2, v2, dropout_p=dropout_p, causal=True)
            attn2 = torch.cat([attn21, attn22], dim=-1)
        else:
            attn1 = flash_attn_func(q1, k1, v, dropout_p=dropout_p, causal=True)
            attn2 = flash_attn_func(q2, k2, v, dropout_p=dropout_p, causal=True)

        attn1, attn2 = attn1.transpose(1, 2), attn2.transpose(1, 2)

        lambda_full = self._compute_lambda(q1)
        attn = attn1 - lambda_full * attn2

        attn = self._process_attention_output(attn, bsz, q_len)
        return attn, None, past_key_value
