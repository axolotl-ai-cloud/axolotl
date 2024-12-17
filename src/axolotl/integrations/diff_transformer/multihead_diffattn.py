"""Re-implemention of differential attention."""
# pylint: disable=invalid-name
import logging
import math
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
import transformers
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
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


class LlamaDifferentialAttention(nn.Module):
    """Differential Attention implementation as described in the Diff Transformer paper.

    This implements a modified attention mechanism that computes the difference between
    two attention patterns, scaled by learned lambda parameters. The mechanism helps
    reduce noise in the attention weights for irrelevant / less relevant tokens.

    Key components:
    - Split head dimension for differential computation
    - Learned lambda parameters that control attention scaling
    - Sublayer normalization on the attention output

    See:
    - https://arxiv.org/abs/2410.05258
    - https://github.com/microsoft/unilm/tree/master/Diff-Transformer

    Args:
        config: Model configuration object containing hidden size, number of heads etc.
        layer_idx: Index of this layer in the transformer stack
        dtype: Data type for the layer parameters
    """

    def __init__(
        self,
        config: Any,
        layer_idx: int,
    ):
        super().__init__()

        # Base model config
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.base_num_heads = config.num_attention_heads
        self.base_num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.layer_idx = layer_idx
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        dtype = torch.float32

        # For Q1 and Q2
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.hidden_size * 2,
            bias=False,
            dtype=dtype,
        )

        # For K1 and K2
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.hidden_size // self.base_num_heads * self.base_num_kv_heads * 2,
            bias=False,
            dtype=dtype,
        )

        # Single V projection
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.hidden_size // self.base_num_heads * self.base_num_kv_heads,
            bias=False,
            dtype=dtype,
        )

        # Output projection
        self.o_proj = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
            dtype=dtype,
        )

        # Initialize differential attention parameters
        self.lambda_init = nn.Parameter(
            torch.full((), lambda_init_fn(self.layer_idx), dtype=dtype),
            requires_grad=False,
        )
        self.lambda_q1 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=dtype).normal_(mean=0, std=0.1)
        )
        self.lambda_k1 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=dtype).normal_(mean=0, std=0.1)
        )
        self.lambda_q2 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=dtype).normal_(mean=0, std=0.1)
        )
        self.lambda_k2 = nn.Parameter(
            torch.zeros(self.head_dim, dtype=dtype).normal_(mean=0, std=0.1)
        )

        self.rotary_emb = LlamaRotaryEmbedding(config=config)

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
    ) -> tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[tuple[torch.Tensor, torch.Tensor]],
    ]:
        bsz, q_len, _ = hidden_states.size()

        # Project to Q1,Q2 and K1,K2
        qp = self.q_proj(hidden_states)
        kp = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Split into Q1,Q2 and K1,K2
        q1, q2 = qp.chunk(2, dim=-1)
        k1, k2 = kp.chunk(2, dim=-1)

        # Reshape Q1,Q2 for attention
        q1 = q1.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        q2 = q2.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        # Reshape K1,K2 for attention
        k1 = k1.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        k2 = k2.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        # Reshape V
        v = v.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        if position_embeddings is None:
            if position_ids is None:
                position_ids = torch.arange(q_len, device=q1.device)
            cos, sin = self.rotary_emb(q1, position_ids)
        else:
            cos, sin = position_embeddings

        q1, k1 = apply_rotary_pos_emb(q1, k1, cos, sin)
        q2, k2 = apply_rotary_pos_emb(q2, k2, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k = torch.stack([k1, k2], dim=1)
            k, v = past_key_value.update(k, v, self.layer_idx, cache_kwargs)
            k1, k2 = k.unbind(dim=1)

        # Repeat KV heads to match Q heads
        k1 = repeat_kv(k1, self.base_num_heads // self.base_num_kv_heads)
        k2 = repeat_kv(k2, self.base_num_heads // self.base_num_kv_heads)
        v = repeat_kv(v, self.base_num_heads // self.base_num_kv_heads)

        # Calculate attention scores for both parts
        # NOTE(Dan): the Differential Transformers paper scales by a constant scaling factor
        # instead of sqrt(head_dim). This could be set on the class as `self.scaling`.
        attn_weights1 = torch.matmul(q1, k1.transpose(-1, -2)) / math.sqrt(
            self.head_dim
        )
        attn_weights2 = torch.matmul(q2, k2.transpose(-1, -2)) / math.sqrt(
            self.head_dim
        )

        # Add this debug step right after computing attention weights in the forward pass
        attn_weights1 = torch.matmul(q1, k1.transpose(-1, -2)) / math.sqrt(
            self.head_dim
        )
        attn_weights2 = torch.matmul(q2, k2.transpose(-1, -2)) / math.sqrt(
            self.head_dim
        )

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : k1.shape[-2]]
            attn_weights1 = attn_weights1 + causal_mask
            attn_weights2 = attn_weights2 + causal_mask

        # Apply softmax separately as per paper
        attn_weights1 = F.softmax(attn_weights1, dim=-1, dtype=torch.float32).type_as(
            attn_weights1
        )
        attn_weights2 = F.softmax(attn_weights2, dim=-1, dtype=torch.float32).type_as(
            attn_weights2
        )
        attn_weights1 = F.dropout(
            attn_weights1, p=self.attention_dropout, training=self.training
        )
        attn_weights2 = F.dropout(
            attn_weights2, p=self.attention_dropout, training=self.training
        )

        # Calculate lambda
        lambda_1 = torch.exp(
            torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()
        ).type_as(q1)
        lambda_2 = torch.exp(
            torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()
        ).type_as(q1)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        # Compute differential attention (following paper's formula)
        attn_weights = attn_weights1 - lambda_full * attn_weights2

        # Apply attention weights to values
        attn = torch.matmul(attn_weights, v)

        # Apply sublayer norm and scaling
        # NOTE(Dan): The differential transformers paper applies sublayer normalization at this
        # point, but this is typically done outside of the attention layer. It would look something
        # like: `attn = self.subln(attn).type_as(attn)`, using `LlamaRMSNorm` or similar.
        attn = attn * (1 - self.lambda_init)

        # Reshape to output
        attn = attn.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
        attn = self.o_proj(attn)

        if output_attentions:
            return attn, attn_weights, past_key_value
        return attn, None, past_key_value


class LlamaDifferentialSdpaAttention(LlamaDifferentialAttention):
    """Differential Attention implementation as described in the Diff Transformer paper.
    This implements the same logic as `LlamaDifferentialAttention`, but uses
    `scaled_dot_product_attention` instead of "manually" computing it under the hood.

    This implements a modified attention mechanism that computes the difference between
    two attention patterns, scaled by learned lambda parameters. The mechanism helps
    reduce noise in the attention weights for irrelevant / less relevant tokens.

    Key components:
    - Split head dimension for differential computation
    - Learned lambda parameters that control attention scaling
    - Sublayer normalization on the attention output

    See:
    - https://arxiv.org/abs/2410.05258
    - https://github.com/microsoft/unilm/tree/master/Diff-Transformer

    Args:
        config: Model configuration object containing hidden size, number of heads etc.
        layer_idx: Index of this layer in the transformer stack
        dtype: Data type for the layer parameters
    """

    def forward(
        self,
        hidden_states: torch.Tensor,  # [bsz, seq_len, hidden_size]
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,  # pylint: disable=unused-argument
    ) -> tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[tuple[torch.Tensor, torch.Tensor]],
    ]:
        if output_attentions:
            transformers.logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,  # pylint: disable=duplicate-code
                attention_mask=attention_mask,  # pylint: disable=duplicate-code
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        bsz, q_len, _ = hidden_states.size()

        # Project to Q1,Q2 and K1,K2
        qp = self.q_proj(hidden_states)
        kp = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Split into Q1,Q2 and K1,K2
        q1, q2 = qp.chunk(2, dim=-1)
        k1, k2 = kp.chunk(2, dim=-1)

        # Reshape Q1,Q2 for attention
        q1 = q1.view(bsz, q_len, self.base_num_heads, self.head_dim).transpose(1, 2)
        q2 = q2.view(bsz, q_len, self.base_num_heads, self.head_dim).transpose(1, 2)
        # Reshape K1,K2 for attention
        k1 = k1.view(bsz, q_len, self.base_num_kv_heads, self.head_dim).transpose(1, 2)
        k2 = k2.view(bsz, q_len, self.base_num_kv_heads, self.head_dim).transpose(1, 2)
        # Reshape V
        v = v.view(bsz, q_len, self.base_num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        if position_embeddings is None:
            if position_ids is None:
                position_ids = torch.arange(q_len, device=q1.device)
            cos, sin = self.rotary_emb(q1, position_ids)
        else:
            cos, sin = position_embeddings

        q1, k1 = apply_rotary_pos_emb(q1, k1, cos, sin)
        q2, k2 = apply_rotary_pos_emb(q2, k2, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k = torch.stack([k1, k2], dim=1)
            k, v = past_key_value.update(k, v, self.layer_idx, cache_kwargs)
            k1, k2 = k.unbind(dim=1)

        # Repeat KV heads to match Q heads
        k1 = repeat_kv(k1, self.base_num_heads // self.base_num_kv_heads)
        k2 = repeat_kv(k2, self.base_num_heads // self.base_num_kv_heads)
        v = repeat_kv(v, self.base_num_heads // self.base_num_kv_heads)

        causal_mask = None
        if attention_mask is not None:
            causal_mask = attention_mask
            causal_mask = causal_mask[:, :, :, : k1.shape[-2]]

        # SDPA with memory-efficient backend requires contiguous inputs on CUDA
        if q1.device.type == "cuda" and causal_mask is not None:
            q1, q2 = q1.contiguous(), q2.contiguous()
            k1, k2 = k1.contiguous(), k2.contiguous()
            v = v.contiguous()

        # Calculate attention using SDPA
        is_causal = attention_mask is None and q_len > 1

        attn_output1 = F.scaled_dot_product_attention(
            q1,
            k1,
            v,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        attn_output2 = F.scaled_dot_product_attention(
            q2,
            k2,
            v,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        # Calculate lambda
        lambda_1 = torch.exp(
            torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()
        ).type_as(q1)
        lambda_2 = torch.exp(
            torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()
        ).type_as(q1)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        # Combine the attention outputs
        attn = attn_output1 - lambda_full * attn_output2

        # Apply sublayer norm and scaling
        attn = attn * (1 - self.lambda_init)

        # Reshape to output
        attn = attn.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
        attn = self.o_proj(attn)

        if output_attentions:
            return (
                attn,
                None,
                past_key_value,
            )  # Note: can't return attn_weights with SDPA
        return attn, None, past_key_value
