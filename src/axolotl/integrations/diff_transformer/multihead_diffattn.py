"""Re-implemention of differential attention."""
# pylint: disable=invalid-name
import logging
import math
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import LlamaRMSNorm as RMSNorm
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


class DifferentialAttention(nn.Module):
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
        is_causal: Whether to use causal (masked) attention
    """

    def __init__(
        self,
        config: Any,
        layer_idx: int,
        dtype: torch.dtype,
        is_causal: bool = True,
    ):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.is_causal = is_causal
        # self.head_dim = self.hidden_size // self.num_heads
        self.head_dim = self.hidden_size // self.num_heads // 2
        self.num_key_value_heads = getattr(
            config, "num_key_value_heads", self.num_heads
        )
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.scaling = (self.head_dim) ** -0.5

        # Initialize projections with correct dtype
        self.q_proj = nn.Linear(
            self.hidden_size, self.hidden_size, bias=False, dtype=dtype
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.hidden_size // self.num_key_value_groups,
            bias=False,
            dtype=dtype,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.hidden_size // self.num_key_value_groups,
            bias=False,
            dtype=dtype,
        )

        self.o_proj = nn.Linear(
            self.hidden_size, self.hidden_size, bias=False, dtype=dtype
        )

        # Initialize differential attention parameters
        self.lambda_init = lambda_init_fn(self.layer_idx)
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

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,  # pylint: disable=unused-argument
    ) -> tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[tuple[torch.Tensor, torch.Tensor]],
    ]:
        bsz, tgt_len, _ = hidden_states.size()

        # Project queries, keys and values
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for attention
        q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, tgt_len, 2 * self.num_key_value_heads, self.head_dim).transpose(
            1, 2
        )
        v = v.view(bsz, tgt_len, self.num_key_value_heads, 2 * self.head_dim).transpose(
            1, 2
        )

        # Generate or unpack cos, sin for rotary positional embeddings
        if position_embeddings is None:
            if position_ids is None:
                position_ids = torch.arange(
                    0, tgt_len, dtype=torch.long, device=q.device
                )
            cos, sin = self.rotary_emb(q, position_ids)
        else:
            cos, sin = position_embeddings

        # Need to adjust cos, sin to match the halved head_dim
        cos = cos[..., : self.head_dim]
        sin = sin[..., : self.head_dim]
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

            # Update cache and get back concatenated states
            k, v = past_key_value.update(k, v, self.layer_idx, cache_kwargs)

        # Prepare for attention
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        # Scale query
        q = q * self.scaling

        # Calculate attention scores
        attn_weights = torch.matmul(q, k.transpose(-1, -2))

        # Apply causal mask
        if attention_mask is None:
            attention_mask = torch.triu(
                torch.full((tgt_len, tgt_len), float("-inf"), device=q.device),
                diagonal=1,
            ).type_as(attn_weights)
        attn_weights = torch.nan_to_num(attn_weights)
        attn_weights = attn_weights + attention_mask

        # Apply softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
            attn_weights
        )

        # Calculate lambda
        lambda_1 = torch.exp(
            torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()
        ).type_as(q)
        lambda_2 = torch.exp(
            torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()
        ).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        # Apply differential attention
        attn_weights = attn_weights.view(
            bsz, self.num_heads, 2, -1, attn_weights.size(-1)
        )
        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]

        # Apply attention to values
        attn = torch.matmul(attn_weights, v)

        # Apply sublayer norm
        attn = self.subln(attn).type_as(attn)
        attn = attn * (1 - self.lambda_init)

        # Reshape and project output
        attn = attn.transpose(1, 2).reshape(
            bsz, tgt_len, self.num_heads * 2 * self.head_dim
        )
        attn = self.o_proj(attn)

        # Return in exact format expected by LLaMA
        if output_attentions:
            return attn, attn_weights, past_key_value
        return attn, None, past_key_value
