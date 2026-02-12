"""
Hugging Face-compatible Laneformer implementation.

This file defines:
- LaneformerTPPreTrainedModel
- LaneformerTPModel (backbone)
- LaneformerTPForCausalLM (head + loss)

It mirrors the TorchTitan Laneformer structure so that state_dict keys largely match,
which makes converting checkpoints trivial (often identity mapping).
"""

from __future__ import annotations

import math

from enum import Enum

from functools import partial
from typing import Optional, Tuple, Union, Unpack

import torch
import torch.nn.functional as F

from torch import nn
from transformers import GradientCheckpointingLayer
from transformers.cache_utils import Cache, DynamicCache
from torch.utils.checkpoint import checkpoint

from transformers.generation import GenerationMixin
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel

from transformers.utils import auto_docstring, can_return_tuple, TransformersKwargs

from .configuration_laneformer import LaneformerTPConfig


# ============================================================
# Rotary helpers (mirrors your implementation)
# ============================================================
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ndim = x.ndim
    assert ndim > 1
    seqlen = x.shape[1]
    freqs_cis = freqs_cis[0:seqlen]
    assert freqs_cis.shape == (seqlen, x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        torch.unsqueeze(x, dim=3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class LaneLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_lanes: int):
        super().__init__()
        self.num_lanes = num_lanes
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.empty(self.get_shape())
        self.reset_parameters()

    def reset_parameters(self):
        raise NotImplementedError

    def get_shape(self) -> Tuple[int, int, int]:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum(
            "bsli,loi->bslo", x, self.weight
        )  # [L, out_features, 1] @ [B*S, L, in_features // L, 1] -> [B*S, L, out_features, 1]


class LaneColumnLinear(LaneLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_lanes: int,
    ):
        super().__init__(
            in_features=in_features, out_features=out_features, num_lanes=num_lanes
        )

    def get_shape(self) -> Tuple[int, int, int]:
        return (self.num_lanes, self.out_features, self.in_features // self.num_lanes)

    def reset_parameters(self):
        w = torch.empty(self.out_features, self.in_features)
        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        self.weight = nn.Parameter(
            w.reshape(
                self.out_features, self.num_lanes, self.in_features // self.num_lanes
            )
            .permute(1, 0, 2)
            .contiguous()
        )


class LaneRowLinear(LaneLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_lanes: int,
    ):
        super().__init__(
            in_features=in_features, out_features=out_features, num_lanes=num_lanes
        )

    def get_shape(self) -> Tuple[int, int, int]:
        return (self.num_lanes, self.out_features // self.num_lanes, self.in_features)

    def reset_parameters(self):
        w = torch.empty(self.out_features, self.in_features)
        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        self.weight = nn.Parameter(w.reshape(self.weight.shape).contiguous())


class LaneLMHead(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_lanes: int):
        super().__init__()
        self.num_lanes = num_lanes
        self.linear = nn.Linear(
            in_features=num_lanes * in_features, out_features=out_features, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, L, d = x.shape
        return self.linear(x.view(B, S, L * d))


class LaneRMSNorm(nn.Module):
    def __init__(self, dim: int, num_lanes: int, eps: float = 1e-8):
        super().__init__()
        self.rms_norm = nn.RMSNorm(dim, eps=eps, elementwise_affine=False)
        self.dim = dim
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(num_lanes, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rms_norm(x) * self.scale

    def reset_parameters(self):
        nn.init.ones_(self.scale)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        if attention_mask.dim() == 2:
            attention_mask = attention_mask[:, None, None, :]
        causal_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value)

    return attn_output, attn_weights


class LaneformerAttention(nn.Module):
    def __init__(self, config: LaneformerTPConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = config.num_attention_heads
        self.num_lanes = config.num_lanes
        self.n_kv_heads = (
            config.num_attention_heads
            if config.num_key_value_heads is None
            else config.num_key_value_heads
        )
        self.n_rep = config.num_attention_heads // self.n_kv_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        # Scaling & causal flag for backend attention APIs
        self.scaling = self.head_dim**-0.5
        self.is_causal = True  # decoder-only model
        # Optional attention implementation selection ("eager", "flash_attention_2", "sdpa", etc.)
        self._attn_implementation = getattr(config, "_attn_implementation", "eager")
        self.wq = LaneRowLinear(
            in_features=config.hidden_size,
            out_features=self.n_heads * self.head_dim,
            num_lanes=config.num_lanes,
        )
        self.wk = LaneRowLinear(
            in_features=config.hidden_size,
            out_features=self.n_kv_heads * self.head_dim,
            num_lanes=config.num_lanes,
        )
        self.wv = LaneRowLinear(
            in_features=config.hidden_size,
            out_features=self.n_kv_heads * self.head_dim,
            num_lanes=config.num_lanes,
        )
        self.wo = LaneColumnLinear(
            in_features=self.n_heads * self.head_dim,
            out_features=config.hidden_size,
            num_lanes=config.num_lanes,
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        *,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor (B, S, L, D)
            freqs_cis (torch.Tensor): Precomputed frequency tensor.

        Returns:
            torch.Tensor: Output tensor after attention (B, S, L, D)

        """
        bs, seqlen, _, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.reshape(bs, seqlen, -1, self.head_dim)
        # (B, T, L, n_kv_heads / L, head_dim)
        xk = xk.reshape(bs, seqlen, -1, self.head_dim)
        # (B, T, L, n_kv_heads / L, head_dim)
        xv = xv.reshape(bs, seqlen, -1, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        xq = xq.transpose(1, 2)  # (B, n_heads, T, head_dim)
        xk = xk.transpose(1, 2)  # (B, n_kv_heads, T, head_dim)
        xv = xv.transpose(1, 2)  # (B, n_kv_heads, T, head_dim)

        # Cache update BEFORE replication
        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": None, "cos": None, "cache_position": cache_position}
            xk, xv = past_key_values.update(xk, xv, self.layer_idx, cache_kwargs)

        # Repeat K/V then perform existing flex/SDPA path
        # (B, n_heads, T, head_dim)
        xk = repeat_kv(xk.transpose(1, 2), self.n_rep).transpose(1, 2)
        # (B, n_heads, T, head_dim)
        xv = repeat_kv(xv.transpose(1, 2), self.n_rep).transpose(1, 2)
        # (B, n_heads, T, head_dim)
        output, _ = eager_attention_forward(
            self,
            query=xq,
            key=xk,
            value=xv,
            attention_mask=attention_mask,
            scaling=self.scaling,
            dropout=0.0,
        )

        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(
                bs,
                seqlen,
                self.num_lanes,
                self.n_heads * self.head_dim // self.num_lanes,
            )
        )
        output = self.wo(output)
        return output


class LaneformerMLP(nn.Module):
    def __init__(self, config: LaneformerTPConfig):
        super().__init__()
        ffn_dim = config.intermediate_size
        dim = config.hidden_size
        num_lanes = config.num_lanes

        self.w1 = LaneRowLinear(dim, ffn_dim, num_lanes=num_lanes)
        self.w2 = LaneColumnLinear(ffn_dim, dim, num_lanes=num_lanes)
        self.w3 = LaneRowLinear(dim, ffn_dim, num_lanes=num_lanes)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ReduceMode(Enum):
    NO_REDUCE = "no_reduce"
    PRESENT = "present"
    PAST = "past"


class LaneformerDecoderLayer(nn.Module):
    def __init__(
        self, config: LaneformerTPConfig, layer_idx: int, reduce_mode: ReduceMode
    ):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.dim = config.hidden_size
        self.attention = LaneformerAttention(config, layer_idx=layer_idx)
        self.feed_forward = LaneformerMLP(config)
        norm_fun = (
            nn.RMSNorm
            if config.replicated_rmsn_scale
            else partial(LaneRMSNorm, num_lanes=config.num_lanes)
        )
        self.attention_norm = norm_fun(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = norm_fun(config.hidden_size, eps=config.norm_eps)
        self.reduce_mode = reduce_mode
        self.num_lanes = config.num_lanes

    def reduce_lanes(
        self, x: torch.Tensor, past: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        x: (B, S, L, D)
        past: (B, S, L, D)
        """
        if self.reduce_mode == ReduceMode.NO_REDUCE:
            return x * math.sqrt(self.num_lanes)
        elif self.reduce_mode == ReduceMode.PRESENT:
            return torch.sum(x, dim=-2, keepdim=True)
        elif self.reduce_mode == ReduceMode.PAST:
            assert past is not None, "Past must be provided if reduce_mode is PAST"
            sum_past = torch.sum(past, dim=-2, keepdim=True) - past
            return x + sum_past
        else:
            raise ValueError(f"Unknown reduce mode: {self.reduce_mode}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        past_attention: Optional[torch.Tensor] = None,
        past_mlp: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = self.attention_norm(hidden_states)
        hidden_states = self.attention(
            x=hidden_states,
            freqs_cis=freqs_cis,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        future_attention = hidden_states
        hidden_states = self.reduce_lanes(hidden_states, past=past_attention)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        future_mlp = hidden_states
        hidden_states = self.reduce_lanes(hidden_states, past=past_mlp)
        hidden_states = residual + hidden_states
        return hidden_states, future_attention, future_mlp


# ============================================================
# Backbone & HF wrappers
# ============================================================
class LaneformerPreTrainedModel(PreTrainedModel):
    config_class = LaneformerTPConfig
    config: LaneformerTPConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_gradient_checkpointing = True

    # Advertise support for Transformers attention backends (vLLM / flash / sdpa routing)
    _supports_attention_backend = True
    # Optionally declare flash attention support if environment provides it; safe to set True
    _supports_flash_attn = True


class LaneformerTPModel(LaneformerPreTrainedModel):
    def __init__(self, config: LaneformerTPConfig):
        super().__init__(config)
        self.gradient_checkpointing = False
        self._gradient_checkpointing_kwargs = {}

        self.padding_idx = config.pad_token_id
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layers = config.num_hidden_layers
        self.eos_id = config.eos_token_id

        self.num_lanes = config.num_lanes
        self.broadcast_delay = config.broadcast_delay
        self.use_comm = config.use_comm
        self.use_early_comm = config.use_early_comm
        self.pre_norm_lane_agg = config.pre_norm_lane_agg
        self.lm_head_type = config.lm_head_type
        self.replicated_rmsn_scale = config.replicated_rmsn_scale

        self.tok_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )

        # Rotary buffer
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                config.hidden_size // config.num_attention_heads,
                config.max_position_embeddings,
                config.rope_theta,
            ),
            persistent=False,
        )

        # Build lane blocks
        self.layers = nn.ModuleList()
        if not (self.use_comm):
            for layer_id in range(config.num_hidden_layers):
                if layer_id < self.broadcast_delay:
                    reduce_mode = (
                        ReduceMode.PRESENT
                        if self.use_early_comm
                        else ReduceMode.NO_REDUCE
                    )
                else:
                    reduce_mode = ReduceMode.NO_REDUCE
                self.layers.append(
                    LaneformerDecoderLayer(
                        config=config, layer_idx=layer_id, reduce_mode=reduce_mode
                    )
                )
        else:
            for layer_id in range(config.num_hidden_layers):
                if layer_id < self.broadcast_delay:
                    reduce_mode = (
                        ReduceMode.PRESENT
                        if self.use_early_comm
                        else ReduceMode.NO_REDUCE
                    )
                else:
                    reduce_mode = ReduceMode.PAST

                self.layers.append(
                    LaneformerDecoderLayer(
                        config=config, layer_idx=layer_id, reduce_mode=reduce_mode
                    )
                )
        if self.pre_norm_lane_agg and not self.lm_head_type == "replicate":
            raise ValueError(
                "pre_norm_lane_agg is only supported with lm_head_type='replicate'"
            )

        norm_fun = (
            nn.RMSNorm
            if config.replicated_rmsn_scale or config.pre_norm_lane_agg
            else partial(LaneRMSNorm, num_lanes=self.num_lanes)
        )

        self.norm = norm_fun(config.hidden_size, eps=config.norm_eps)

    # Required by vllm integration
    def get_input_embeddings(self):
        return self.tok_embeddings

    # Required by vllm integration
    def set_input_embeddings(self, new_emb):
        self.tok_embeddings = new_emb

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.tok_embeddings(input_ids)

        if use_cache is None:
            use_cache = False

        if getattr(self, "gradient_checkpointing", False) and self.training:
            use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        causal_masks = []
        for layer_id in range(self.config.num_hidden_layers):
            unclamped_layers = getattr(self.config, 'unclamped_layers', [])
            # Use full attention if layer is in the unclamped list
            if layer_id in unclamped_layers:
                causal_mask = create_causal_mask(
                    config=self.config,
                    input_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    cache_position=cache_position,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                )
            elif (
                layer_id < self.config.sliding_window_n_layers
                or layer_id
                >= self.config.num_hidden_layers + self.config.sliding_window_n_layers
            ):
                causal_mask = create_sliding_window_causal_mask(
                    config=self.config,
                    input_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    cache_position=cache_position,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                )
            else:
                causal_mask = create_causal_mask(
                    config=self.config,
                    input_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    cache_position=cache_position,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                )
            causal_masks.append(causal_mask)
        freqs_cis = self.freqs_cis[position_ids].unsqueeze(-2)

        hidden_states = inputs_embeds
        hidden_states = hidden_states[:, :, None, :].expand(-1, -1, self.num_lanes, -1)
        past_attentions = []
        past_mlps = []
        for i in range(self.config.num_hidden_layers):
            if i < self.broadcast_delay:
                past_attention = None
                past_mlp = None
            else:
                past_attention = past_attentions[i - self.broadcast_delay]
                past_mlp = past_mlps[i - self.broadcast_delay]
            # hidden_states, future_attention, future_mlp = self.layers[i](
            #     hidden_states=hidden_states,
            #     attention_mask=causal_masks[i],
            #     position_ids=position_ids,
            #     past_key_values=past_key_values,
            #     past_attention=past_attention,
            #     past_mlp=past_mlp,
            #     use_cache=use_cache,
            #     cache_position=cache_position,
            #     freqs_cis=freqs_cis,
            #     **kwargs,
            # )

            layer = self.layers[i]

            # Determine past_* for your lane-comm logic (same as before)
            if i < self.broadcast_delay:
                past_attention = None
                past_mlp = None
            else:
                past_attention = past_attentions[i - self.broadcast_delay]
                past_mlp = past_mlps[i - self.broadcast_delay]

            # ✅ checkpoint only during training when gradient_checkpointing is enabled
            if getattr(self, "gradient_checkpointing", False) and self.training:
                # IMPORTANT: checkpointed functions must take Tensor inputs;
                # we capture non-tensor args via closure, and we also ensure past_key_values is None.
                def custom_forward(hs):
                    out_hs, fut_attn, fut_mlp = layer(
                        hidden_states=hs,
                        attention_mask=causal_masks[i],
                        position_ids=position_ids,
                        past_key_values=None,          # ✅ no cache during GC
                        past_attention=past_attention,  # captured
                        past_mlp=past_mlp,              # captured
                        use_cache=False,
                        cache_position=cache_position,
                        freqs_cis=freqs_cis,
                        **kwargs,
                    )
                    return out_hs, fut_attn, fut_mlp

                # If you care about use_reentrant, you can hardcode False (common on torch 2.x)
                hidden_states, future_attention, future_mlp = checkpoint(
                    custom_forward,
                    hidden_states,
                    use_reentrant=False,
                )
            else:
                hidden_states, future_attention, future_mlp = layer(
                    hidden_states=hidden_states,
                    attention_mask=causal_masks[i],
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    past_attention=past_attention,
                    past_mlp=past_mlp,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    freqs_cis=freqs_cis,
                    **kwargs,
                )

            past_attentions.append(future_attention)
            past_mlps.append(future_mlp)

        if self.pre_norm_lane_agg:
            # (B, S, L, D) -> (B, S, D)
            hidden_states = hidden_states.sum(dim=-2)
            hidden_states = self.norm(hidden_states)
        elif self.lm_head_type == "replicate":  # and naturally not pre_lane_agg
            # (B, S, L, D) -> (B, S, D)
            hidden_states = self.norm(hidden_states)
            hidden_states = hidden_states.mean(dim=-2)
        else:
            # (B, S, L, D)
            hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring
class LaneformerTPForCausalLM(LaneformerPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: LaneformerTPConfig):
        super().__init__(config)
        self.gradient_checkpointing = False
        self._gradient_checkpointing_kwargs = {}
        self.model = LaneformerTPModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head_type = config.lm_head_type
        if self.lm_head_type == "replicate":
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        elif self.lm_head_type == "lane":
            self.lm_head = LaneLMHead(
                in_features=config.hidden_size,
                out_features=config.vocab_size,
                num_lanes=config.num_lanes,
            )
        elif self.lm_head_type == "vocab_parallel":
            self.lm_head = LaneRowLinear(
                in_features=config.hidden_size,
                out_features=config.vocab_size,
                num_lanes=config.num_lanes,
            )

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        if self.lm_head_type == "replicate":
            logits = self.lm_head(hidden_states[:, slice_indices, :])
        else:
            logits = self.lm_head(hidden_states[:, slice_indices, :, :])
            if self.lm_head_type == "vocab_parallel":
                # (B, S, L, D) -> (B, S, D)
                B, S, L, D = logits.shape
                logits = logits.reshape(B, S, L * D)

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
