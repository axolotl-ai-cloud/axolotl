"""Ricursive (Ripple) modeling, adapted for training.

Source: https://huggingface.co/ricursiveAI/Ripple-65B-Preview/blob/main/modeling_ricursive.py
Revision: e27e171

The published file targets transformers 4.57 inference: it hardcodes SDPA with
sliding-window masks built from absolute positions (packed sequences attend
across documents), cannot gradient checkpoint, and computes its own loss that
ignores ``num_items_in_batch``. Attention, masking, caching and the loss now go
through transformers; the Ricursive block numerics are unchanged.
"""

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import initialization as init
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import can_return_tuple
from transformers.utils.generic import TransformersKwargs, merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from .configuration_ricursive import RicursiveConfig


class RicursiveRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


class RicursiveRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        # recomputed per call: from_pretrained does not restore non-persistent buffers
        inv_freq = 1.0 / (
            self.theta
            ** (torch.arange(0, self.dim, 2, device=x.device).float() / self.dim)
        )
        inv_freq_expanded = (
            inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        with torch.autocast(device_type=x.device.type, enabled=False):
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1).float()
    sin = sin.unsqueeze(1).float()
    orig_dtype = q.dtype
    q_float, k_float = q.float(), k.float()
    q_embed = (q_float * cos) + (rotate_half(q_float) * sin)
    k_embed = (k_float * cos) + (rotate_half(k_float) * sin)
    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class RicursiveAttention(nn.Module):
    """Gated GQA attention: sliding layers use RoPE, full layers use none (NoPE)."""

    def __init__(self, config: RicursiveConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.is_causal = True
        self.sliding_window = (
            config.sliding_window
            if config.layer_types[layer_idx] == "sliding_attention"
            else None
        )

        hidden_size = config.hidden_size
        self.q_proj = nn.Linear(
            hidden_size, config.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            hidden_size, config.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            hidden_size, config.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, hidden_size, bias=False
        )
        if config.use_valres:
            self.value_res_lambda = nn.Parameter(torch.zeros(1))

        self.q_norm = RicursiveRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RicursiveRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        if config.use_gated_attention:
            self.g_proj = nn.Linear(
                hidden_size, config.num_attention_heads * self.head_dim, bias=False
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        value_residual: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        value_input = hidden_states
        if value_residual is not None:
            blend = torch.sigmoid(self.value_res_lambda[0]).to(
                dtype=hidden_states.dtype
            )
            value_input = torch.lerp(
                value_residual.to(dtype=hidden_states.dtype), hidden_states, blend
            )

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
        value_states = self.v_proj(value_input).view(hidden_shape)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()

        if self.config.use_gated_attention:
            gate = self.g_proj(hidden_states)
            if self.config.gate_full_precision:
                gate = gate.float()
            attn_output = attn_output * torch.sigmoid(gate).to(attn_output.dtype)

        return self.o_proj(attn_output), attn_weights


class RicursiveMLP(nn.Module):
    def __init__(self, config: RicursiveConfig):
        super().__init__()
        self.clamp_threshold = config.ffn_clamp_threshold
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if self.clamp_threshold:
            gate = gate.clamp(min=None, max=self.clamp_threshold)
            up = up.clamp(min=-self.clamp_threshold, max=self.clamp_threshold)
        return self.down_proj(F.silu(gate) * up)


class RicursiveDecoderLayer(GradientCheckpointingLayer):
    """PeriNorm block: each sublayer is pre- and post-normed before the residual add."""

    def __init__(self, config: RicursiveConfig, layer_idx: int):
        super().__init__()
        self.self_attn = RicursiveAttention(config, layer_idx)
        self.mlp = RicursiveMLP(config)
        self.input_layernorm = RicursiveRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = RicursiveRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = RicursiveRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = RicursiveRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        value_residual: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        attn_output, _ = self.self_attn(
            self.input_layernorm(hidden_states),
            value_residual,
            position_embeddings,
            attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = hidden_states + self.post_attention_layernorm(attn_output)

        mlp_output = self.mlp(self.pre_feedforward_layernorm(hidden_states))
        return hidden_states + self.post_feedforward_layernorm(mlp_output)


class RicursivePreTrainedModel(PreTrainedModel):
    config_class = RicursiveConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["RicursiveDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": RicursiveDecoderLayer,
        "attentions": RicursiveAttention,
    }

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, RicursiveAttention) and hasattr(
            module, "value_res_lambda"
        ):
            init.zeros_(module.value_res_lambda)


class RicursiveModel(RicursivePreTrainedModel):
    def __init__(self, config: RicursiveConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [RicursiveDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RicursiveRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RicursiveRotaryEmbedding(
            config.head_dim, theta=config.rope_theta
        )
        self.gradient_checkpointing = False
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # unset for checkpoints with the scale folded into embed_tokens
        if self.config.embed_scale is not None:
            inputs_embeds = inputs_embeds * self.config.embed_scale

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            position_ids = (
                torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
                + past_seen_tokens
            )
            position_ids = position_ids.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds
        # the value residual is seeded from the scaled embeddings
        value_residual = inputs_embeds if self.config.use_valres else None
        rope = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer, layer_type in zip(
            self.layers, self.config.layer_types, strict=True
        ):
            # value_residual is positional so reentrant checkpointing still
            # propagates its gradient
            hidden_states = decoder_layer(
                hidden_states,
                value_residual,
                attention_mask=causal_mask_mapping[layer_type],
                position_embeddings=rope if layer_type == "sliding_attention" else None,
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class RicursiveForCausalLM(RicursivePreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: RicursiveConfig):
        super().__init__(config)
        self.model = RicursiveModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

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


__all__ = ["RicursiveForCausalLM", "RicursiveModel", "RicursivePreTrainedModel"]
