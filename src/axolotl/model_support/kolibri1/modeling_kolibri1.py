"""Kolibri-1 (Aleph Alpha) for transformers.

Ported from the ``aleph-alpha-inference`` vLLM plugin, the only published
implementation. It is Qwen3-MoE with sliding-window attention on 4 of every 5
layers, NoPE full-attention layers, sigmoid routing with a selection bias, an
ungated shared expert, and sandwich norms.
"""

import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn
from transformers import initialization as init
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import MoeModelOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeExperts,
    Qwen3MoeForCausalLM,
    Qwen3MoeMLP,
    Qwen3MoePreTrainedModel,
    Qwen3MoeRMSNorm,
    Qwen3MoeRotaryEmbedding,
    Qwen3MoeTopKRouter,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import OutputRecorder, capture_outputs


@strict
class Kolibri1Config(Qwen3MoeConfig):
    model_type = "kolibri1"

    head_dim: int = 128
    shared_expert_intermediate_size: int = 512
    use_sliding_window: bool = True
    sliding_window: int | None = 513
    layer_types: list[str] | None = None
    router_aux_loss_coef: float = 0.0

    def __post_init__(self, **kwargs):
        if self.output_router_logits or self.router_aux_loss_coef:
            raise ValueError(
                "Kolibri-1 balances experts with a routing bias; the inherited softmax "
                "load-balancing loss does not apply, so output_router_logits and "
                "router_aux_loss_coef are unsupported."
            )
        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if (i + 1) % 5 == 0 else "sliding_attention"
                for i in range(self.num_hidden_layers)
            ]
        super().__post_init__(**kwargs)


class Kolibri1Attention(Qwen3MoeAttention):
    def __init__(self, config: Kolibri1Config, layer_idx: int):
        super().__init__(config, layer_idx)
        # full-attention layers carry no positional encoding (NoPE)
        layer_type = config.layer_types[layer_idx]  # type: ignore[index]
        self.use_rope = layer_type == "sliding_attention"
        self.sliding_window = config.sliding_window if self.use_rope else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # fmt: off
        # kept verbatim so the LoRA QKV kernel source rewrite matches
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        # fmt: on

        if self.use_rope:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Kolibri1TopKRouter(Qwen3MoeTopKRouter):
    def __init__(self, config: Kolibri1Config):
        super().__init__(config)
        self.register_buffer("e_score_correction_bias", torch.zeros(self.num_experts))

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states.float(), self.weight.float())
        # the bias only steers selection; weights use the unbiased sigmoid
        router_indices = torch.topk(
            router_logits + self.e_score_correction_bias.float(), self.top_k, dim=-1
        ).indices
        router_scores = torch.sigmoid(router_logits.gather(1, router_indices))
        if self.norm_topk_prob:
            router_scores = router_scores / (
                router_scores.sum(dim=-1, keepdim=True) + 1e-20
            )
        return router_logits, router_scores.to(input_dtype), router_indices


class Kolibri1SparseMoeBlock(nn.Module):
    def __init__(self, config: Kolibri1Config):
        super().__init__()
        self.experts = Qwen3MoeExperts(config)
        self.gate = Kolibri1TopKRouter(config)
        self.shared_experts = Qwen3MoeMLP(
            config, intermediate_size=config.shared_expert_intermediate_size
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        _, routing_weights, selected_experts = self.gate(hidden_states)
        final_hidden_states = self.experts(
            hidden_states, selected_experts, routing_weights
        ) + self.shared_experts(hidden_states)
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)


class Kolibri1DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Kolibri1Config, layer_idx: int):
        super().__init__()
        self.self_attn = Kolibri1Attention(config, layer_idx)
        self.mlp = Kolibri1SparseMoeBlock(config)
        self.input_layernorm = Qwen3MoeRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attn_norm = Qwen3MoeRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3MoeRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.post_ffn_norm = Qwen3MoeRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states, _ = self.self_attn(
            hidden_states=self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + self.post_attn_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + self.post_ffn_norm(hidden_states)


class Kolibri1PreTrainedModel(Qwen3MoePreTrainedModel):
    config_class = Kolibri1Config
    _no_split_modules = ["Kolibri1DecoderLayer"]
    _can_record_outputs = {
        "router_logits": OutputRecorder(Kolibri1TopKRouter, index=0),
        "hidden_states": Kolibri1DecoderLayer,
        "attentions": Kolibri1Attention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Kolibri1TopKRouter):
            init.zeros_(module.e_score_correction_bias)

    @classmethod
    def _can_set_experts_implementation(cls) -> bool:
        # transformers greps the module source for @use_experts_implementation,
        # which Qwen3MoeExperts carries but this file does not spell
        return True


class Kolibri1Model(Kolibri1PreTrainedModel):
    def __init__(self, config: Kolibri1Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [Kolibri1DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3MoeRotaryEmbedding(config=config)
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
        **kwargs,
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            position_ids = (
                torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
                + past_seen_tokens
            ).unsqueeze(0)

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
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        return MoeModelOutputWithPast(
            last_hidden_state=self.norm(hidden_states),
            past_key_values=past_key_values,
        )


class Kolibri1ForCausalLM(Kolibri1PreTrainedModel, Qwen3MoeForCausalLM):
    def __init__(self, config: Kolibri1Config):
        # skip Qwen3MoeForCausalLM.__init__, which would build a Qwen3MoeModel
        PreTrainedModel.__init__(self, config)
        self.model = Kolibri1Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.post_init()
