"""Modeling for differential transformers."""

import math
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaMLP,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
)

from .diff_attn import (
    LlamaDifferentialAttention,
    LlamaDifferentialAttentionBase,
    LlamaDifferentialFlashAttention2,
    LlamaDifferentialSdpaAttention,
)


class LlamaDifferentialConfig(LlamaConfig):
    """Configuration class for Differential LLaMA model."""

    def __init__(
        self,
        split_heads: bool = False,
        sublayer_norm: bool = True,
        zero_init: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.split_heads = split_heads
        self.sublayer_norm = sublayer_norm
        self.zero_init = zero_init
        self.architectures = ["LlamaDifferentialModel"]
        self._attn_implementations = {
            "eager": "differential_eager",
            "sdpa": "differential_sdpa",
            "flash_attention_2": "differential_flash_attention_2",
        }


class LlamaDifferentialPreTrainedModel(LlamaPreTrainedModel):
    """Base class for differential LLaMA models."""

    config_class = LlamaDifferentialConfig
    base_model_prefix = "llama_differential"

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (LlamaDifferentialAttentionBase, LlamaModel)):
            module.gradient_checkpointing = value


def lambda_init_fn(depth: int) -> float:
    """Initialize lambda parameter based on layer depth."""
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class LlamaDifferentialModel(LlamaDifferentialPreTrainedModel):
    """Differential version of the LLaMA model."""

    def __init__(self, config: LlamaDifferentialConfig):
        super().__init__(config)
        # Map attn implementations to classes
        self.attn_implementation_to_class = {
            "differential_eager": LlamaDifferentialAttention,
            "differential_sdpa": LlamaDifferentialSdpaAttention,
            "differential_flash_attention_2": LlamaDifferentialFlashAttention2,
        }

        # Get correct attention implementation
        attn_implementation = getattr(config, "_attn_implementation", "eager")
        if attn_implementation in config._attn_implementations:
            attn_implementation = config._attn_implementations[attn_implementation]

        self.attention_class = self.attn_implementation_to_class.get(
            attn_implementation, LlamaDifferentialAttention
        )

        # Initialize model components
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [
                LlamaDifferentialDecoderLayer(
                    config=config, layer_idx=i, attention_class=self.attention_class
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Check if either input_ids or inputs_embeds is provided
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
            device = input_ids.device
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
            device = inputs_embeds.device
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        # Initialize past_key_values if needed
        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))

        # Create attention mask if not provided
        if attention_mask is not None:
            attention_mask = self._prepare_attention_mask(
                attention_mask, (batch_size, seq_length), device
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        # Initialize lists to store outputs
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_cache = () if use_cache else None

        for _, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)  # type: ignore

            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_cache += (layer_outputs[-1],)  # type: ignore

            if output_attentions:
                all_self_attns += (layer_outputs[1],)  # type: ignore

        # Add last hidden state
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)  # type: ignore

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _prepare_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Tuple[int, int],
        device: torch.device,
    ) -> torch.Tensor:
        """Prepare attention mask for computing attention."""
        # Create causal mask
        # [batch_size, seq_length] -> [batch_size, 1, seq_length, seq_length]
        combined_attention_mask = None
        _, seq_length = input_shape

        if self.config.is_decoder:
            seq_ids = torch.arange(seq_length, device=device)
            causal_mask = (
                seq_ids[None, None, :].repeat(1, seq_length, 1)
                <= seq_ids[None, :, None]
            )
            causal_mask = causal_mask.to(attention_mask.dtype)

            if causal_mask.shape[1:] != (seq_length, seq_length):
                causal_mask = causal_mask[:, :seq_length, :seq_length]

            # Extend attention mask
            combined_attention_mask = (
                causal_mask[None, None, :, :] * attention_mask[:, None, None, :]
            )
        else:
            combined_attention_mask = attention_mask[:, None, None, :]

        return combined_attention_mask

    @classmethod
    def from_llama(
        cls,
        llama_model: LlamaModel,
        differential_config: Optional[LlamaDifferentialConfig] = None,
    ) -> "LlamaDifferentialModel":
        """Convert a standard LLaMA model to use differential attention."""
        if differential_config is None:
            # pylint: disable=protected-access
            differential_config = LlamaDifferentialConfig.from_pretrained(
                llama_model.config._name_or_path
            )

        # Create new model
        new_model = cls(differential_config)

        # Copy non-attention weights directly
        new_model.embed_tokens.load_state_dict(llama_model.embed_tokens.state_dict())
        new_model.norm.load_state_dict(llama_model.norm.state_dict())

        # Copy layer weights, handling attention layers specially
        for new_layer, old_layer in zip(new_model.layers, llama_model.layers):
            # Copy self-attention weights with special handling
            if differential_config.split_heads:
                # Split heads mode
                new_layer.self_attn.q_proj.weight.data.copy_(
                    old_layer.self_attn.q_proj.weight.data
                )
                new_layer.self_attn.k_proj.weight.data.copy_(
                    old_layer.self_attn.k_proj.weight.data
                )
            else:
                # Double projection mode - copy weights to positive components
                new_layer.self_attn.q_proj.weight.data[
                    : differential_config.hidden_size
                ].copy_(old_layer.self_attn.q_proj.weight.data)
                new_layer.self_attn.k_proj.weight.data[
                    : differential_config.hidden_size
                ].copy_(old_layer.self_attn.k_proj.weight.data)

                # Zero out relevant parameters for exact equivalence
                if differential_config.zero_init:
                    old_kv_size = old_layer.self_attn.k_proj.weight.data.size(0)
                    new_layer.self_attn.q_proj.weight.data[
                        new_layer.self_attn.hidden_size :
                    ] = 0
                    new_layer.self_attn.k_proj.weight.data[old_kv_size:] = 0
                    nn.init.zeros_(new_layer.self_attn.lambda_q1)
                    nn.init.zeros_(new_layer.self_attn.lambda_k1)
                    nn.init.zeros_(new_layer.self_attn.lambda_q2)
                    nn.init.zeros_(new_layer.self_attn.lambda_k2)
                    nn.init.zeros_(new_layer.self_attn.lambda_init)

            # Copy remaining weights
            new_layer.self_attn.v_proj.load_state_dict(
                old_layer.self_attn.v_proj.state_dict()
            )
            new_layer.self_attn.o_proj.load_state_dict(
                old_layer.self_attn.o_proj.state_dict()
            )

            # Copy MLP and layer norm weights
            new_layer.mlp.load_state_dict(old_layer.mlp.state_dict())
            new_layer.input_layernorm.load_state_dict(
                old_layer.input_layernorm.state_dict()
            )
            new_layer.post_attention_layernorm.load_state_dict(
                old_layer.post_attention_layernorm.state_dict()
            )

        return new_model


class LlamaDifferentialDecoderLayer(nn.Module):
    """Custom decoder layer for diffrential Llama model."""

    def __init__(
        self, config: LlamaDifferentialConfig, layer_idx: int, attention_class
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = attention_class(config, layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Layer forward pass with differential attention.
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)  # type: ignore

        if use_cache:
            outputs += (present_key_value,)  # type: ignore

        return outputs  # type: ignore
