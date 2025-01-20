import logging
from typing import Tuple, Optional, Unpack, Callable, Union

import torch
from torch import nn
from transformers import LlamaConfig, Cache, DynamicCache
from transformers.activations import ACT2FN
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, eager_attention_forward, LlamaRMSNorm, \
    LlamaForCausalLM, LlamaModel, LlamaRotaryEmbedding

from axolotl.integrations.rrt.modeling.linear import RelaxedRecursiveDoraLinear

logger = logging.getLogger(__name__)

class RelaxedRecursiveLlamaConfig(LlamaConfig):
    """
    Configuration for Relaxed Recursive Llama.
    """

    model_type = "llama-rrt"
    recurse_layers: int  = 4
    rank: int
    alpha: int
    use_dora: bool = True


class RelaxedRecursiveLlamaMLP(nn.Module):
    def __init__(self, config: RelaxedRecursiveLlamaConfig):
        super().__init__()
        recurse_loops = config.num_hidden_layers // config.recurse_layers
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = RelaxedRecursiveDoraLinear(self.hidden_size, self.intermediate_size, recurse_loops, config.rank, config.alpha, bias=config.mlp_bias, use_dora=config.use_dora)
        self.up_proj = RelaxedRecursiveDoraLinear(self.hidden_size, self.intermediate_size, recurse_loops, config.rank, config.alpha, bias=config.mlp_bias, use_dora=config.use_dora)
        self.down_proj = RelaxedRecursiveDoraLinear(self.intermediate_size, self.hidden_size, recurse_loops, config.rank, config.alpha, bias=config.mlp_bias, use_dora=config.use_dora)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x, loop_idx: int):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x, loop_idx)) * self.up_proj(x, loop_idx), loop_idx)
        return down_proj


class RelaxedRecursiveLlamaAttention(nn.Module):
    """
    A single attention layer of the Relaxed Recursive Llama.
    """

    def __init__(self, config: RelaxedRecursiveLlamaConfig, layer_idx: int):
        super().__init__()
        recurse_loops = config.num_hidden_layers // config.recurse_layers
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = RelaxedRecursiveDoraLinear(
            config.hidden_size, config.num_attention_heads * self.head_dim, recurse_loops, config.rank, config.alpha, bias=config.attention_bias, use_dora=config.use_dora
        )
        self.k_proj = RelaxedRecursiveDoraLinear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, recurse_loops, config.rank, config.alpha, bias=config.attention_bias, use_dora=config.use_dora
        )
        self.v_proj = RelaxedRecursiveDoraLinear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, recurse_loops, config.rank, config.alpha, bias=config.attention_bias, use_dora=config.use_dora
        )
        self.o_proj = RelaxedRecursiveDoraLinear(
            config.num_attention_heads * self.head_dim, config.hidden_size, recurse_loops, config.rank, config.alpha, bias=config.attention_bias, use_dora=config.use_dora
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        loop_idx: int,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states, loop_idx).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states, loop_idx).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states, loop_idx).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output, loop_idx)
        return attn_output, attn_weights



class RelaxedRecursiveLlamaDecoderLayer(nn.Module):
    """
    A single layer of the Relaxed Recursive Llama decoder.
    """

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        recurse_loops = config.num_hidden_layers // config.recurse_layers
        self.hidden_size = config.hidden_size

        self.self_attn = RelaxedRecursiveLlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = RelaxedRecursiveLlamaMLP(config)

        self.input_layernorm_list = nn.ModuleList([LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps) for _ in range(recurse_loops)])
        self.post_attention_layernorm_list = nn.ModuleList([LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps) for _ in range(recurse_loops)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        loop_idx: int,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm_list[loop_idx](hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            loop_idx=loop_idx,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm_list[loop_idx](hidden_states)
        hidden_states = self.mlp(hidden_states, loop_idx)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class RelaxedRecursiveLlamaModel(LlamaModel):
    config_class = RelaxedRecursiveLlamaConfig

    def __init__(self, config):
        super(LlamaModel, self).__init__(config)
        self.recurse_loops = config.num_hidden_layers // config.recurse_layers
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [RelaxedRecursiveLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.recurse_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for loop_idx in range(self.recurse_loops):
            for decoder_layer in self.layers[: self.config.recurse_layers]:
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        loop_idx,
                        causal_mask,
                        position_ids,
                        past_key_values,
                        output_attentions,
                        use_cache,
                        cache_position,
                        position_embeddings,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        loop_idx,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        **flash_attn_kwargs,
                    )

                hidden_states = layer_outputs[0]

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()


class RelaxedRecursiveLlamaForCausalLM(LlamaForCausalLM):
    config_class = RelaxedRecursiveLlamaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = RelaxedRecursiveLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_nb_trainable_parameters(self) -> tuple[int, int, int]:
        r"""
        Returns the number of trainable parameters and the number of all parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        lora_params = 0
        for name, param in self.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            # Due to the design of 4bit linear layers from bitsandbytes
            # one needs to multiply the number of parameters by 2 to get
            # the correct number of parameters
            if param.__class__.__name__ == "Params4bit":
                if hasattr(param, "element_size"):
                    num_bytes = param.element_size()
                elif not hasattr(param, "quant_storage"):
                    num_bytes = 1
                else:
                    num_bytes = param.quant_storage.itemsize
                num_params = num_params * 2 * num_bytes

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
            if "lora_" in name:
                lora_params += num_params

        return trainable_params, all_param, lora_params
