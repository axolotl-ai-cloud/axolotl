"""Flash attention monkey patch for llama model"""

# copied from https://github.com/lm-sys/FastChat/blob/main/fastchat/train/llama_flash_attn_monkey_patch.py

import logging
from functools import partial
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import transformers
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer as OriginalLlamaDecoderLayer,
)
from transformers.models.llama.modeling_llama import (
    LlamaMLP,
)

from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
)

from axolotl.monkeypatch.utils import get_cu_seqlens_from_pos_ids, set_module_name
from axolotl.monkeypatch.fused_module import FusedAttention, FusedMLP
from axolotl.monkeypatch.flash_module import flashattn_forward


LOG = logging.getLogger("axolotl")


def replace_llama_mlp_with_swiglu(model):
    for name, module in model.named_modules():
        if isinstance(module, LlamaMLP):
            mlp = FusedMLP(
                module.config, module.gate_proj, module.up_proj, module.down_proj
            )
            set_module_name(model, name, mlp)


def replace_llama_qkv_with_fused(model):
    for name, module in model.named_modules():
        if isinstance(module, LlamaAttention):
            qkv = FusedAttention(
                module.config,
                module.q_proj,
                module.k_proj,
                module.v_proj,
                module.o_proj,
            )
            set_module_name(model, name, qkv)


def replace_llama_attn_with_flash_attn(
    packed: Optional[bool] = False,
    cross_entropy: Optional[bool] = False,
    rms_norm: Optional[bool] = False,
):
    transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = (  # pylint: disable=protected-access
        _prepare_decoder_attention_mask
    )
    transformers.models.llama.modeling_llama.LlamaAttention.forward = flashattn_forward
    transformers.models.llama.modeling_llama.LlamaAttention.apply_rotary_fn = apply_rotary_pos_emb
    transformers.models.llama.modeling_llama.LlamaAttention.repeat_kv_fn = repeat_kv
    if packed:
        transformers.models.llama.modeling_llama.LlamaDecoderLayer = LlamaDecoderLayer
        transformers.models.llama.modeling_llama.LlamaModel.forward = (
            llama_model_forward
        )

    # skip only if explicitly disabled
    if cross_entropy:
        try:
            from flash_attn.losses.cross_entropy import CrossEntropyLoss

            LOG.info("patching with flash_attn.losses.cross_entropy")
            transformers.models.llama.modeling_llama.CrossEntropyLoss = partial(
                CrossEntropyLoss, inplace_backward=True
            )
        except ImportError:
            LOG.info(
                "optimized flash-attention CrossEntropyLoss not found (run `pip install 'git+https://github.com/Dao-AILab/flash-attention.git#egg=xentropy_cuda_lib&subdirectory=csrc/xentropy'`)"
            )

    # skip only if explicitly disabled
    if rms_norm:
        try:
            from flash_attn.ops.rms_norm import RMSNorm

            class LlamaRMSNorm(RMSNorm):
                """Patched LLamaRMSNorm"""

                def __init__(self, hidden_size, eps=1e-6):
                    super().__init__(hidden_size, eps=eps)

            LOG.info("patching with flash_attn.ops.rms_norm")
            transformers.models.llama.modeling_llama.LlamaRMSNorm = LlamaRMSNorm
        except ImportError:
            LOG.info(
                "optimized flash-attention RMSNorm not found (run `pip install 'git+https://github.com/Dao-AILab/flash-attention.git#egg=dropout_layer_norm&subdirectory=csrc/layer_norm'`)"
            )



# Disable the transformation of the attention mask in LlamaModel as the flash attention
# requires the attention mask to be the same as the key_padding_mask
def _prepare_decoder_attention_mask(
    self,
    attention_mask,
    input_shape,
    inputs_embeds,
    past_key_values_length,
):  # pylint: disable=unused-argument
    # [bsz, seq_len]
    return attention_mask



def llama_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
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

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
        )
    if input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError(
            "You have to specify either decoder_input_ids or decoder_inputs_embeds"
        )

    seq_length_with_past = seq_length
    past_key_values_length = 0

    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length

    cu_seqlens = None
    max_seqlen = None
    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=device,
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()
        cu_seqlens, max_seqlen = get_cu_seqlens_from_pos_ids(position_ids)
        cu_seqlens = cu_seqlens.squeeze()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    # embed positions
    if attention_mask is None:
        attention_mask = torch.ones(
            (batch_size, seq_length_with_past),
            dtype=torch.bool,
            device=inputs_embeds.device,
        )
        padding_mask = None
    else:
        if 0 in attention_mask:
            padding_mask = attention_mask
        else:
            padding_mask = None

    attention_mask = (
        self._prepare_decoder_attention_mask(  # pylint: disable=protected-access
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )
    )

    hidden_states = inputs_embeds

    if self.gradient_checkpointing and self.training:
        if use_cache:
            transformers.logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        if self.gradient_checkpointing and self.training:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(
                        *inputs,
                    )

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(decoder_layer),
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                None,
                padding_mask,
                cu_seqlens,
                max_seqlen,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                padding_mask=padding_mask,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
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


class LlamaDecoderLayer(OriginalLlamaDecoderLayer):
    """
    patched version of LlamaDecoderLayer to pass through the precalculated cu_seqlens
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        padding_mask: Optional[torch.LongTensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cu_seqlens (`torch.Tensor`, *optional*) cumulative sequence len when packing
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
            padding_mask=padding_mask,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
