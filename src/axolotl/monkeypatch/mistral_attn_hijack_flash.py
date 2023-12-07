"""Flash attention monkey patch for mistral model"""
# pylint: disable=duplicate-code

import logging
from typing import List, Optional, Tuple, Union

import torch
import transformers
from einops import rearrange
from flash_attn.bert_padding import pad_input, unpad_input
from flash_attn.flash_attn_interface import (  # pylint: disable=ungrouped-imports
    flash_attn_kvpacked_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
)
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.mistral.modeling_mistral import (
    MistralAttention as OriginalMistralAttention,
)
from transformers.models.mistral.modeling_mistral import (
    MistralDecoderLayer as OriginalMistralDecoderLayer,
)
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb, repeat_kv

from axolotl.monkeypatch.utils import get_cu_seqlens_from_pos_ids

from axolotl.monkeypatch.flash_modules import (
    flashattn_forward,
    replace_cross_entropy,
    replace_rms_norm
)

LOG = logging.getLogger("axolotl.monkeypatch.mistral")


def replace_mistral_attn_with_flash_attn(
    packed: Optional[bool] = False,
    cross_entropy: Optional[bool] = False,
    rms_norm: Optional[bool] = False,
):
    transformers.models.mistral.modeling_mistral.MistralModel._prepare_decoder_attention_mask = (  # pylint: disable=protected-access
        _prepare_decoder_attention_mask
    )
    transformers.models.mistral.modeling_mistral.MistralAttention.forward = (
        flashattn_forward
    )
    transformers.models.mistral.modeling_mistral.MistralAttention.apply_rotary_fn = apply_rotary_pos_emb
    transformers.models.mistral.modeling_mistral.MistralAttention.repeat_kv_fn = repeat_kv
    if packed:
        transformers.models.mistral.modeling_mistral.MistralDecoderLayer = (
            MistralDecoderLayer
        )
        transformers.models.mistral.modeling_mistral.MistralModel.forward = (
            mistral_model_forward
        )
    if cross_entropy:
        replace_cross_entropy(transformers.mistral.llama.modeling_mistral, "CrossEntropyLoss")
    if rms_norm:
        replace_rms_norm(transformers.mistral.llama.modeling_mistral, "MistralRMSNorm")


@torch.jit.script
def _make_sliding_window_causal_mask(
    bsz: int,
    tgt_len: int,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
    sliding_window: int = 4096,
):
    """
    Make causal mask used for sliding window attention
    """
    tensor = torch.full(
        (tgt_len, tgt_len),
        fill_value=1,
        device=device,
    )
    mask = torch.tril(tensor, diagonal=0)
    # make the mask banded to account for sliding window
    # NOTE: HF implementation is wrong as of 14-10-2023 for torch.triu, needs +1
    mask = torch.triu(mask, diagonal=-sliding_window + 1)
    mask = torch.log(mask).to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [
                torch.zeros(
                    tgt_len, past_key_values_length, dtype=dtype, device=device
                ),
                mask,
            ],
            dim=-1,
        )
    return mask[None, None, :, :].expand(
        bsz, 1, tgt_len, tgt_len + past_key_values_length
    )


# Disable the transformation of the attention mask in LlamaModel as the flash attention
# requires the attention mask to be the same as the key_padding_mask
def _prepare_decoder_attention_mask(
    self,
    attention_mask,
    input_shape,
    inputs_embeds,
    past_key_values_length,
    sliding_window,
):  # pylint: disable=unused-argument
    # [bsz, seq_len]
    if attention_mask is None:
        return attention_mask

    # NOTE: attention mask and sliding masks are only broadcastable in certain scenarios.
    # Without attention_mask.shape[0] == 1, error will trigger after eval loss but only when wandb is enabled.
    if input_shape[-1] > 1 and attention_mask.shape[0] == 1:
        sliding_window_mask = _make_sliding_window_causal_mask(
            bsz=input_shape[0],
            tgt_len=input_shape[1],
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
            sliding_window=sliding_window,
        )
        attention_mask = attention_mask + sliding_window_mask
    else:
        LOG.info("skipping sliding window mask, not broadcastable with attention mask")

    return attention_mask


def mistral_model_forward(
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
    attention_mask = (
        self._prepare_decoder_attention_mask(  # pylint: disable=protected-access
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
            sliding_window=self.config.sliding_window,
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
                    return module(*inputs)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(decoder_layer),
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                None,
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


class MistralDecoderLayer(OriginalMistralDecoderLayer):
    """
    patched version of MistralDecoderLayer to pass through the precalculated cu_seqlens
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
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
