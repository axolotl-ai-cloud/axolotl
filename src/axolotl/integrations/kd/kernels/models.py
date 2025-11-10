"""
model patcher for chunked top-k kl-div
"""

from typing import Optional, Union, Unpack

import torch
from transformers import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
    from transformers.utils import LossKwargs

    class TransformersKwargs(FlashAttentionKwargs, LossKwargs):
        """
        placeholder kwargs for hf model classes
        """

except ImportError:
    from transformers.utils.generic import (  # type: ignore[no-redef]
        TransformersKwargs,
    )

from axolotl.utils.callbacks.models import get_causal_lm_model_cls_prefix


def kldiv_forward_llama_like(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    target_logprobs: Optional[torch.Tensor] = None,
    target_token_ids: Optional[torch.LongTensor] = None,
    target_mask: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    **kwargs: Unpack[TransformersKwargs],  # type: ignore[misc]
) -> CausalLMOutputWithPast:
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

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs.last_hidden_state

    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    # TODO, we can optimize this further by filtering hidden_states on sequence dimension using labels != -100
    # self._loss_function should be LigerFusedLinearKLTopKLogprobLoss

    loss = self._loss_function(
        self.lm_head.weight,
        hidden_states,
        target_token_ids,
        target_logprobs,
        target_mask,
        true_labels=labels,
    )
    num_items_in_batch = kwargs.pop("num_items_in_batch", -1)
    if num_items_in_batch is not None and num_items_in_batch > 0:
        loss = loss / num_items_in_batch

    return CausalLMOutputWithPast(
        loss=loss,
        logits=None,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def apply_kernel(model_type):
    # Dynamically import the module and attention class
    module_path = f"transformers.models.{model_type}.modeling_{model_type}"
    model_cls_prefix, _ = get_causal_lm_model_cls_prefix(model_type)
    module = __import__(module_path, fromlist=[f"{model_cls_prefix}ForCausalLM"])
    model_cls = getattr(module, f"{model_cls_prefix}ForCausalLM")
    model_cls.forward = kldiv_forward_llama_like
