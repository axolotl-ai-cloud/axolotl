"""Mllama CCE patch."""

# pylint: disable=duplicate-code

from types import MethodType
from typing import Optional, Tuple, Union

import torch
import transformers
from cut_cross_entropy.transformers.utils import (
    PatchOptions,
    TransformersModelT,
    apply_lce,
)
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.mllama.modeling_mllama import (
    MLLAMA_INPUTS_DOCSTRING,
    _prepare_cross_attention_mask,
)
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.utils.deprecation import deprecate_kwarg

_PATCH_OPTS: PatchOptions | None = None


@deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
@add_start_docstrings_to_model_forward(MLLAMA_INPUTS_DOCSTRING)
@replace_return_docstrings(
    output_type=CausalLMOutputWithPast, config_class="MllamaTextConfig"
)
def cce_forward(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    cross_attention_states: Optional[torch.LongTensor] = None,
    cross_attention_mask: Optional[torch.LongTensor] = None,
    full_text_row_masked_out_mask: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    defer_logits_calculation: bool = False,
    **loss_kwargs,
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        logits_to_keep (`int` or `torch.Tensor`, *optional*):
            If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
            `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
            token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
            If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
            This is useful when using packed tensor format (single dimension for batch and sequence length).

        defer_logits_calculation (`bool`, *optional*):
            If `True`, defer logits calculation to the ConditionalGeneration forward. This is used to avoid the
            memory overhead of calculating logits using regular lm_head forward pass and to use CCE.

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, MllamaForCausalLM

    >>> model = MllamaForCausalLM.from_pretrained("Llama-3.2-11B-Vision")
    >>> tokenizer = AutoTokenizer.from_pretrained("Llama-3.2-11B-Vision")

    >>> prompt = "If I had to write a haiku, it would be:"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=40, do_sample=True, temperature=0.6)
    >>> result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    >>> print(result)
    If I had to write a haiku, it would be: "Snowflakes gently fall" - simple, yet peaceful.
    I love the idea of snowflakes gently falling, each one
    ```
    """
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
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        cross_attention_states=cross_attention_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        cross_attention_mask=cross_attention_mask,
        full_text_row_masked_out_mask=full_text_row_masked_out_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]
    loss = None
    logits = None

    slice_indices = (
        slice(-logits_to_keep, None)
        if isinstance(logits_to_keep, int)
        else logits_to_keep
    )

    if _PATCH_OPTS is not None and _PATCH_OPTS.use_lce(labels, self.training):
        assert labels is not None
        loss = apply_lce(
            hidden_states[:, slice_indices, :],
            self.lm_head.weight,
            labels,
            _PATCH_OPTS,
            **loss_kwargs,
        )
    elif _PATCH_OPTS is not None and defer_logits_calculation:
        # defer logits calculation to the ConditionalGeneration forward
        logits = hidden_states[:, slice_indices, :]
    else:
        logits = self.lm_head(hidden_states[:, slice_indices, :]).float()

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **loss_kwargs)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


@deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
@add_start_docstrings_to_model_forward(MLLAMA_INPUTS_DOCSTRING)
@replace_return_docstrings(
    output_type=CausalLMOutputWithPast, config_class="MllamaConfig"
)
def cce_forward_multimodal(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    aspect_ratio_mask: Optional[torch.Tensor] = None,
    aspect_ratio_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    cross_attention_mask: Optional[torch.Tensor] = None,
    cross_attention_states: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[list[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    **loss_kwargs,
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        logits_to_keep (`int` or `torch.Tensor`, *optional*):
            If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
            `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
            token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
            If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
            This is useful when using packed tensor format (single dimension for batch and sequence length).


    Returns:

    Example:

    ```python
    >>> from PIL import Image
    >>> import requests
    >>> from transformers import AutoProcessor, MllamaForConditionalGeneration

    >>> checkpoint = "meta-llama/Llama-3.2-11B-Vision"
    >>> model = MllamaForConditionalGeneration.from_pretrained(checkpoint)
    >>> processor = AutoProcessor.from_pretrained(checkpoint)

    >>> prompt = "<|image|>If I had to write a haiku for this one"
    >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> inputs = processor(text=prompt, images=image, return_tensors="pt")

    >>> # Generate
    >>> output = model.generate(**inputs, max_new_tokens=15)

    >>> prompt_len = inputs.input_ids.shape[-1]
    >>> generated_ids = output[:, prompt_len:]
    >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    >>> print(generated_text)
    [', it would be:.\\nA stop sign in Chinatown.\\n']
    ```
    """
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
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if pixel_values is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
        )

    if pixel_values is not None and cross_attention_states is not None:
        raise ValueError(
            "`pixel_values` and `cross_attention_states` cannot be provided simultaneously"
        )

    if pixel_values is not None:
        if aspect_ratio_ids is None:
            raise ValueError(
                "`aspect_ratio_ids` must be provided if `pixel_values` is provided"
            )
        # get vision tokens from vision model
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            aspect_ratio_ids=aspect_ratio_ids,
            aspect_ratio_mask=aspect_ratio_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )
        cross_attention_states = vision_outputs[0]
        cross_attention_states = self.multi_modal_projector(
            cross_attention_states
        ).reshape(
            -1, cross_attention_states.shape[-2], self.hidden_size  # type: ignore
        )

    if cross_attention_mask is not None:
        cross_attention_mask, full_text_row_masked_out_mask = (
            _prepare_cross_attention_mask(
                cross_attention_mask,
                num_vision_tokens=self.vision_model.num_patches,
                dtype=self.dtype,
            )
        )
    else:
        full_text_row_masked_out_mask = None

    if cross_attention_mask is not None and cache_position is not None:
        cross_attention_mask = cross_attention_mask[:, :, cache_position]
        full_text_row_masked_out_mask = full_text_row_masked_out_mask[
            :, :, cache_position
        ]

    outputs = self.language_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        cross_attention_states=cross_attention_states,
        cross_attention_mask=cross_attention_mask,
        full_text_row_masked_out_mask=full_text_row_masked_out_mask,
        past_key_values=past_key_values,
        use_cache=use_cache,
        inputs_embeds=inputs_embeds,
        output_hidden_states=output_hidden_states,
        output_attentions=output_attentions,
        return_dict=return_dict,
        cache_position=cache_position,
        logits_to_keep=logits_to_keep,
        defer_logits_calculation=True,  # enable deferred logits calculation
        **loss_kwargs,
    )

    hidden_states = outputs[0]
    loss = None
    logits = None

    if _PATCH_OPTS is not None and _PATCH_OPTS.use_lce(labels, self.training):
        assert labels is not None
        loss = apply_lce(
            hidden_states,
            self.language_model.lm_head.weight,
            labels,
            _PATCH_OPTS,
            **loss_kwargs,
        )
    else:
        # Temporary fix to calculate the loss in main class, as the model's vocab size may be resized
        logits = hidden_states

        if labels is not None:
            loss = self.loss_function(
                logits, labels, self.config.get_text_config().vocab_size, **loss_kwargs
            )

    if not return_dict:
        return (loss,) + outputs if loss is not None else outputs

    return CausalLMOutputWithPast(
        loss=loss,
        logits=outputs.logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def patch_mllama(
    maybe_model: TransformersModelT | str | transformers.PretrainedConfig,
    patch_options: PatchOptions,
) -> TransformersModelT | None:

    global _PATCH_OPTS  # pylint: disable=global-statement
    from transformers.models.mllama import modeling_mllama

    _PATCH_OPTS = patch_options

    if isinstance(maybe_model, transformers.PreTrainedModel):
        assert isinstance(
            maybe_model, modeling_mllama.MllamaForConditionalGeneration
        ), f"Expected a MllamaForConditionalGeneration model. Got {type(maybe_model)}."
        maybe_model.forward = MethodType(cce_forward_multimodal, maybe_model)

        # patch the language model
        maybe_model.language_model.forward = MethodType(
            cce_forward, maybe_model.language_model
        )
        return maybe_model

    modeling_mllama.MllamaForConditionalGeneration.forward = cce_forward_multimodal

    # patch the causal language model
    modeling_mllama.MllamaForCausalLM.forward = cce_forward
    return None
