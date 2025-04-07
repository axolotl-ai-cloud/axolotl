"""Llama4 CCE patch. Adapted from transformers 4.51.0."""

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
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama4.modeling_llama4 import (
    _CONFIG_FOR_DOC,
    LLAMA4_INPUTS_DOCSTRING,
    Llama4CausalLMOutputWithPast,
)
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)

_PATCH_OPTS: PatchOptions | None = None


@add_start_docstrings_to_model_forward(LLAMA4_INPUTS_DOCSTRING)
@replace_return_docstrings(
    output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
)
def cce_forward(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
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
    **kwargs,
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    Args:
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

        defer_logits_calculation (`bool`, *optional*, defaults to `False`):
            If `True`, defer logits calculation to the ConditionalGeneration forward. This is used to avoid the
            memory overhead of calculating logits using regular lm_head forward pass and to use CCE.

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, Llama4ForCausalLM

    >>> model = Llama4ForCausalLM.from_pretrained("meta-llama4/Llama4-2-7b-hf")
    >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama4/Llama4-2-7b-hf")

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""
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
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs[0]
    loss = None
    logits = None

    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
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
            **kwargs,
        )
    elif _PATCH_OPTS is not None and defer_logits_calculation:
        # defer logits calculation to the ConditionalGeneration forward
        logits = hidden_states[:, slice_indices, :]
    else:
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

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


@replace_return_docstrings(
    output_type=Llama4CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
)
def cce_forward_multimodal(
    self,
    input_ids: torch.LongTensor | None = None,
    pixel_values: torch.FloatTensor | None = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[list[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    vision_feature_layer: Optional[Union[int, list[int]]] = None,
    vision_feature_select_strategy: Optional[str] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    image_sizes: torch.Tensor | None = None,
    **lm_kwargs,
) -> Union[Tuple, Llama4CausalLMOutputWithPast]:
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
    >>> from transformers import AutoProcessor, LlavaForConditionalGeneration

    >>> model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
    >>> processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

    >>> prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
    >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> inputs = processor(images=image, text=prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(**inputs, max_new_tokens=15)
    >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "USER:  \nWhat's the content of the image? ASSISTANT: The image features a busy city street with a stop sign prominently displayed"
    ```"""

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
    vision_feature_layer = (
        vision_feature_layer
        if vision_feature_layer is not None
        else self.config.vision_config.vision_feature_layer
    )
    vision_feature_select_strategy = (
        vision_feature_select_strategy
        if vision_feature_select_strategy is not None
        else self.config.vision_config.vision_feature_select_strategy
    )

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if pixel_values is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
        )

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    if pixel_values is not None:
        image_features = self.get_image_features(
            pixel_values=pixel_values,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            image_sizes=image_sizes,
        )
        original_inputs_embeds_shape = inputs_embeds.shape

        vision_flat = image_features.view(-1, image_features.size(-1))
        projected_vision_flat = self.multi_modal_projector(vision_flat)

        special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
        final_mask = special_image_mask.to(inputs_embeds.device)
        inputs_embeds = inputs_embeds.view(-1, inputs_embeds.size(-1))  # type: ignore

        final_mask_1d = final_mask[..., 0].reshape(-1)
        num_tokens_to_fill = final_mask_1d.sum()

        if num_tokens_to_fill != projected_vision_flat.size(0):
            raise ValueError(
                f"Mismatch: final_mask wants {num_tokens_to_fill} embeddings, "
                f"but multi_modal_projector returned {projected_vision_flat.size(0)}"
            )

        expanded_mask = final_mask_1d.unsqueeze(-1).expand(-1, inputs_embeds.size(-1))
        inputs_embeds = inputs_embeds.masked_scatter(
            expanded_mask, projected_vision_flat
        )  # type: ignore
        inputs_embeds = inputs_embeds.view(original_inputs_embeds_shape)  # type: ignore

    outputs = self.language_model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        logits_to_keep=logits_to_keep,
        defer_logits_calculation=True,  # enable deferred logits calculation
        **lm_kwargs,
    )

    hidden_states = outputs[0]
    loss = None
    logits = None

    if _PATCH_OPTS is not None and _PATCH_OPTS.use_lce(labels, self.training):
        assert labels is not None
        # TODO: check if need to handle attention_mask
        loss = apply_lce(
            hidden_states,
            self.language_model.lm_head.weight,
            labels,
            _PATCH_OPTS,
            **lm_kwargs,
        )
    else:
        logits = hidden_states
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -(logits.shape[1] - 1) :].to(
                    logits.device
                )
                shift_logits = logits[..., :-1, :][
                    shift_attention_mask.to(logits.device) != 0
                ].contiguous()
                shift_labels = labels[..., 1:][
                    shift_attention_mask.to(labels.device) != 0
                ].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1).to(shift_logits.device),
            )

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Llama4CausalLMOutputWithPast(
        loss=loss,
        logits=logits,  # type: ignore  # TODO: check if need to create dummy logits
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_hidden_states=image_features if pixel_values is not None else None,
    )


def patch_llama4_text(
    maybe_model: TransformersModelT | str | transformers.PretrainedConfig,
    patch_options: PatchOptions,
) -> TransformersModelT | None:
    global _PATCH_OPTS  # pylint: disable=global-statement
    from transformers.models.llama4 import modeling_llama4

    _PATCH_OPTS = patch_options

    if isinstance(maybe_model, transformers.PreTrainedModel):
        assert isinstance(
            maybe_model, modeling_llama4.Llama4ForCausalLM
        ), f"Expected a Llama4ForCausalLM model. Got {type(maybe_model)}."
        maybe_model.forward = MethodType(cce_forward, maybe_model)

        return maybe_model

    setattr(
        modeling_llama4.Llama4ForCausalLM,
        "forward",
        cce_forward,
    )
    return None


def patch_llama4(
    maybe_model: TransformersModelT | str | transformers.PretrainedConfig,
    patch_options: PatchOptions,
) -> TransformersModelT | None:

    global _PATCH_OPTS  # pylint: disable=global-statement
    from transformers.models.llama4 import modeling_llama4

    _PATCH_OPTS = patch_options

    if isinstance(maybe_model, transformers.PreTrainedModel):
        assert isinstance(
            maybe_model, modeling_llama4.Llama4ForConditionalGeneration
        ), f"Expected a Llama4ForConditionalGeneration model. Got {type(maybe_model)}."
        maybe_model.forward = MethodType(cce_forward_multimodal, maybe_model)

        # patch the language model
        maybe_model.language_model.forward = MethodType(
            cce_forward, maybe_model.language_model
        )
        return maybe_model

    setattr(
        modeling_llama4.Llama4ForConditionalGeneration,
        "forward",
        cce_forward_multimodal,
    )

    # patch the causal language model
    setattr(modeling_llama4.Llama4ForCausalLM, "forward", cce_forward)
    return None
