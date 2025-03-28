"""Mistral and Mistral3 CCE patch."""

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
from transformers.models.mistral3.modeling_mistral3 import (
    Mistral3CausalLMOutputWithPast,
)
from transformers.models.mistral.modeling_mistral import (
    _CONFIG_FOR_DOC,
    MISTRAL_INPUTS_DOCSTRING,
    KwargsForCausalLM,
)
from transformers.processing_utils import Unpack
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    is_torchdynamo_compiling,
    replace_return_docstrings,
)
from transformers.utils.deprecation import deprecate_kwarg

_PATCH_OPTS: PatchOptions | None = None


@deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
@add_start_docstrings_to_model_forward(MISTRAL_INPUTS_DOCSTRING)
@replace_return_docstrings(
    output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
)
def cce_forward(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: Optional[torch.Tensor] | None = None,
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
    **kwargs: Unpack[KwargsForCausalLM],
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
    >>> from transformers import AutoTokenizer, MistralForCausalLM

    >>> model = MistralForCausalLM.from_pretrained("meta-mistral/Mistral-2-7b-hf")
    >>> tokenizer = AutoTokenizer.from_pretrained("meta-mistral/Mistral-2-7b-hf")

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


def cce_forward_multimodal(
    self,
    input_ids: torch.LongTensor | None = None,
    pixel_values: torch.FloatTensor | None = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[list[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    vision_feature_layer: Optional[Union[int, list[int]]] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    image_sizes: torch.Tensor | None = None,
    **lm_kwargs,
) -> Union[Tuple, Mistral3CausalLMOutputWithPast]:
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
    >>> from transformers import AutoProcessor, Mistral3ForConditionalGeneration

    >>> model = Mistral3ForConditionalGeneration.from_pretrained("mistralai/Mistral-Small-3.1-24B-Instruct-2503")
    >>> processor = AutoProcessor.from_pretrained("mistralai/Mistral-Small-3.1-24B-Instruct-2503")

    >>> prompt = "<s>[INST][IMG]What is the image?[/INST]"
    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> inputs = processor(images=image, text=prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(**inputs, max_new_tokens=15)
    >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "What is the image?The image depicts two cats lying on a pink blanket."
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
        else self.config.vision_feature_layer
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
            image_sizes=image_sizes,
        )

        special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
        special_image_mask = special_image_mask.expand_as(inputs_embeds).to(
            inputs_embeds.device
        )
        if (
            not is_torchdynamo_compiling()
            and inputs_embeds[special_image_mask].numel() != image_features.numel()
        ):
            n_image_tokens = (input_ids == self.config.image_token_index).sum()
            n_image_features = image_features.shape[0] * image_features.shape[1]
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)  # type: ignore

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

    return Mistral3CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_hidden_states=image_features if pixel_values is not None else None,
    )


def patch_mistral(
    maybe_model: TransformersModelT | str | transformers.PretrainedConfig,
    patch_options: PatchOptions,
) -> TransformersModelT | None:
    global _PATCH_OPTS  # pylint: disable=global-statement
    from transformers.models.mistral import modeling_mistral

    _PATCH_OPTS = patch_options

    if isinstance(maybe_model, transformers.PreTrainedModel):
        assert isinstance(
            maybe_model, modeling_mistral.MistralForCausalLM
        ), f"Expected a MistralForCausalLM model. Got {type(maybe_model)}."
        maybe_model.forward = MethodType(cce_forward, maybe_model)
        return maybe_model

    modeling_mistral.MistralForCausalLM.forward = cce_forward
    return None


def patch_mistral3(
    maybe_model: TransformersModelT | str | transformers.PretrainedConfig,
    patch_options: PatchOptions,
) -> TransformersModelT | None:
    global _PATCH_OPTS  # pylint: disable=global-statement
    from transformers.models.mistral import modeling_mistral
    from transformers.models.mistral3 import modeling_mistral3

    _PATCH_OPTS = patch_options

    if isinstance(maybe_model, transformers.PreTrainedModel):
        assert isinstance(
            maybe_model, modeling_mistral3.Mistral3ForConditionalGeneration
        ), f"Expected a Mistral3ForConditionalGeneration model. Got {type(maybe_model)}."
        maybe_model.forward = MethodType(cce_forward_multimodal, maybe_model)

        # patch the causal model to enable deferred logits calculation
        maybe_model.language_model.forward = MethodType(
            cce_forward, maybe_model.language_model
        )
        return maybe_model

    modeling_mistral3.Mistral3ForConditionalGeneration.forward = cce_forward_multimodal
    # patch the causal model to enable deferred logits calculation
    modeling_mistral.MistralForCausalLM.forward = cce_forward
    return None
