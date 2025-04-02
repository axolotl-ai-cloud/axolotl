"""Monkeypatch for gemma3 conditional generation forward to fix loss exploding"""

# pylint: disable=duplicate-code

from typing import Optional, Tuple, Union

import torch
from transformers.cache_utils import Cache
from transformers.models.gemma3.modeling_gemma3 import (
    _CONFIG_FOR_DOC,
    GEMMA3_INPUTS_DOCSTRING,
    Gemma3CausalLMOutputWithPast,
    logger,
)
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    is_torchdynamo_compiling,
    replace_return_docstrings,
)
from transformers.utils.deprecation import deprecate_kwarg


@deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
@add_start_docstrings_to_model_forward(GEMMA3_INPUTS_DOCSTRING)
@replace_return_docstrings(
    output_type=Gemma3CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
)
def new_forward(
    self,
    input_ids: torch.LongTensor = None,
    pixel_values: torch.FloatTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[list[torch.FloatTensor], Cache]] = None,
    token_type_ids: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    **lm_kwargs,
) -> Union[Tuple, Gemma3CausalLMOutputWithPast]:
    r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.text_config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.text_config.vocab_size]`.

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
    >>> from transformers import AutoProcessor, Gemma3ForConditionalGeneration

    >>> model = Gemma3ForConditionalGeneration.from_pretrained("google/Gemma3-test-224px-hf")
    >>> processor = AutoProcessor.from_pretrained("google/Gemma3-test-224px-hf")

    >>> prompt = "answer en Where is the cow standing?"
    >>> url = "https://huggingface.co/gv-hf/Gemma3-test-224px-hf/resolve/main/cow_beach_1.png"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> inputs = processor(images=image, text=prompt,  return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(**inputs, max_length=30)
    >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "answer en Where is the cow standing?\nbeach"
    ```"""

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

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

    is_training = token_type_ids is not None and labels is not None

    # Replace image id with PAD if the image token is OOV, to avoid index-errors
    if input_ids is not None and self.config.image_token_index >= self.vocab_size:
        special_image_mask = input_ids == self.config.image_token_index
        llm_input_ids = input_ids.clone()
        llm_input_ids[special_image_mask] = 0
    else:
        llm_input_ids = input_ids

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(llm_input_ids)

    if cache_position is None:
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )

    # Merge text and images
    if pixel_values is not None:
        image_features = self.get_image_features(pixel_values)

        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(
                    self.config.image_token_index,
                    dtype=torch.long,
                    device=inputs_embeds.device,
                )
            )
        else:
            special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(
                -1
            )
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(
                inputs_embeds.device
            )

        if (
            not is_torchdynamo_compiling()
            and inputs_embeds[special_image_mask].numel() != image_features.numel()
        ):
            image_tokens_in_text = (special_image_mask).sum(dim=1).sum(dim=0)[0]
            raise ValueError(
                f"Number of images does not match number of special image tokens in the input text. "
                f"Got {image_tokens_in_text} image tokens in the text but {image_features.shape[0] * image_features.shape[1]} "
                "tokens from image embeddings."
            )
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

    # mask out pad-token-ids in labels for BC
    if labels is not None and self.pad_token_id in labels:
        logger.warning_once(
            "`labels` contains `pad_token_id` which will be masked with `config.ignore_index`. "
            "You have to mask out `pad_token_id` when preparing `labels`, this behavior will be removed in v.4.46.",
        )
        labels = torch.where(
            input_ids == self.pad_token_id, self.config.ignore_index, labels
        )

    causal_mask = self._update_causal_mask(  # pylint: disable=protected-access
        attention_mask,
        token_type_ids,
        past_key_values,
        cache_position,
        inputs_embeds,
        is_training,
    )
    outputs = self.language_model(
        attention_mask=causal_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        logits_to_keep=logits_to_keep,
        **lm_kwargs,
    )

    logits = outputs[0]
    loss = None
    if labels is not None:
        if attention_mask is not None:
            # Get the shifted attention mask
            shift_attention_mask = attention_mask[:, -logits.shape[1] + 1 :].to(
                logits.device
            )  # +1 for shift

            # Filter logits and labels based on attention mask
            valid_indices = shift_attention_mask != 0
            filtered_logits = logits[..., :-1, :][valid_indices]
            filtered_labels = labels[..., 1:][valid_indices.to(labels.device)]

            # TODO: do we need to handle num_items_in_batch given we filter the logits and labels?

            loss = self.loss_function(
                logits=filtered_logits,
                labels=None,  # we pass shift_labels
                shift_labels=filtered_labels,
                vocab_size=self.config.text_config.vocab_size,
                **lm_kwargs,
            )
        else:
            # Standard case without filtering
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
                **lm_kwargs,
            )
    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Gemma3CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_hidden_states=image_features if pixel_values is not None else None,
    )


def patch_gemma3conditionalgeneration_forward():
    from transformers.models.gemma3.modeling_gemma3 import (
        Gemma3ForConditionalGeneration,
    )

    Gemma3ForConditionalGeneration.forward = new_forward
