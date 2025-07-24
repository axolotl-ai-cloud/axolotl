"""Monkeypatch for voxtral to fix leaf node and dtype mismatch"""


def _forward(
    self,
    input_ids,
    input_features,
    attention_mask,
    position_ids,
    past_key_values,
    inputs_embeds,
    labels,
    use_cache,
    cache_position,
    logits_to_keep,
    **kwargs,
):
    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    if input_features is not None:
        audio_embeds = self.get_audio_embeds(input_features)

        # Cast audio_embeds to match inputs_embeds dtype
        audio_embeds = audio_embeds.to(inputs_embeds.dtype)

        # replace text-audio token placeholders with audio embeddings
        audio_token_mask = input_ids == self.config.audio_token_id

        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[audio_token_mask] = audio_embeds

    outputs = self.language_model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        labels=labels,
        use_cache=use_cache,
        cache_position=cache_position,
        logits_to_keep=logits_to_keep,
        **kwargs,
    )
    return outputs


def patch_voxtral_conditional_generation_forward():
    from transformers.models.voxtral.modeling_voxtral import (
        VoxtralForConditionalGeneration,
    )

    old_forward = VoxtralForConditionalGeneration.forward
    VoxtralForConditionalGeneration.forward = _forward

    def unpatch():
        VoxtralForConditionalGeneration.forward = old_forward

    return unpatch
