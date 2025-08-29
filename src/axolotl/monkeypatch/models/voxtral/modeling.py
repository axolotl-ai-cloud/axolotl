"""Monkeypatch for voxtral to fix leaf node and dtype mismatch"""

from typing import Optional, Union

import torch
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast


def patch_voxtral_conditional_generation_forward():
    from transformers.models.voxtral.modeling_voxtral import (
        VoxtralForConditionalGeneration,
    )

    # Store the original forward method
    old_forward = VoxtralForConditionalGeneration.forward

    def _forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
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

    # Apply the patch
    VoxtralForConditionalGeneration.forward = _forward

    def unpatch():
        """Restore the original forward method"""
        VoxtralForConditionalGeneration.forward = old_forward

    return unpatch
