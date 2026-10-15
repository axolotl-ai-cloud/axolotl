"""Model-local packed encoder/canvas helpers for DiffusionGemma."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
from transformers.cache_utils import DynamicCache
from transformers.models.diffusion_gemma import DiffusionGemmaForBlockDiffusion
from transformers.utils import ModelOutput


@dataclass
class PackedDiffusionGemmaOutput(ModelOutput):
    """Output of one packed encoder/canvas forward for DDP-aware training."""

    logits: torch.Tensor | None = None
    past_key_values: Any | None = None
    encoder_last_hidden_state: torch.Tensor | None = None
    encoder_logits: torch.Tensor | None = None
    denoised_input_ids: torch.Tensor | None = None


class AxolotlDiffusionGemmaForBlockDiffusion(DiffusionGemmaForBlockDiffusion):
    """DiffusionGemma with an explicit packed encoder/canvas forward contract."""

    # transformers answers these by scanning the source of cls.__module__ for its
    # decorators, which this module lacks; defer to the upstream class.
    @classmethod
    def _can_set_experts_implementation(cls) -> bool:
        return DiffusionGemmaForBlockDiffusion._can_set_experts_implementation()

    @classmethod
    def _can_set_attn_implementation(cls) -> bool:
        return DiffusionGemmaForBlockDiffusion._can_set_attn_implementation()

    def forward(
        self,
        *args,
        encoder_input_ids: torch.Tensor | None = None,
        encoder_attention_mask: dict[str, torch.Tensor] | None = None,
        encoder_position_ids: torch.Tensor | None = None,
        decoder_input_ids: torch.Tensor | None = None,
        decoder_attention_mask: dict[str, torch.Tensor] | None = None,
        decoder_position_ids: torch.Tensor | None = None,
        past_key_values=None,
        self_conditioning_logits: torch.Tensor | None = None,
        self_conditioning_token_mask: torch.Tensor | None = None,
        unroll_steps: int | None = None,
        pilot_for_single_step: bool = True,
        grad_through_steps: bool = False,
        k1_conditioning_mask: torch.Tensor | None = None,
        recurrent_conditioning_mask: torch.Tensor | None = None,
        update_mask: torch.Tensor | None = None,
        kernel_options: dict | None = None,
        **kwargs,
    ):
        if encoder_input_ids is None:
            if past_key_values is not None:
                kwargs["past_key_values"] = past_key_values
            if decoder_input_ids is not None:
                kwargs["decoder_input_ids"] = decoder_input_ids
            if decoder_attention_mask is not None:
                kwargs["decoder_attention_mask"] = decoder_attention_mask
            if decoder_position_ids is not None:
                kwargs["decoder_position_ids"] = decoder_position_ids
            if self_conditioning_logits is not None:
                kwargs["self_conditioning_logits"] = self_conditioning_logits
            return super().forward(*args, **kwargs)
        if args or any(
            value is None
            for value in (
                encoder_attention_mask,
                encoder_position_ids,
                decoder_input_ids,
                decoder_attention_mask,
                decoder_position_ids,
            )
        ):
            raise ValueError(
                "packed DiffusionGemma forward requires every encoder/canvas field"
            )
        assert encoder_attention_mask is not None
        assert encoder_position_ids is not None
        assert decoder_input_ids is not None
        assert decoder_attention_mask is not None
        assert decoder_position_ids is not None
        if unroll_steps is not None:
            if update_mask is None:
                raise ValueError("packed unroll requires an update mask")
            return forward_packed_encoder_canvas_unroll(
                self,
                encoder_input_ids=encoder_input_ids,
                encoder_attention_mask=encoder_attention_mask,
                encoder_position_ids=encoder_position_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                decoder_position_ids=decoder_position_ids,
                past_key_values=past_key_values,
                steps=unroll_steps,
                pilot_for_single_step=pilot_for_single_step,
                grad_through_steps=grad_through_steps,
                k1_conditioning_mask=k1_conditioning_mask,
                recurrent_conditioning_mask=recurrent_conditioning_mask,
                update_mask=update_mask,
                kernel_options=kernel_options,
            )
        output = forward_packed_encoder_canvas(
            self,
            encoder_input_ids=encoder_input_ids,
            encoder_attention_mask=encoder_attention_mask,
            encoder_position_ids=encoder_position_ids,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            past_key_values=past_key_values,
            self_conditioning_logits=self_conditioning_logits,
            self_conditioning_token_mask=self_conditioning_token_mask,
            kernel_options=kernel_options,
        )
        return PackedDiffusionGemmaOutput(
            logits=output.logits,
            past_key_values=output.past_key_values,
            encoder_last_hidden_state=output.encoder_last_hidden_state,
            encoder_logits=_encoder_logits(self, output.encoder_last_hidden_state),
        )


AxolotlDiffusionGemmaForBlockDiffusion.__name__ = "DiffusionGemmaForBlockDiffusion"


def _native_diffusion_gemma(model):
    """Find the module owning DiffusionGemma's encoder and decoder."""
    candidate = model
    while not (hasattr(candidate, "model") and hasattr(candidate.model, "encoder")):
        candidate = getattr(candidate, "model", None)
        if candidate is None:
            raise TypeError("expected a DiffusionGemma model or its PEFT wrapper")
    return candidate


def _attention_masks_for_model(
    masks: dict[str, torch.Tensor], config, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    """Convert dense boolean masks for the eager attention implementation."""
    if getattr(config, "_attn_implementation", None) != "eager":
        return masks
    minimum = torch.finfo(dtype).min
    return {
        name: (
            torch.where(mask, torch.zeros((), device=mask.device, dtype=dtype), minimum)
            if isinstance(mask, torch.Tensor) and mask.dtype is torch.bool
            else mask
        )
        for name, mask in masks.items()
    }


def encode_packed_prefix(
    model,
    encoder_input_ids: torch.Tensor,
    encoder_attention_mask: dict[str, torch.Tensor],
    encoder_position_ids: torch.Tensor,
    kernel_options: dict | None = None,
):
    """Build the read-only decoder cache and retain encoder hidden states for AR loss."""
    native_model = _native_diffusion_gemma(model)
    encoded = native_model.model.encoder(
        input_ids=encoder_input_ids,
        attention_mask=_attention_masks_for_model(
            encoder_attention_mask,
            native_model.model.encoder.language_model.config,
            native_model.model.encoder.language_model.embed_tokens.weight.dtype,
        ),
        position_ids=encoder_position_ids,
        # Packed documents are isolated by the supplied masks. A config-backed
        # cache would evict old keys for sliding layers across document boundaries.
        past_key_values=DynamicCache(),
        kernel_options=kernel_options,
    )
    return SimpleNamespace(
        past_key_values=encoded.past_key_values,
        encoder_last_hidden_state=encoded.last_hidden_state,
    )


def decode_packed_canvas(
    model,
    decoder_input_ids: torch.Tensor,
    cache,
    decoder_attention_mask: dict[str, torch.Tensor],
    decoder_position_ids: torch.Tensor,
    logits: torch.Tensor | None = None,
    token_gate: torch.Tensor | None = None,
    kernel_options: dict | None = None,
) -> torch.Tensor:
    """Run the native decoder with independent self-conditioning per canvas token."""
    native_model = _native_diffusion_gemma(model)
    decoder = native_model.model.decoder
    inputs = decoder.embed_tokens(decoder_input_ids)
    decoder_attention_mask = _attention_masks_for_model(
        decoder_attention_mask, decoder.text_config, inputs.dtype
    )
    if logits is None:
        soft = torch.zeros_like(inputs)
    else:
        if logits.shape[:2] != decoder_input_ids.shape:
            raise ValueError(
                "self-conditioning logits must align with decoder input IDs"
            )
        soft = torch.matmul(
            logits.softmax(-1, dtype=torch.float32).to(
                decoder.embed_tokens.weight.dtype
            ),
            decoder.embed_tokens.weight,
        ) * decoder.embed_tokens.embed_scale.to(inputs.dtype)
    if token_gate is not None:
        if token_gate.shape != soft.shape[:2]:
            raise ValueError("self-conditioning token gate must be [batch, canvas]")
        soft = soft * token_gate[..., None].to(soft.dtype)
    hidden = decoder.self_conditioning(inputs, soft)
    positions = {
        kind: decoder.rotary_emb(hidden, decoder_position_ids, kind)
        for kind in decoder.unique_layer_types
    }
    for index, layer in enumerate(
        decoder.layers[: decoder.text_config.num_hidden_layers]
    ):
        hidden = layer(
            hidden,
            position_embeddings=positions[decoder.text_config.layer_types[index]],
            attention_mask=decoder_attention_mask[
                decoder.text_config.layer_types[index]
            ],
            position_ids=decoder_position_ids,
            past_key_values=cache,
            kernel_options=kernel_options,
        )
    logits = native_model.lm_head(decoder.norm(hidden)).float()
    softcap = getattr(native_model, "final_logit_softcapping", None)
    if softcap is None:
        return logits
    return torch.tanh(logits / softcap) * softcap


def _encoder_logits(model, hidden_states: torch.Tensor | None) -> torch.Tensor | None:
    if hidden_states is None:
        return None
    native_model = _native_diffusion_gemma(model)
    logits = native_model.lm_head(hidden_states).float()
    softcap = getattr(native_model, "final_logit_softcapping", None)
    if softcap is None:
        return logits
    return torch.tanh(logits / softcap) * softcap


def forward_packed_encoder_canvas(
    model,
    *,
    encoder_input_ids: torch.Tensor,
    encoder_attention_mask: dict[str, torch.Tensor],
    encoder_position_ids: torch.Tensor,
    decoder_input_ids: torch.Tensor,
    decoder_attention_mask: dict[str, torch.Tensor],
    decoder_position_ids: torch.Tensor,
    past_key_values=None,
    self_conditioning_logits: torch.Tensor | None = None,
    self_conditioning_token_mask: torch.Tensor | None = None,
    kernel_options: dict | None = None,
):
    """Encode once when needed, then decode a packed canvas with token-level SC."""
    encoder_last_hidden_state = None
    if past_key_values is None:
        encoded = encode_packed_prefix(
            model,
            encoder_input_ids,
            encoder_attention_mask,
            encoder_position_ids,
            kernel_options,
        )
        past_key_values = encoded.past_key_values
        encoder_last_hidden_state = encoded.encoder_last_hidden_state
    return SimpleNamespace(
        logits=decode_packed_canvas(
            model,
            decoder_input_ids,
            past_key_values,
            decoder_attention_mask,
            decoder_position_ids,
            self_conditioning_logits,
            self_conditioning_token_mask,
            kernel_options,
        ),
        past_key_values=past_key_values,
        encoder_last_hidden_state=encoder_last_hidden_state,
    )


def forward_packed_encoder_canvas_unroll(
    model,
    *,
    encoder_input_ids: torch.Tensor,
    encoder_attention_mask: dict[str, torch.Tensor],
    encoder_position_ids: torch.Tensor,
    decoder_input_ids: torch.Tensor,
    decoder_attention_mask: dict[str, torch.Tensor],
    decoder_position_ids: torch.Tensor,
    steps: int,
    pilot_for_single_step: bool,
    grad_through_steps: bool,
    k1_conditioning_mask: torch.Tensor | None,
    recurrent_conditioning_mask: torch.Tensor | None,
    update_mask: torch.Tensor,
    past_key_values=None,
    kernel_options: dict | None = None,
) -> PackedDiffusionGemmaOutput:
    """Run a packed denoising unroll inside one DDP-visible model forward."""
    encoder_last_hidden_state = None
    if past_key_values is None:
        encoded = encode_packed_prefix(
            model,
            encoder_input_ids,
            encoder_attention_mask,
            encoder_position_ids,
            kernel_options,
        )
        past_key_values = encoded.past_key_values
        encoder_last_hidden_state = encoded.encoder_last_hidden_state

    from axolotl.core.trainers.diffusion_lm.unroll import run_unroll

    def forward_step(state, conditioning, conditioning_mask):
        return decode_packed_canvas(
            model,
            state,
            past_key_values,
            decoder_attention_mask,
            decoder_position_ids,
            conditioning,
            conditioning_mask,
            kernel_options,
        )

    _, logits, final_state = run_unroll(
        state=decoder_input_ids,
        update_mask=update_mask,
        steps=steps,
        grad_through_steps=grad_through_steps,
        supports_self_conditioning=True,
        k1_conditioning_mask=k1_conditioning_mask,
        recurrent_conditioning_mask=recurrent_conditioning_mask,
        forward_step=forward_step,
        logits_from_outputs=lambda output: output,
        update_state=lambda state, output, mask: torch.where(
            mask, output.detach().argmax(-1).to(state.dtype), state
        ),
        pilot_for_single_step=pilot_for_single_step,
    )
    return PackedDiffusionGemmaOutput(
        logits=logits,
        past_key_values=past_key_values,
        encoder_last_hidden_state=encoder_last_hidden_state,
        encoder_logits=_encoder_logits(model, encoder_last_hidden_state),
        denoised_input_ids=torch.where(
            update_mask, logits.detach().argmax(-1).to(final_state.dtype), final_state
        ),
    )
