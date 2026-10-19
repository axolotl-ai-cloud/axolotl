"""Model-local packed encoder/canvas helpers for DiffusionGemma."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
from transformers.cache_utils import DynamicCache
from transformers.models.diffusion_gemma import DiffusionGemmaForBlockDiffusion
from transformers.utils import ModelOutput

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


@dataclass
class PackedDiffusionGemmaOutput(ModelOutput):
    """Output of one packed encoder/canvas forward for DDP-aware training."""

    logits: torch.Tensor | None = None
    past_key_values: Any | None = None
    encoder_last_hidden_state: torch.Tensor | None = None
    encoder_logits: torch.Tensor | None = None
    denoised_input_ids: torch.Tensor | None = None


def _encoder_feed_forward(layer, hidden_states: torch.Tensor) -> torch.Tensor:
    """Dense MLP + routed experts half of a DiffusionGemma encoder/decoder text layer."""
    residual = hidden_states
    hidden_states = layer.pre_feedforward_layernorm(hidden_states)
    hidden_states = layer.mlp(hidden_states)
    hidden_states_1 = layer.post_feedforward_layernorm_1(hidden_states)

    hidden_states_flat = residual.reshape(-1, residual.shape[-1])
    hidden_states_2_for_experts = layer.pre_feedforward_layernorm_2(hidden_states_flat)
    _, top_k_weights, top_k_index = layer.router(hidden_states_flat)
    hidden_states_2 = layer.experts(
        hidden_states_2_for_experts, top_k_index, top_k_weights
    )
    hidden_states_2 = hidden_states_2.reshape(residual.shape)
    hidden_states_2 = layer.post_feedforward_layernorm_2(hidden_states_2)

    hidden_states = hidden_states_1 + hidden_states_2
    hidden_states = layer.post_feedforward_layernorm(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states * layer.layer_scalar


def _checkpointed_feed_forward(layer, hidden_states: torch.Tensor) -> torch.Tensor:
    """The checkpointed region: the feed-forward half with autocast's cast cache off.

    Autocast caches casts of leaf weights (the fp32 LoRA factors) for the lifetime of
    the outer autocast context. The forward pass hits that cache (tied encoder/decoder
    factors were cast earlier), the backward recompute does not, so the region's op
    stream differs and selective checkpointing's per-op replay fails ("invocation
    index ... not found in storage"). Without the cache both passes cast identically.
    """
    device_type = hidden_states.device.type
    if torch.is_autocast_enabled(device_type):
        with torch.autocast(
            device_type,
            dtype=torch.get_autocast_dtype(device_type),
            cache_enabled=False,
        ):
            return _encoder_feed_forward(layer, hidden_states)
    return _encoder_feed_forward(layer, hidden_states)


def _encoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: torch.Tensor = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values=None,
    **kwargs,
) -> torch.Tensor:
    """DiffusionGemma text-layer forward with only the feed-forward half checkpointed.

    Encoder: HF keeps encoder layers out of gradient checkpointing because attention
    appends to the shared KV cache, and a full-layer replay would append twice. Only
    the attention half touches the cache, so the dense MLP + MoE half (the bulk of the
    saved activations) is recomputed in backward while attention and the cache write
    run exactly once.

    Decoder: the same split replaces HF's whole-layer checkpoint. Its attention runs on
    the canvas only (small), and keeping the compiled FlexAttention call out of the
    selective-checkpointing region keeps SAC's per-op replay bookkeeping valid: the
    first, compiling flex call issues extra ops that its recompute does not, and SAC
    then fails with "invocation index ... not found in storage".
    """
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, _ = self.self_attn(
        hidden_states=hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        **kwargs,
    )
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = residual + hidden_states
    if self.training and torch.is_grad_enabled():
        # Non-reentrant only: reentrant checkpointing runs the forward under no_grad,
        # which would cut the decoder -> encoder (tied LoRA) gradient. The SAC
        # context_fn force-saves the router's topk so the recompute replays the
        # forward's expert choice (a near-tie could otherwise flip it silently).
        return torch.utils.checkpoint.checkpoint(
            _checkpointed_feed_forward,
            self,
            hidden_states,
            **self._axolotl_checkpoint_kwargs,
        )
    return _encoder_feed_forward(self, hidden_states)


def set_encoder_feed_forward_checkpointing(
    model, enabled: bool, checkpoint_kwargs: dict | None = None
) -> int:
    """Checkpoint (or restore) the feed-forward half of every DiffusionGemma text layer."""
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
        DiffusionGemmaDecoderTextLayer,
        DiffusionGemmaEncoderTextLayer,
    )

    count = 0
    for module in model.modules():
        if not isinstance(
            module, (DiffusionGemmaEncoderTextLayer, DiffusionGemmaDecoderTextLayer)
        ):
            continue
        if isinstance(module, DiffusionGemmaDecoderTextLayer):
            # the split forward replaces HF's whole-layer checkpoint
            module.gradient_checkpointing = False
        if enabled:
            module._axolotl_checkpoint_kwargs = dict(
                checkpoint_kwargs or {"use_reentrant": False}
            )
            module.forward = _encoder_layer_forward.__get__(module)
        else:
            module.__dict__.pop("forward", None)
            module.__dict__.pop("_axolotl_checkpoint_kwargs", None)
        count += 1
    return count


def _register_router_topk_saves(model) -> int:
    """Force-save every text router's ``topk`` under selective checkpointing.

    Encoder and decoder routers both run inside checkpointed regions (the encoder's
    feed-forward half, the decoder's whole layer), so the scoped save applies to the
    forward and to the recompute, which then reuses the forward's expert choice.
    """
    from axolotl.monkeypatch.selective_checkpointing import register_mandatory_save

    routers = [
        module
        for name, module in model.named_modules()
        if name.endswith(".router")
        and type(module).__name__ == "DiffusionGemmaTextRouter"
        and not getattr(module, "_axolotl_topk_saved", False)
    ]
    if routers:
        register_mandatory_save(ops={"aten::topk"}, within=routers)
        for module in routers:
            module._axolotl_topk_saved = True
    return len(routers)


def _sac_checkpoint_kwargs(gradient_checkpointing_kwargs: dict | None) -> dict:
    """Non-reentrant checkpoint kwargs carrying a SAC ``context_fn``.

    ``selective_checkpointing:`` (``apply_selective_checkpointing``) supplies its own
    context_fn; without it, a policy that saves only the registered ops (the router
    topk) is installed, as the expert-parallel plugin does.
    """
    kwargs = dict(gradient_checkpointing_kwargs or {})
    if kwargs.get("use_reentrant", False):
        raise ValueError(
            "DiffusionGemma requires gradient_checkpointing_kwargs.use_reentrant: false; "
            "reentrant checkpointing drops the decoder -> encoder gradient and cannot "
            "replay the router's topk."
        )
    kwargs["use_reentrant"] = False
    if kwargs.get("context_fn") is None:
        from axolotl.monkeypatch.selective_checkpointing import build_sac_context_fn

        kwargs["context_fn"] = build_sac_context_fn(save=[])
    return kwargs


class AxolotlDiffusionGemmaForBlockDiffusion(DiffusionGemmaForBlockDiffusion):
    """DiffusionGemma with an explicit packed encoder/canvas forward contract."""

    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs=None, **kwargs
    ):
        routers = _register_router_topk_saves(self)
        checkpoint_kwargs = _sac_checkpoint_kwargs(gradient_checkpointing_kwargs)
        super().gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=checkpoint_kwargs, **kwargs
        )
        # Encoder and decoder layers checkpoint only their feed-forward half.
        count = set_encoder_feed_forward_checkpointing(self, True, checkpoint_kwargs)
        LOG.info(
            "DiffusionGemma: checkpointing the feed-forward half of %d encoder/decoder "
            "layers; "
            "router topk saved for recompute (%d routers)",
            count,
            routers,
        )

    def gradient_checkpointing_disable(self, *args, **kwargs):
        super().gradient_checkpointing_disable(*args, **kwargs)
        set_encoder_feed_forward_checkpointing(self, False)

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
        return_encoder_logits: bool = True,
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
                return_encoder_logits=return_encoder_logits,
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
            # Full-vocabulary logits over every prompt token are [tokens, 262k] fp32;
            # skip them when no encoder AR loss consumes them.
            encoder_logits=(
                _encoder_logits(self, output.encoder_last_hidden_state)
                if return_encoder_logits
                else None
            ),
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
    return_encoder_logits: bool = True,
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
        encoder_logits=(
            _encoder_logits(model, encoder_last_hidden_state)
            if return_encoder_logits
            else None
        ),
        denoised_input_ids=torch.where(
            update_mask, logits.detach().argmax(-1).to(final_state.dtype), final_state
        ),
    )
