"""Native source loader for Nemotron Labs Diffusion."""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers.utils import ModelOutput


@dataclass
class SelectedLogitsOutput(ModelOutput):
    logits: torch.Tensor | None = None
    axolotl_selected_logits: bool | None = None


def project_selected_logits(
    hidden: torch.Tensor,
    head: torch.nn.Module,
    rows: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    if rows.shape != positions.shape or rows.ndim != 2:
        raise ValueError(
            "selected logits coordinates must be matching [examples, questions]"
        )
    return head(hidden[rows.clamp_min(0), positions.clamp_min(0)])


def resolve_nemotron_model_class(
    model_source: str | Path, *, revision: str | None = None
) -> type:
    """Resolve native code while preserving the requested Hub revision."""
    source = Path(model_source)
    resolved_revision = None
    if not (source / "modeling_nemotron_labs_diffusion.py").is_file():
        from huggingface_hub import snapshot_download

        source = Path(
            snapshot_download(
                repo_id=str(model_source),
                revision=revision,
                allow_patterns=[
                    "config.json",
                    "configuration_nemotron_labs_diffusion.py",
                    "modeling_ministral.py",
                    "modeling_nemotron_labs_diffusion.py",
                ],
            )
        )
        if source.parent.name == "snapshots" and re.fullmatch(
            r"[0-9a-f]{40}", source.name
        ):
            resolved_revision = source.name
    required_files = (
        "configuration_nemotron_labs_diffusion.py",
        "modeling_ministral.py",
        "modeling_nemotron_labs_diffusion.py",
    )
    missing = [name for name in required_files if not (source / name).is_file()]
    if missing:
        raise ValueError(f"Nemotron source lacks required native files: {missing}")
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    model_class = get_class_from_dynamic_module(
        "modeling_nemotron_labs_diffusion.NemotronLabsDiffusionModel",
        str(source),
        local_files_only=True,
    )

    class MaskAwareNemotron(model_class):  # type: ignore[valid-type, misc]
        _supports_flex_attn = True
        supports_diffusion_varlen = True
        _axolotl_resolved_revision = resolved_revision

        def __init__(self, config):
            if config.dlm_paradigm != "bidirectional":
                raise ValueError(
                    "Native Nemotron support currently requires dlm_paradigm='bidirectional'."
                )
            super().__init__(config)
            enable_nemotron_explicit_attention_mask(self)

        supports_selected_logits = True

        def forward(self, *args, **kwargs):
            selected = kwargs.pop("axolotl_selected_logits", None)
            metadata = kwargs.get("diffusion_varlen")
            if metadata is not None:
                from axolotl.integrations.diffusion.lm.varlen import VarlenMetadata

                if (
                    not isinstance(metadata, VarlenMetadata)
                    or kwargs.get("attention_mask") is not None
                    or kwargs.get("past_key_values") is not None
                    or kwargs.get("use_cache")
                    or kwargs.get("use_causal_mask")
                    or kwargs.get("labels") is not None
                    or len(args) > 1
                ):
                    raise ValueError(
                        "Nemotron varlen requires VarlenMetadata, no mask/cache, and bidirectional mode"
                    )
                position_ids = kwargs.get("position_ids")
                if position_ids is None or position_ids.shape != (
                    metadata.batch_size,
                    metadata.sequence_length,
                ):
                    raise ValueError(
                        "Nemotron varlen requires document-local position_ids"
                    )
                if kwargs.get("cache_position") is None:
                    kwargs["cache_position"] = position_ids[0]
                kwargs["use_cache"] = False
                kwargs["use_causal_mask"] = False
            attention_mask = kwargs.get("attention_mask")
            is_block_mask = type(attention_mask).__name__ == "BlockMask"
            if getattr(attention_mask, "ndim", None) == 4 or is_block_mask:
                kwargs["use_causal_mask"] = True
                position_ids = kwargs.get("position_ids")
                if position_ids is not None and kwargs.get("cache_position") is None:
                    kwargs["cache_position"] = position_ids[0]
            if selected is None:
                return super().forward(*args, **kwargs)
            if kwargs.get("labels") is not None:
                raise ValueError("selected logits do not accept labels")
            if not isinstance(selected, tuple) or len(selected) != 2:
                raise TypeError("axolotl_selected_logits must be (rows, positions)")
            rows, positions = selected
            if not isinstance(rows, torch.Tensor) or not isinstance(
                positions, torch.Tensor
            ):
                raise TypeError("selected logits coordinates must be tensors")
            if rows.shape != positions.shape or rows.ndim != 2:
                raise ValueError(
                    "selected logits coordinates must be matching [examples, questions]"
                )
            hidden_output = super().forward(
                *args, output_last_hidden_states_only=True, **kwargs
            )
            return SelectedLogitsOutput(
                logits=project_selected_logits(
                    hidden_output.last_hidden_state,
                    self.diffusion_head,
                    rows,
                    positions,
                ),
                axolotl_selected_logits=True,
            )

        @torch.no_grad()
        def generate_with_denoising_steps(
            self,
            prompt_ids: torch.Tensor,
            max_new_tokens: int,
            block_length: int,
            denoising_steps: int,
            threshold: float | None = None,
            causal_context: bool = True,
            temperature: float = 0.0,
            eos_token_id: int | None = None,
            max_thinking_tokens: int | None = None,
            end_think_token_id: int | None = None,
        ) -> tuple[torch.Tensor, int]:
            """Run the pinned sampler with an explicit per-block denoising count."""
            if denoising_steps < 1:
                raise ValueError("Nemotron denoising_steps must be at least 1.")
            if max_new_tokens % block_length:
                raise ValueError(
                    "Nemotron max_new_tokens must be divisible by block_length."
                )
            if eos_token_id is None:
                eos_token_id = getattr(self.config, "eos_token_id", None)
            source = importlib.import_module(model_class.__module__)
            mask_id = self.mask_token_id
            x_accum = prompt_ids.clone()
            batch_size = prompt_ids.shape[0]
            num_blocks = max_new_tokens // block_length
            nfe = 0

            def set_diffusion_lm(value: bool) -> None:
                for layer in self.encoder.layers:
                    if hasattr(layer.self_attn, "diffusion_lm"):
                        layer.self_attn.diffusion_lm = value

            if causal_context:
                set_diffusion_lm(False)
            output = self(prompt_ids, use_cache=True, use_causal_mask=causal_context)
            past_key_values = output.past_key_values
            if causal_context:
                set_diffusion_lm(True)

            next_token = None
            if causal_context:
                last_logit = output.logits[:, -1, :]
                if temperature > 0:
                    next_token = torch.multinomial(
                        torch.softmax(last_logit / temperature, dim=-1), num_samples=1
                    )
                else:
                    next_token = torch.argmax(last_logit, dim=-1, keepdim=True)

            for num_block in range(num_blocks):
                mask_block = torch.full(
                    (batch_size, block_length),
                    mask_id,
                    dtype=prompt_ids.dtype,
                    device=prompt_ids.device,
                )
                if causal_context:
                    assert next_token is not None
                    mask_block[:, 0] = next_token[:, 0]

                x_accum = torch.cat([x_accum, mask_block], dim=1)
                block_start = prompt_ids.size(1) + num_block * block_length
                block_slice = slice(block_start, block_start + block_length)

                if end_think_token_id is not None and max_thinking_tokens is not None:
                    tokens_before = num_block * block_length
                    tokens_after = tokens_before + block_length
                    if tokens_after > max_thinking_tokens:
                        gen_so_far = x_accum[:, prompt_ids.size(1) : block_start]
                        has_end_think = (
                            (gen_so_far == end_think_token_id).any(dim=1)
                            if gen_so_far.size(1) > 0
                            else torch.zeros(
                                batch_size,
                                dtype=torch.bool,
                                device=prompt_ids.device,
                            )
                        )
                        if not has_end_think.all():
                            offset = max(0, max_thinking_tokens - tokens_before)
                            inject_pos = block_start + offset
                            for batch_idx in range(batch_size):
                                if not has_end_think[batch_idx]:
                                    x_accum[batch_idx, inject_pos] = end_think_token_id

                mask_block_idx0 = x_accum[:, block_slice] == mask_id
                num_transfer_tokens = source._get_num_transfer_tokens(
                    mask_block_idx0, denoising_steps
                )

                for step in range(denoising_steps):
                    mask_block_idx = x_accum[:, block_slice] == mask_id
                    if mask_block_idx.sum() == 0:
                        break

                    nfe += 1
                    logits_block = self(
                        x_accum[:, block_slice],
                        past_key_values=past_key_values,
                        use_cache=False,
                    ).logits
                    x0, transfer_idx = source._get_transfer_index(
                        logits_block,
                        temperature,
                        mask_block_idx,
                        x_accum[:, block_slice],
                        num_transfer_tokens=num_transfer_tokens[:, step],
                        threshold=threshold,
                    )
                    current = x_accum[:, block_slice].clone()
                    current[transfer_idx] = x0[transfer_idx]
                    x_accum[:, block_slice] = current

                    if eos_token_id is not None:
                        block_tokens = x_accum[:, block_slice]
                        eos_mask = block_tokens == eos_token_id
                        if eos_mask.any(dim=1).any():
                            after_eos = eos_mask.cumsum(dim=1).bool()
                            mask_before = (block_tokens == mask_id) & ~after_eos
                            if (eos_mask.any(dim=1) & ~mask_before.any(dim=1)).any():
                                break

                if causal_context:
                    set_diffusion_lm(False)
                output = self(
                    x_accum[:, block_slice],
                    past_key_values=past_key_values,
                    use_cache=True,
                    use_causal_mask=causal_context,
                )
                past_key_values = output.past_key_values
                nfe += 1

                if causal_context:
                    set_diffusion_lm(True)
                    last_logit = output.logits[:, -1, :]
                    if temperature > 0:
                        next_token = torch.multinomial(
                            torch.softmax(last_logit / temperature, dim=-1),
                            num_samples=1,
                        )
                    else:
                        next_token = torch.argmax(last_logit, dim=-1, keepdim=True)

                if eos_token_id is not None:
                    gen_so_far = x_accum[:, prompt_ids.size(1) :]
                    is_eos = gen_so_far == eos_token_id
                    if is_eos.any(dim=1).all():
                        first_eos = is_eos.to(torch.int64).argmax(dim=1)
                        max_eos = first_eos.max().item()
                        return x_accum[:, : prompt_ids.size(1) + max_eos + 1], nfe

            return x_accum, nfe

        @torch.no_grad()
        def infill_with_denoising_steps(
            self,
            input_ids: torch.Tensor,
            update_mask: torch.Tensor,
            denoising_steps: int,
            temperature: float = 0.0,
            threshold: float | None = None,
        ) -> tuple[torch.Tensor, int]:
            """Denoise selected positions in one bidirectional native sequence."""
            if denoising_steps < 1:
                raise ValueError("Nemotron denoising_steps must be at least 1.")
            if (
                input_ids.shape != update_mask.shape
                or update_mask.dtype is not torch.bool
            ):
                raise ValueError(
                    "Nemotron infill update_mask must be bool and match input_ids."
                )
            source = importlib.import_module(model_class.__module__)
            state = input_ids.clone()
            state[update_mask] = self.mask_token_id
            transfers_per_step = source._get_num_transfer_tokens(
                update_mask, denoising_steps
            )
            nfe = 0
            for step in range(denoising_steps):
                current_mask = update_mask & (state == self.mask_token_id)
                if not current_mask.any():
                    break
                logits = self(
                    input_ids=state, use_cache=False, use_causal_mask=False
                ).logits
                nfe += 1
                x0, transfer = source._get_transfer_index(
                    logits,
                    temperature,
                    current_mask,
                    state,
                    num_transfer_tokens=transfers_per_step[:, step],
                    threshold=threshold,
                )
                state[transfer] = x0[transfer]
            return state, nfe

    MaskAwareNemotron.__name__ = model_class.__name__
    from .cut_cross_entropy import apply_pending_nemotron_cce_patch

    apply_pending_nemotron_cce_patch(MaskAwareNemotron)
    return MaskAwareNemotron


def enable_nemotron_explicit_attention_mask(model: Any) -> None:
    """Make the native bidirectional implementation honor an explicit 4D mask.

    The native remote source sends ``None`` to its attention
    interface in diffusion mode.  Packed batches supply a complete document
    visibility mask, so route that mask through the ordinary attention branch
    for this model instance only.  No parameters or state-dict keys change.
    """
    encoder = getattr(model, "encoder", None)
    layers = getattr(encoder, "layers", None)
    if (
        layers is None
        or len(layers) == 0
        or any(not hasattr(layer, "self_attn") for layer in layers)
    ):
        raise ValueError(
            "Nemotron attention adaptation requires encoder.layers with self_attn modules."
        )
    assert encoder is not None
    if getattr(encoder, "_axolotl_explicit_mask_enabled", False):
        return
    attention_type = type(encoder.layers[0].self_attn)
    source = importlib.import_module(attention_type.__module__)

    class MaskAwareAttention(attention_type):  # type: ignore[valid-type, misc]
        def forward(
            self,
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values=None,
            cache_position=None,
            use_cache=False,
            **kwargs,
        ):
            sparse_attention = type(attention_mask).__name__ == "BlockMask"
            if sparse_attention and past_key_values is not None:
                raise ValueError(
                    "Nemotron sparse attention does not support KV caching."
                )
            if sparse_attention and self.training and self.attention_dropout:
                raise ValueError(
                    "Nemotron sparse attention requires attention_dropout=0."
                )
            if getattr(attention_mask, "dtype", None) == source.torch.bool:
                attention_mask = source.torch.where(
                    attention_mask,
                    source.torch.zeros((), device=attention_mask.device),
                    source.torch.full(
                        (),
                        source.torch.finfo(hidden_states.dtype).min,
                        device=attention_mask.device,
                    ),
                ).to(hidden_states.dtype)
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)
            query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            cos, sin = position_embeddings
            query_states, key_states = source.apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )
            query_states = query_states * source._get_llama_4_attn_scale(
                cache_position,
                self.config.rope_parameters.get("llama_4_scaling_beta"),
                self.config.rope_parameters.get("original_max_position_embeddings"),
            ).to(query_states.dtype)
            metadata = kwargs.pop("diffusion_varlen", None)
            if metadata is not None:
                if (
                    attention_mask is not None
                    or not self.diffusion_lm
                    or past_key_values is not None
                    or use_cache
                    or kwargs.get("use_causal_mask", False)
                    or self.training
                    and self.attention_dropout
                ):
                    raise ValueError(
                        "Nemotron varlen requires maskless bidirectional uncached dropout-free attention"
                    )
                from axolotl.integrations.diffusion.lm.varlen import varlen_attention

                attn_output = varlen_attention(
                    query_states, key_states, value_states, metadata, scale=self.scaling
                )
                return self.o_proj(
                    attn_output.reshape(*input_shape, -1).contiguous()
                ), None
            if past_key_values is not None:
                if use_cache:
                    key_states, value_states = past_key_values.update(
                        key_states,
                        value_states,
                        self.layer_idx,
                        {"sin": sin, "cos": cos, "cache_position": cache_position},
                    )
                else:
                    cached = past_key_values.layers[self.layer_idx]
                    key_states = source.torch.cat([cached.keys, key_states], dim=-2)
                    value_states = source.torch.cat(
                        [cached.values, value_states], dim=-2
                    )
            attention_interface = source.eager_attention_forward
            if self.config._attn_implementation != "eager":
                attention_interface = source.ALL_ATTENTION_FUNCTIONS[
                    self.config._attn_implementation
                ]
            interface_kwargs = {
                "dropout": 0.0 if not self.training else self.attention_dropout,
                "scaling": self.scaling,
                **kwargs,
            }
            if self.diffusion_lm:
                interface_kwargs["is_causal"] = False
            else:
                interface_kwargs["sliding_window"] = getattr(
                    self.config, "sliding_window", None
                )
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                **interface_kwargs,
            )
            return self.o_proj(
                attn_output.reshape(*input_shape, -1).contiguous()
            ), attn_weights

    for layer in encoder.layers:
        original = layer.self_attn
        replacement = MaskAwareAttention(original.config, original.layer_idx)
        replacement.load_state_dict(original.state_dict())
        layer.self_attn = replacement
    encoder._axolotl_explicit_mask_enabled = True
