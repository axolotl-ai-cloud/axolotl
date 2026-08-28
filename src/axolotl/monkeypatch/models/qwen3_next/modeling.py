"""Monkeypatch for Qwen3_Next model to pass position_ids to linear attention."""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from transformers.integrations.accelerate import force_accelerate_hooks

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

try:
    from fla.modules.convolution import causal_conv1d as fla_causal_conv1d
except ImportError:
    fla_causal_conv1d = None

try:
    from fla.ops.gated_delta_rule import (
        chunk_gated_delta_rule as fla_chunk_gated_delta_rule,
    )
except ImportError:
    fla_chunk_gated_delta_rule = None


def get_cu_seqlens(position_ids):
    """
    Adapted from transformers.modeling_flash_attention_utils.prepare_fa_kwargs_from_position_ids.

    https://github.com/huggingface/transformers/blob/0f1b128d3359a26bd18be99c26d7f04fb3cba914/src/transformers/modeling_flash_attention_utils.py#L316
    """
    tensor_kwargs = {"dtype": torch.int32, "device": position_ids.device}

    position_ids = position_ids.view(-1)
    indices_q = (position_ids == 0).nonzero().view(-1)

    cu_seq_lens_q = torch.cat(
        (
            indices_q.to(**tensor_kwargs),
            torch.tensor(position_ids.size(), **tensor_kwargs),
        )
    )

    return cu_seq_lens_q


def patch_qwen3_next_decoder_layer():
    """Patch Qwen3NextDecoderLayer to pass position_ids to linear attention."""
    try:
        from transformers.models.qwen3_next.modeling_qwen3_next import (
            Qwen3NextDecoderLayer,
        )
    except ImportError:
        LOG.warning("Qwen3Next model not found, skipping patch")
        return

    # Store original forward method
    original_decoder_forward = Qwen3NextDecoderLayer.forward

    def patched_decoder_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Token Mixer
        if self.block_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                cache_position=cache_position,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        elif self.block_type == "full_attention":
            # Self Attention
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # For the MoE layers, we need to unpack
        if isinstance(hidden_states, Tuple):
            hidden_states, _ = hidden_states
        hidden_states = residual + hidden_states

        return hidden_states

    # Apply the patches
    Qwen3NextDecoderLayer.forward = patched_decoder_forward

    def unpatch():
        """Restore the original forward method"""
        Qwen3NextDecoderLayer.forward = original_decoder_forward

    return unpatch


def patch_qwen3_next_gateddelta_layer():
    """Patch Qwen3NextGatedDeltaNet to parse cu_seqlens and pass to chunk_gated_delta_rule"""
    try:
        from transformers.models.qwen3_next import modeling_qwen3_next as modeling
    except ImportError:
        LOG.warning("Qwen3Next model not found, skipping patch")
        return None

    # Store original forward method
    original_gated_delta_net_forward = modeling.Qwen3NextGatedDeltaNet.forward

    @force_accelerate_hooks("conv1d")
    def patched_gated_delta_net_forward(
        self,
        hidden_states: torch.Tensor,
        cache_params=None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[
            torch.LongTensor
        ] = None,  # unused: no cache in packed training
        **kwargs,
    ):
        hidden_states = modeling.apply_mask_to_padding_states(
            hidden_states, attention_mask
        )

        # Set up dimensions for reshapes later
        seq_len = hidden_states.shape[1]

        use_precomputed_states = (
            cache_params is not None and cache_params.has_previous_state(self.layer_idx)
        )

        # Compute cu_seqlens early for use by both causal_conv1d and chunk_gated_delta_rule
        cu_seqlens = None
        if not use_precomputed_states and position_ids is not None:
            cu_seqlens = get_cu_seqlens(position_ids=position_ids)

        if cu_seqlens is not None and (
            fla_causal_conv1d is None or fla_chunk_gated_delta_rule is None
        ):
            # the transformers fallbacks accept cu_seqlens but ignore it, which would
            # silently mix tokens across packed samples
            raise RuntimeError(
                "Packed sequences require flash-linear-attention (cu_seqlens support "
                "in causal_conv1d and chunk_gated_delta_rule). Install "
                "flash-linear-attention or disable packing."
            )

        projected_states_qkvz = self.in_proj_qkvz(hidden_states)
        projected_states_ba = self.in_proj_ba(hidden_states)
        query, key, value, z, b, a = self.fix_query_key_value_ordering(
            projected_states_qkvz, projected_states_ba
        )
        query, key, value = (
            x.reshape(x.shape[0], x.shape[1], -1) for x in (query, key, value)
        )

        mixed_qkv = torch.cat((query, key, value), dim=-1)  # [B, T, D]
        mixed_qkv = mixed_qkv.transpose(1, 2)  # [B, D, T]

        if (
            use_precomputed_states
            and seq_len == 1
            and not cache_params.layers[self.layer_idx].record_past
        ):
            conv_state = cache_params.layers[self.layer_idx].conv_states[0]
            mixed_qkv = modeling.causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )
        elif cu_seqlens is not None:
            if cache_params is not None:
                cache_params.update_conv_state(
                    mixed_qkv, self.layer_idx, conv_kernel_size=self.conv_kernel_size
                )
            # FLA Triton causal_conv1d: [B, T, D] in/out, with cu_seqlens support
            mixed_qkv = mixed_qkv.transpose(1, 2)
            mixed_qkv, _ = fla_causal_conv1d(
                x=mixed_qkv,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
                cu_seqlens=cu_seqlens,
            )
            mixed_qkv = mixed_qkv.transpose(1, 2)  # back to [B, D, T]
        else:
            if cache_params is not None:
                mixed_qkv = cache_params.update_conv_state(
                    mixed_qkv, self.layer_idx, conv_kernel_size=self.conv_kernel_size
                )

            mixed_qkv = modeling.causal_conv1d_fn(
                mixed_qkv,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                activation=self.activation,
            )

            # Drop the additional previous states
            if cache_params is not None:
                mixed_qkv = mixed_qkv[:, :, -seq_len:]

        mixed_qkv = mixed_qkv.transpose(1, 2)  # [B, T, D]
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim,
                self.key_dim,
                self.value_dim,
            ],
            dim=-1,
        )
        query = query.reshape(query.shape[0], query.shape[1], -1, self.head_k_dim)
        key = key.reshape(key.shape[0], key.shape[1], -1, self.head_k_dim)
        value = value.reshape(value.shape[0], value.shape[1], -1, self.head_v_dim)

        beta = b.sigmoid()
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        recurrent_state = (
            cache_params.layers[self.layer_idx].recurrent_states[0]
            if use_precomputed_states
            else None
        )
        if use_precomputed_states and seq_len == 1:
            core_attn_out, last_recurrent_state = (
                modeling.torch_recurrent_gated_delta_rule(
                    query,
                    key,
                    value,
                    g=g,
                    beta=beta,
                    initial_state=recurrent_state,
                    output_final_state=cache_params is not None,
                    use_qk_l2norm_in_kernel=True,
                )
            )
        elif cu_seqlens is not None:
            core_attn_out, last_recurrent_state = fla_chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
        else:
            core_attn_out, last_recurrent_state = modeling.torch_chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )

        # Update cache
        if cache_params is not None:
            cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

        z_shape_og = z.shape
        # reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(
            core_attn_out.shape[0], core_attn_out.shape[1], -1
        )

        output = self.out_proj(core_attn_out)
        return output

    # Apply the patches
    modeling.Qwen3NextGatedDeltaNet.forward = patched_gated_delta_net_forward

    def unpatch():
        """Restore the original forward method"""
        modeling.Qwen3NextGatedDeltaNet.forward = original_gated_delta_net_forward

    return unpatch


def patch_qwen3_next_modeling_packing():
    """Apply all Qwen3Next model patches."""
    patch_qwen3_next_decoder_layer()
    patch_qwen3_next_gateddelta_layer()

    LOG.info("Applied Qwen3Next patch for packing")
