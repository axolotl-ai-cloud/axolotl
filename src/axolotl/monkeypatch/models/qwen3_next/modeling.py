"""Monkeypatch for Qwen3_Next model to pass position_ids to linear attention."""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


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
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                cache_position=cache_position,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        elif self.layer_type == "full_attention":
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
        from transformers.models.qwen3_next.modeling_qwen3_next import (
            Qwen3NextDynamicCache,
            Qwen3NextGatedDeltaNet,
            apply_mask_to_padding_states,
        )
    except ImportError:
        LOG.warning("Qwen3Next model not found, skipping patch")
        return

    # Store original forward method
    original_gated_delta_net_forward = Qwen3NextGatedDeltaNet.forward

    def patched_gated_delta_net_forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Optional[Qwen3NextDynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ):
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

        # Set up dimensions for reshapes later
        batch_size, seq_len, _ = hidden_states.shape

        use_precomputed_states = (
            cache_params is not None
            and cache_params.has_previous_state
            and seq_len == 1
            and cache_position is not None
        )

        # getting projected states from cache if it exists
        if cache_params is not None:
            conv_state = cache_params.conv_states[self.layer_idx]
            recurrent_state = cache_params.recurrent_states[self.layer_idx]

        projected_states_qkvz = self.in_proj_qkvz(hidden_states)
        projected_states_ba = self.in_proj_ba(hidden_states)
        query, key, value, z, b, a = self.fix_query_key_value_ordering(
            projected_states_qkvz, projected_states_ba
        )
        query, key, value = (
            x.reshape(x.shape[0], x.shape[1], -1) for x in (query, key, value)
        )

        mixed_qkv = torch.cat((query, key, value), dim=-1)
        mixed_qkv = mixed_qkv.transpose(1, 2)

        if use_precomputed_states:
            # 2. Convolution sequence transformation
            # NOTE: the conv state is updated in `causal_conv1d_update`
            mixed_qkv = self.causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )
        else:
            if cache_params is not None:
                conv_state = F.pad(
                    mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0)
                )
                cache_params.conv_states[self.layer_idx] = conv_state
            if self.causal_conv1d_fn is not None:
                mixed_qkv = self.causal_conv1d_fn(
                    x=mixed_qkv,
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=None,
                )
            else:
                mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

        mixed_qkv = mixed_qkv.transpose(1, 2)
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

        if not use_precomputed_states:
            cu_seqlens = get_cu_seqlens(position_ids=position_ids)
            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )

        else:
            core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
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
            cache_params.recurrent_states[self.layer_idx] = last_recurrent_state

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
    Qwen3NextGatedDeltaNet.forward = patched_gated_delta_net_forward

    def unpatch():
        """Restore the original forward method"""
        Qwen3NextGatedDeltaNet.forward = original_gated_delta_net_forward

    return unpatch


def patch_qwen3_next_imports():
    """Patch Qwen3Next imports to use try/except instead of is_flash_linear_attention_available."""
    try:
        import transformers.models.qwen3_next.modeling_qwen3_next as qwen3_modeling
    except ImportError:
        LOG.warning("Qwen3Next model not found, skipping import patch")
        return

    # Save original values for unpatch
    original_FusedRMSNormGated = getattr(qwen3_modeling, "FusedRMSNormGated", None)
    original_chunk_gated_delta_rule = getattr(
        qwen3_modeling, "chunk_gated_delta_rule", None
    )
    original_fused_recurrent_gated_delta_rule = getattr(
        qwen3_modeling, "fused_recurrent_gated_delta_rule", None
    )
    original_is_fast_path_available = getattr(
        qwen3_modeling, "is_fast_path_available", False
    )

    try:
        from fla.modules import FusedRMSNormGated
        from fla.ops.gated_delta_rule import (
            chunk_gated_delta_rule,
            fused_recurrent_gated_delta_rule,
        )

        qwen3_modeling.FusedRMSNormGated = FusedRMSNormGated
        qwen3_modeling.chunk_gated_delta_rule = chunk_gated_delta_rule
        qwen3_modeling.fused_recurrent_gated_delta_rule = (
            fused_recurrent_gated_delta_rule
        )

        # Force is_fast_path_available to be True
        # fla has triton kernels for causal_conv1d
        qwen3_modeling.is_fast_path_available = True
    except ImportError:
        qwen3_modeling.chunk_gated_delta_rule = None
        qwen3_modeling.fused_recurrent_gated_delta_rule = None
        qwen3_modeling.FusedRMSNormGated = None

    def unpatch():
        """Restore the original import values"""
        qwen3_modeling.FusedRMSNormGated = original_FusedRMSNormGated
        qwen3_modeling.chunk_gated_delta_rule = original_chunk_gated_delta_rule
        qwen3_modeling.fused_recurrent_gated_delta_rule = (
            original_fused_recurrent_gated_delta_rule
        )
        qwen3_modeling.is_fast_path_available = original_is_fast_path_available

    return unpatch


def patch_qwen3_next_modeling_packing():
    """Apply all Qwen3Next model patches."""
    patch_qwen3_next_imports()
    patch_qwen3_next_decoder_layer()
    patch_qwen3_next_gateddelta_layer()

    LOG.info("Applied Qwen3Next patch for packing")
