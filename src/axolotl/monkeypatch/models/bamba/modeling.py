"""Sample-packing and context-parallelism patch for Bamba (Mamba2/Attention hybrid).

Upstream BambaMixer already accepts seq_idx on forward/cuda_kernels_forward,
but the BambaDecoderLayer does not compute it from position_ids.  This patch:
1. Derives seq_idx via get_seq_idx(position_ids) in the decoder layer.
2. Forces the slow path when CP is active (the fused path doesn't return SSM
   state).  CP correction itself is handled by ``wrap_mamba_scan_for_cp``.
"""

import importlib

from axolotl.monkeypatch.models.mamba_utils import (
    get_seq_idx,  # noqa: F401
    is_cp_active,
    wrap_mamba_scan_for_cp,
)
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def patch_bamba_modeling_packing():
    """Patch Bamba for sample packing: compute seq_idx and add CP correction."""
    try:
        mod = importlib.import_module("transformers.models.bamba.modeling_bamba")
    except ImportError:
        LOG.warning("bamba not found in transformers, skipping packing patches")
        return

    BambaMixer = mod.BambaMixer
    BambaDecoderLayer = mod.BambaDecoderLayer

    original_cuda_kernels_forward = BambaMixer.cuda_kernels_forward

    def patched_cuda_kernels_forward(
        self,
        hidden_states,
        cache_params=None,
        cache_position=None,
        attention_mask=None,
        seq_idx=None,
    ):
        # Upstream's fused path (mamba_split_conv1d_scan_combined) doesn't
        # return SSM state, so CP correction can't run.  Temporarily clear
        # self.training to make the upstream condition
        # ``if self.training and cache_params is None`` fall through to the
        # slow path which calls mamba_chunk_scan_combined (wrapped for CP).
        force_slow = is_cp_active() and self.training and cache_params is None
        if force_slow:
            self.training = False
        try:
            return original_cuda_kernels_forward(
                self,
                hidden_states,
                cache_params,
                cache_position,
                attention_mask,
                seq_idx,
            )
        finally:
            if force_slow:
                self.training = True

    BambaMixer.cuda_kernels_forward = patched_cuda_kernels_forward

    def patched_decoder_forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "mamba":
            is_decoding = (
                past_key_values is not None and past_key_values.has_previous_state
            )
            if position_ids is not None and not is_decoding:
                kwargs["seq_idx"] = get_seq_idx(position_ids)

            hidden_states = self.mamba(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                cache_position=cache_position,
                attention_mask=attention_mask,
                **kwargs,
            )
            self_attn_weights = None
        elif self.layer_type == "attention":
            hidden_states, self_attn_weights = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_ff_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        return outputs

    BambaDecoderLayer.forward = patched_decoder_forward

    wrap_mamba_scan_for_cp(mod)

    LOG.info("Applied Bamba sample packing patch (seq_idx + CP correction)")
