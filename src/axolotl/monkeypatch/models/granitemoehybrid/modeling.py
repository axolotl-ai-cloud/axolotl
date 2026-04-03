"""Sample-packing and context-parallelism patch for Granite MoE Hybrid (Mamba2/Attention/MoE).

Upstream GraniteMoeHybridMambaLayer already accepts seq_idx on
forward/cuda_kernels_forward, and GraniteMoeHybridDecoderLayer passes **kwargs
through to the mixer.  However, the decoder layer does not receive position_ids
directly — it arrives at the model level.

This patch:
1. Injects seq_idx computation into GraniteMoeHybridModel.forward so it flows
   through kwargs -> decoder_layer -> mamba mixer automatically.
2. Forces the slow path when CP is active (the fused path doesn't return SSM
   state).  CP correction is handled by ``wrap_mamba_scan_for_cp``.
"""

import importlib

from axolotl.monkeypatch.models.mamba_utils import (
    ensure_mamba_kernels_loaded,
    get_seq_idx,  # noqa: F401
    is_cp_active,
    wrap_mamba_scan_for_cp,
)
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def patch_granitemoehybrid_modeling_packing():
    """Patch Granite MoE Hybrid for sample packing: seq_idx + CP correction."""
    try:
        mod = importlib.import_module(
            "transformers.models.granitemoehybrid.modeling_granitemoehybrid"
        )
    except ImportError:
        LOG.warning(
            "granitemoehybrid not found in transformers, skipping packing patches"
        )
        return

    ensure_mamba_kernels_loaded(mod)

    GraniteMoeHybridModel = mod.GraniteMoeHybridModel
    GraniteMoeHybridMambaLayer = mod.GraniteMoeHybridMambaLayer

    # Patch 1: Model-level seq_idx injection
    original_model_forward = GraniteMoeHybridModel.forward

    def patched_model_forward(self, *args, **kwargs):
        position_ids = kwargs.get("position_ids")
        if position_ids is None and len(args) > 2:
            position_ids = args[2]

        past_key_values = kwargs.get("past_key_values")
        if past_key_values is None and len(args) > 3:
            past_key_values = args[3]

        is_decoding = (
            past_key_values is not None
            and hasattr(past_key_values, "has_previous_state")
            and past_key_values.has_previous_state
        )

        if position_ids is not None and not is_decoding and "seq_idx" not in kwargs:
            kwargs["seq_idx"] = get_seq_idx(position_ids)

        return original_model_forward(self, *args, **kwargs)

    GraniteMoeHybridModel.forward = patched_model_forward

    # Patch 2: Minimal wrapper to force slow path when CP is active.
    # The fused mamba_split_conv1d_scan_combined doesn't return SSM state, so
    # CP correction (handled by the scan wrapper) needs the slow path.
    original_cuda_kernels_forward = GraniteMoeHybridMambaLayer.cuda_kernels_forward

    def patched_cuda_kernels_forward(
        self,
        hidden_states,
        cache_params=None,
        attention_mask=None,
        seq_idx=None,
    ):
        force_slow = (
            (seq_idx is not None or is_cp_active())
            and self.training
            and cache_params is None
        )
        if force_slow:
            self.training = False
        try:
            return original_cuda_kernels_forward(
                self,
                hidden_states,
                cache_params,
                attention_mask,
                seq_idx,
            )
        finally:
            if force_slow:
                self.training = True

    GraniteMoeHybridMambaLayer.cuda_kernels_forward = patched_cuda_kernels_forward

    wrap_mamba_scan_for_cp(mod)

    LOG.info(
        "Applied Granite MoE Hybrid sample packing patch (seq_idx + CP correction)"
    )
