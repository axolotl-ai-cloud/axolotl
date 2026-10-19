"""Sample-packing patch for Granite MoE Hybrid (Mamba2/Attention/MoE).

Upstream GraniteMoeHybridMambaLayer already accepts seq_idx on
forward/cuda_kernels_forward, and GraniteMoeHybridDecoderLayer passes **kwargs
through to the mixer.  However, the decoder layer does not receive position_ids
directly — it arrives at the model level.

This patch:
1. Injects seq_idx computation into GraniteMoeHybridModel.forward so it flows
   through kwargs -> decoder_layer -> mamba mixer automatically.
2. Forces the slow path when CP is active (the fused path doesn't return SSM
   state).
"""

import importlib

from axolotl.monkeypatch.models.mamba_utils import (
    ensure_mamba_kernels_loaded,
    is_cp_active,
    patch_model_forward_seq_idx,
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

    patch_model_forward_seq_idx(GraniteMoeHybridModel)

    # Minimal wrapper to force slow path when CP is active.
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

    LOG.info("Applied Granite MoE Hybrid sample packing patch (seq_idx)")
