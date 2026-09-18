"""Sample-packing and context-parallelism patch for NemotronH (Mamba2/Attention/MoE hybrid).

Threads seq_idx (derived from position_ids) into the Mamba2 SSM kernels so
packed-sequence boundaries reset SSM state. Upstream never passes one, which
leaks hidden state across boundaries. Attention and MoE blocks need no changes
— transformers builds block-diagonal masks from position_ids for attention.

CP correction (ring-shift of SSM state + additive output fix) is handled by
``wrap_mamba_scan_for_cp`` from ``mamba_utils``, which wraps the chunk-scan
call at the module level.
"""

import functools
import importlib

from axolotl.monkeypatch.models.mamba_utils import (
    assert_mamba2_scan_honours_seq_idx,
    get_seq_idx,
    is_cp_active,
    mamba2_seq_idx_kernels_available,
    wrap_mamba_scan_for_cp,
)
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def _is_quantized(weight) -> bool:
    return weight is not None and (
        not weight.is_floating_point()
        or getattr(weight, "quant_state", None) is not None
    )


def guard_nemotron_h_fused_scan(mod=None):
    """Skip the fused Mamba2 training kernel where its assumptions break.

    ``mamba_split_conv1d_scan_combined`` multiplies by ``out_proj.weight``
    itself, which fails on a bitsandbytes-quantized weight, and it never
    surfaces the final SSM state that the CP correction needs. Returning
    ``None`` is upstream's own "kernel unavailable" signal, so the mixer falls
    back to the separate conv / chunk-scan calls.
    """
    if mod is None:
        mod = _import_modeling()
        if mod is None:
            return

    if getattr(mod, "_axolotl_fused_scan_guard", False):
        return

    original_fused = mod.mamba2_split_conv1d_scan_combined

    @functools.wraps(original_fused)
    def guarded_fused(*args, **kwargs):
        if is_cp_active() or _is_quantized(kwargs.get("outproj_weight")):
            return None
        return original_fused(*args, **kwargs)

    mod.mamba2_split_conv1d_scan_combined = guarded_fused
    mod._axolotl_fused_scan_guard = True


def _import_modeling():
    try:
        return importlib.import_module(
            "transformers.models.nemotron_h.modeling_nemotron_h"
        )
    except ImportError:
        LOG.warning("nemotron_h not found in transformers, skipping patches")
        return None


def patch_nemotron_h_modeling_packing(kernels_enabled: bool = False):
    """Patch NemotronH for sample packing: seq_idx threading into Mamba2 SSM kernels.

    _get_unpad_data is handled by SUPPORTED_MULTIPACK_MODEL_TYPES / patch_for_multipack().
    This function only applies the seq_idx patches that are unique to nemotron_h.

    ``kernels_enabled`` mirrors ``use_kernels``: transformers then kernelizes the
    mixer with the Hub Mamba2 kernels, which take seq_idx like the pip ones.
    """
    mod = _import_modeling()
    if mod is None:
        return

    if not (kernels_enabled or mamba2_seq_idx_kernels_available()):
        raise RuntimeError(
            "Nemotron-H sample packing / context parallelism requires Mamba2 "
            "kernels: the transformers torch fallbacks drop the seq_idx argument, "
            "which silently mixes SSM state across packed samples. Either install "
            "them (`pip install mamba-ssm causal-conv1d`) or set `use_kernels: "
            "true` to use the prebuilt Hub kernels."
        )

    NemotronHBlock = mod.NemotronHBlock

    if getattr(NemotronHBlock.forward, "_axolotl_seq_idx_patch", False):
        return

    def patched_block_forward(
        self,
        hidden_states,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
        use_cache=False,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))

        if self.block_type == "linear_attention":
            is_decoding = (
                past_key_values is not None
                and past_key_values.has_previous_state(self.layer_idx)
            )
            seq_idx = (
                get_seq_idx(position_ids)
                if position_ids is not None and not is_decoding
                else None
            )
            if seq_idx is not None:
                # which scan implementation is live is only settled once the
                # model is built, so the check happens on the first packed batch
                assert_mamba2_scan_honours_seq_idx(mod)
            hidden_states = self.mixer(
                hidden_states,
                cache_params=past_key_values,
                attention_mask=attention_mask,
                seq_idx=seq_idx,
            )
        elif self.block_type == "full_attention":
            hidden_states, _ = self.mixer(
                hidden_states=hidden_states,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=use_cache,
                **kwargs,
            )
        else:
            hidden_states = self.mixer(hidden_states)

        hidden_states = residual + hidden_states
        return hidden_states

    patched_block_forward._axolotl_seq_idx_patch = True
    NemotronHBlock.forward = patched_block_forward

    guard_nemotron_h_fused_scan(mod)
    wrap_mamba_scan_for_cp(mod)

    LOG.info("Applied NemotronH sample packing patch (seq_idx threading into Mamba2)")
