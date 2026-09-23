"""Sample-packing patch for the pure-SSM transformers Mamba2 model.

Its forward does not accept ``position_ids``, so the ForCausalLM wrapper turns
them into ``seq_idx`` and stashes the boundaries on every block, which passes
them to its mixer as a kwarg. The mixer forwards kwargs into its kernels, so a
live kernel that takes ``seq_idx`` is all the SSD scan needs to reset state at
each document.

Mamba1 (``mamba``, ``falcon_mamba``) is not covered: its selective scan has no
boundary argument, so those model types reject packing at config validation.
"""

import functools
import importlib

import torch

from axolotl.monkeypatch.models.mamba_utils import (
    get_seq_idx,
    kernel_accepts,
    require_seq_idx_kernels,
)
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

SEQ_IDX_KERNELS = (
    "causal_conv1d_fn",
    "mamba2_split_conv1d_scan_combined",
    "mamba2_chunk_scan",
)


def _binarize(attention_mask):
    # multipack encodes one id per document; the mixer only needs 0/1 for padding
    if attention_mask is None:
        return None
    return (attention_mask != 0).to(attention_mask.dtype)


def _patch_causal_lm(causal_lm_cls) -> None:
    """Turn ``position_ids`` into ``seq_idx`` stashed on every block.

    The blocks read the stash at call time, so a gradient-checkpointing recompute
    sees the same boundaries as the original forward.
    """
    if getattr(causal_lm_cls.forward, "_axolotl_seq_idx_patch", False):
        return
    original_forward = causal_lm_cls.forward

    @functools.wraps(original_forward)
    def patched_forward(self, *args, **kwargs):
        position_ids = kwargs.pop("position_ids", None)
        seq_idx = None
        if position_ids is not None and kwargs.get("cache_params") is None:
            seq_idx = get_seq_idx(position_ids)
            # a packed eval batch would otherwise get a fresh cache and the stock path
            kwargs["use_cache"] = False
            kwargs["attention_mask"] = _binarize(kwargs.get("attention_mask"))
        for block in self.backbone.layers:
            block._axolotl_seq_idx = seq_idx
        return original_forward(self, *args, **kwargs)

    patched_forward._axolotl_seq_idx_patch = True
    causal_lm_cls.forward = patched_forward


def _patch_block(block_cls) -> None:
    """Hand the stashed ``seq_idx`` to the mixer."""
    if getattr(block_cls.forward, "_axolotl_seq_idx_patch", False):
        return
    original_forward = block_cls.forward

    @functools.wraps(original_forward)
    def patched_forward(
        self, hidden_states, cache_params=None, attention_mask=None, **kwargs
    ):
        seq_idx = getattr(self, "_axolotl_seq_idx", None)
        if seq_idx is None or cache_params is not None:
            return original_forward(
                self,
                hidden_states,
                cache_params=cache_params,
                attention_mask=attention_mask,
                **kwargs,
            )
        residual = hidden_states
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)
        hidden_states = self.mixer(
            hidden_states,
            cache_params=cache_params,
            attention_mask=attention_mask,
            seq_idx=seq_idx,
            **kwargs,
        )
        return residual + hidden_states

    patched_forward._axolotl_seq_idx_patch = True
    block_cls.forward = patched_forward


def _assert_packed_ready(mixer, mod) -> None:
    """Fail closed before a packed batch reaches a kernel that would drop seq_idx."""
    if "cuda" not in mixer.in_proj.weight.device.type:
        raise RuntimeError(
            "mamba2 sample packing needs the CUDA kernels; the torch fallbacks "
            "have no document boundaries."
        )
    if getattr(mod, "_axolotl_seq_idx_verified", False):
        return
    for name in SEQ_IDX_KERNELS:
        if kernel_accepts(getattr(mod, name), "seq_idx") is False:
            raise RuntimeError(
                f"mamba2 sample packing: `{name}` is the transformers torch "
                "fallback, which drops seq_idx and mixes state across packed samples. "
                "Install mamba-ssm and causal-conv1d or set `use_kernels: true`."
            )
    mod._axolotl_seq_idx_verified = True


def _patch_mixer(mod, mixer_cls) -> None:
    if getattr(mixer_cls.forward, "_axolotl_seq_idx_patch", False):
        return
    original_forward = mixer_cls.forward

    @functools.wraps(original_forward)
    def patched_forward(self, hidden_states, *args, seq_idx=None, **kwargs):
        if seq_idx is not None and kwargs.get("cache_params") is None:
            _assert_packed_ready(self, mod)
            kwargs["seq_idx"] = seq_idx
        return original_forward(self, hidden_states, *args, **kwargs)

    patched_forward._axolotl_seq_idx_patch = True
    mixer_cls.forward = patched_forward


def patch_mamba2_modeling_packing(kernels_enabled: bool = False) -> None:
    try:
        mod = importlib.import_module("transformers.models.mamba2.modeling_mamba2")
    except ImportError:
        LOG.warning("mamba2 not found in transformers, skipping packing patches")
        return
    require_seq_idx_kernels(mod, SEQ_IDX_KERNELS, "mamba2", kernels_enabled)

    _patch_causal_lm(mod.Mamba2ForCausalLM)
    _patch_block(mod.Mamba2Block)
    _patch_mixer(mod, mod.Mamba2Mixer)

    LOG.info("Applied Mamba2 sample packing patch (seq_idx threading into the SSM)")
