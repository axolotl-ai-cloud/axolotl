"""Monkeypatch for Qwen3_Next model to pass position_ids to linear attention.

Qwen3-Next and Qwen3.5 share patch infrastructure but have different GatedDeltaNet
attribute layouts. This module re-exports the Qwen3-Next patcher from the shared
qwen3_5/modeling.py implementation, and also exposes granular patch functions
for individual components.
"""

import importlib

from axolotl.monkeypatch.models.qwen3_5.modeling import (
    _inject_fla_kernels,
    _make_qwen3_next_gated_delta_forward,
    _patched_decoder_forward,
    get_cu_seqlens,
    patch_qwen3_next_modeling_packing,
)

__all__ = [
    "get_cu_seqlens",
    "patch_qwen3_next_decoder_layer",
    "patch_qwen3_next_gateddelta_layer",
    "patch_qwen3_next_imports",
    "patch_qwen3_next_modeling_packing",
]

_MODULE = "transformers.models.qwen3_next.modeling_qwen3_next"


def patch_qwen3_next_decoder_layer():
    """Patch Qwen3NextDecoderLayer.forward and return an unpatch function."""
    module = importlib.import_module(_MODULE)
    cls = module.Qwen3NextDecoderLayer
    original = cls.forward
    cls.forward = _patched_decoder_forward

    def unpatch():
        cls.forward = original

    return unpatch


def patch_qwen3_next_gateddelta_layer():
    """Patch Qwen3NextGatedDeltaNet.forward and return an unpatch function."""
    module = importlib.import_module(_MODULE)
    cls = module.Qwen3NextGatedDeltaNet
    original = cls.forward
    cls.forward = _make_qwen3_next_gated_delta_forward(
        module.apply_mask_to_padding_states
    )

    def unpatch():
        cls.forward = original

    return unpatch


def patch_qwen3_next_imports():
    """Inject FLA kernels into the Qwen3Next modeling module."""
    module = importlib.import_module(_MODULE)
    _inject_fla_kernels(module)
    return None
