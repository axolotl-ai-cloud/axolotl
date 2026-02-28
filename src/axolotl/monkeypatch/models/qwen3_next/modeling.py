"""Monkeypatch for Qwen3_Next model to pass position_ids to linear attention.

Qwen3-Next and Qwen3.5 share patch infrastructure but have different GatedDeltaNet
attribute layouts. This module re-exports the Qwen3-Next patcher from the shared
qwen3_5/modeling.py implementation.
"""

from axolotl.monkeypatch.models.qwen3_5.modeling import (
    patch_qwen3_next_modeling_packing,
)

__all__ = ["patch_qwen3_next_modeling_packing"]
