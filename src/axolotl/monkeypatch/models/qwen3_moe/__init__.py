"""Grouped MoE monkeypatches for Qwen3 MoE architectures."""

from .modeling import patch_qwen3_moe_grouped_experts

__all__ = ["patch_qwen3_moe_grouped_experts"]
