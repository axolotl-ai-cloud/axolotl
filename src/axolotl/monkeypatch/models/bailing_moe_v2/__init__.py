"""Grouped MoE patches for Ring/Bailing (BailingMoeV2) architectures."""

from .modeling import patch_model_with_grouped_experts

__all__ = ["patch_model_with_grouped_experts"]

