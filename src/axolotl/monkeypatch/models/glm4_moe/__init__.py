"""
Grouped MoE patches for GLM4 MoE architectures.
"""

from .modeling import patch_glm4_moe_grouped_experts

__all__ = ["patch_glm4_moe_grouped_experts"]
