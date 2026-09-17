"""
TiledMLP monkey patches
"""

from .patch import (
    patch_tiled_mlp,
    patch_tiled_mlp_moe_instances,
)

__all__ = [
    "patch_tiled_mlp",
    "patch_tiled_mlp_moe_instances",
]
