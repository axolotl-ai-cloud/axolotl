"""
TiledMLP monkey patches
"""

from .patch import (
    patch_tiled_mlp,
    patch_tiled_mlp_deepspeed,
)

__all__ = [
    "patch_tiled_mlp_deepspeed",
    "patch_tiled_mlp",
]
