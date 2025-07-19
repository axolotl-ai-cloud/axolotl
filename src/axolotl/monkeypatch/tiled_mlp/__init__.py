"""
TiledMLP monkey patches
"""

from .patch import (
    patch_tiled_mlp_deepspeed,
    patch_tiled_mlp_fsdp,
)

__all__ = [
    "patch_tiled_mlp_deepspeed",
    "patch_tiled_mlp_fsdp",
]
