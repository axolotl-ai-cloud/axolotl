"""Nemotron Labs Diffusion VLM image-token constants for the light canvas path.

Mirrors ``axolotl.processing_strategies`` (a test asserts they agree), which imports
transformers at module load; canvas building and validation need only these ids.
"""

from __future__ import annotations

from collections.abc import Sequence

NEMOTRON_VLM_IMAGE_START_ID = 18
NEMOTRON_VLM_IMAGE_PAD_ID = 19
NEMOTRON_VLM_IMAGE_BREAK_ID = 20
NEMOTRON_VLM_IMAGE_END_ID = 21
NEMOTRON_VLM_IMAGE_START_MARKER = "<|image_start|>"
NEMOTRON_VLM_TOKEN_SIDE = 28


def merged_patch_count(size: tuple[int, int] | Sequence[int]) -> int:
    """Image-pad tokens (model features) for a resized ``(height, width)``."""
    height, width = (int(value) for value in size)
    if (
        height <= 0
        or width <= 0
        or height % NEMOTRON_VLM_TOKEN_SIDE
        or width % NEMOTRON_VLM_TOKEN_SIDE
    ):
        raise ValueError(
            f"image_sizes must be positive multiples of {NEMOTRON_VLM_TOKEN_SIDE}"
        )
    return (height // NEMOTRON_VLM_TOKEN_SIDE) * (width // NEMOTRON_VLM_TOKEN_SIDE)
