"""Compatibility exports for diffusion sample generation."""

from axolotl.core.trainers.diffusion_lm.generation import (
    _clean_masked_text,
    _diffusion_step,
    _sample_sequences_from_dataloader,
    generate,
    generate_samples,
)

__all__ = [
    "generate_samples",
    "generate",
    "_sample_sequences_from_dataloader",
    "_clean_masked_text",
    "_diffusion_step",
]
