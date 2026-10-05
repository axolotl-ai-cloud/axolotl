"""Compatibility exports for diffusion generation callbacks."""

from axolotl.core.trainers.diffusion_lm.callbacks import DiffusionGenerationCallback
from axolotl.core.trainers.diffusion_lm.generation import generate_samples

__all__ = ["DiffusionGenerationCallback", "generate_samples"]
