"""Block-diffusion training plugin for Google's DiffusionGemma."""

from .args import DiffusionGemmaArgs
from .plugin import DiffusionGemmaPlugin

__all__ = [
    "DiffusionGemmaArgs",
    "DiffusionGemmaPlugin",
]
