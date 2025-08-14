"""
Diffusion LM training plugin for Axolotl.

This plugin enables diffusion language model training using the LLaDA approach.
"""

from .args import DiffusionArgs
from .plugin import DiffusionPlugin

__all__ = ["DiffusionArgs", "DiffusionPlugin"]
