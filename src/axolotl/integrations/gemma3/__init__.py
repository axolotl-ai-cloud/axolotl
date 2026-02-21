"""Gemma3 integration for loading multimodal checkpoints as text-only models."""

from .args import Gemma3TextFromMultimodalArgs
from .plugin import Gemma3TextFromMultimodalPlugin

__all__ = [
    "Gemma3TextFromMultimodalArgs",
    "Gemma3TextFromMultimodalPlugin",
]
