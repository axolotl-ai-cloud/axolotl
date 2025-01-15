"""
Rala config class
"""
from transformers import LlamaConfig


class LlamaRalaConfig(LlamaConfig):
    """
    Configuration for LlamaRala model
    """

    softmax_every: int = 6  # every 8th layer applies softmax
