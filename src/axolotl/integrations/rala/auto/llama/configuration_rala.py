"""
Rala config class
"""
from transformers import LlamaConfig


class LlamaRalaConfig(LlamaConfig):
    """
    Configuration for LlamaRala model
    """

    softmax_every: int = 6  # every N-th layer applies softmax
