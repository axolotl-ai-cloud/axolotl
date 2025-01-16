"""
Rala config class
"""
from transformers import LlamaConfig


class LlamaRalaConfig(LlamaConfig):
    """
    Configuration for LlamaRala model
    """

    model_type = "llama-rala"
    softmax_every: int = 6  # every N-th layer applies softmax
