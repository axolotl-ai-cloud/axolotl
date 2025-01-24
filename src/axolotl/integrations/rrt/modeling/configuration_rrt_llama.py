"""
module for custom configuration for relaxed recursive transformers model
"""
from transformers import LlamaConfig


class RelaxedRecursiveLlamaConfig(LlamaConfig):
    """
    Configuration for Relaxed Recursive Llama.
    """

    model_type: str = "llama-rrt"
    recurse_layers: int = 4
    rank: int
    alpha: int
    use_dora: bool = True
