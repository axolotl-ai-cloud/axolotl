from transformers import LlamaConfig


class RelaxedRecursiveLlamaConfig(LlamaConfig):
    """
    Configuration for Relaxed Recursive Llama.
    """

    model_type = "llama-rrt"
    recurse_layers: int  = 4
    rank: int
    alpha: int
    use_dora: bool = True
