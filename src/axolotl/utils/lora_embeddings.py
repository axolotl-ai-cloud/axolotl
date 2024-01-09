"""
helpers for lora embeddings
"""


def get_linear_embedding_layers(model_type):
    """
    returns the linear embedding layers needed for loras, dependent on the model arch
    """
    if model_type == "phi-msft":
        return ["embd", "lm_head.linear"]
    return ["lm_head", "embed_tokens"]
