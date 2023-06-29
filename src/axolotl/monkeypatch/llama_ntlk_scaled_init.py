# pylint: skip-file
"""
Non-linear interpolation scheme that adjusts the RoPE's base. Adapted from https://twitter.com/yampeleg/status/1674430869828956161
"""
import transformers

old_init = transformers.models.llama.modeling_llama.LlamaRotaryEmbedding.__init__


def ntk_scaled_init(self, dim, max_position_embeddings=2048, base=10000, device=None):
    # The method is just these three lines
    max_position_embeddings = 16384
    a = 8  # Alpha value
    base = base * a ** (dim / (dim - 2))  # Base change formula

    old_init(self, dim, max_position_embeddings, base, device)


def replace_llama_rope_init_with_ntlk_scaled_init():
    transformers.models.llama.modeling_llama.LlamaRotaryEmbedding.__init__ = (
        ntk_scaled_init
    )
