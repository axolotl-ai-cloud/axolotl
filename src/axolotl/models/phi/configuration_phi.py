# pylint: skip-file
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
from typing import Optional

from transformers import PretrainedConfig


class PhiConfig(PretrainedConfig):
    """Phi configuration."""

    model_type = "phi"
    attribute_map = {
        "max_position_embeddings": "n_positions",
        "hidden_size": "n_embd",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
        self,
        vocab_size: int = 50304,
        n_positions: int = 2048,
        n_embd: int = 1024,
        n_layer: int = 20,
        n_inner: Optional[int] = None,
        n_head: int = 16,
        n_head_kv: Optional[int] = None,
        rotary_dim: Optional[int] = 32,
        activation_function: Optional[str] = "gelu_new",
        flash_attn: bool = False,
        flash_rotary: bool = False,
        fused_dense: bool = False,
        attn_pdrop: float = 0.0,
        embd_pdrop: float = 0.0,
        resid_pdrop: float = 0.0,
        layer_norm_epsilon: float = 1e-5,
        initializer_range: float = 0.02,
        tie_word_embeddings: bool = False,
        pad_vocab_size_multiple: int = 64,
        **kwargs
    ) -> None:
        self.vocab_size = int(
            math.ceil(vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple
        )
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_inner = n_inner
        self.n_head = n_head
        self.n_head_kv = n_head_kv
        self.rotary_dim = min(rotary_dim, n_embd // n_head)
        self.activation_function = activation_function
        self.flash_attn = flash_attn
        self.flash_rotary = flash_rotary
        self.fused_dense = fused_dense
        self.attn_pdrop = attn_pdrop
        self.embd_pdrop = embd_pdrop
        self.resid_pdrop = resid_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
