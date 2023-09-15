# pylint: skip-file

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
from typing import Any, Dict, List, Optional, Union

from transformers import PretrainedConfig


class MixFormerSequentialConfig(PretrainedConfig):
    """MixFormer (sequential for DeepSpeed) configuration."""

    model_type = "mixformer-sequential"

    attribute_map = {
        "max_position_embeddings": "n_positions",
        "hidden_size": "n_embd",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
        "input_emb_layer": "embd_layer",  # `input_emb_layer` key is for backward compatibility
        "blocks": "architecture",  # `blocks` key is for backward compatibility
    }

    def __init__(
        self,
        vocab_size: Optional[int] = 50304,
        n_positions: Optional[int] = 2048,
        n_embd: Optional[int] = 1024,
        n_layer: Optional[int] = 20,
        n_inner: Optional[int] = None,
        n_head: Optional[int] = 16,
        rotary_dim: Optional[int] = 32,
        activation_function: Optional[str] = "gelu_new",
        embd_layer: Optional[str] = "default",
        architecture: Union[Dict[str, Any], List[Dict[str, Any]]] = None,
        embd_pdrop: Optional[float] = 0.0,
        resid_pdrop: Optional[float] = 0.0,
        layer_norm_epsilon: Optional[float] = 1e-5,
        initializer_range: Optional[float] = 0.02,
        tie_word_embeddings: Optional[bool] = False,
        pad_vocab_size_multiple: Optional[int] = 64,
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
        self.rotary_dim = min(rotary_dim, n_embd // n_head)
        self.activation_function = activation_function
        self.embd_layer = embd_layer
        self.architecture = architecture
        self.embd_pdrop = embd_pdrop
        self.resid_pdrop = resid_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
