"""
Fused MLP layer for incrementally improved training efficiency
"""

import torch
from transformers.models.llama.modeling_llama import LlamaMLP
from xformers.ops import SwiGLU

from axolotl.monkeypatch.utils import set_module_name


class FusedMLP(torch.nn.Module):
    """
    Fused MLP layer for incrementally improved training efficiency
    """

    def __init__(
        self,
        config,
        gate_proj: torch.nn.Linear,
        up_proj: torch.nn.Linear,
        down_proj: torch.nn.Linear,
    ):
        super().__init__()
        self.config = config
        self.swiglu = SwiGLU(
            in_features=config.hidden_size,
            hidden_features=config.intermediate_size,
            bias=False,
            _pack_weights=True,
        )
        # overwrite initialized weights with pretrained weights
        self.swiglu.w12.weight.data = torch.cat(
            (gate_proj.weight.data, up_proj.weight.data), dim=0
        )
        self.swiglu.w3.weight.data = down_proj.weight.data

    def _post_training(self, model, name):
        w1, w2 = torch.split(  # pylint: disable=invalid-name
            self.swiglu.w12.weight.data, self.config.intermediate_size, dim=0
        )

        # Assign the split weights back to the original layers
        new_mlp = LlamaMLP(self.config)
        new_mlp.gate_proj.weight.data = w1
        new_mlp.up_proj.weight.data = w2
        new_mlp.down_proj.weight.data = self.swiglu.w3.weight.data

        set_module_name(model, name, new_mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=invalid-name
        return self.swiglu(x)
