import torch
from typing import List
from xformers.ops import SwiGLU
from transformers.models.llama.modeling_llama import (
    LlamaMLP,
    LlamaAttention
)
from axolotl.monkeypatch.utils import set_module_name

class FusedAttention(LlamaAttention):
    def __init__(self, config, q: torch.nn.Linear, k: torch.nn.Linear, v: torch.nn.Linear, o: torch.nn.Linear):
        super().__init__(config)
        self.config = config
        self.init_device = next(iter(q.state_dict().values())).device

        # define equivalent fused qkv projection
        self.out_features: List[int] = [q.out_features, k.out_features, v.out_features]
        self.qkv_proj = torch.nn.Linear(q.in_features, sum(self.out_features), device=self.init_device, bias=False)
        self.o_proj = o

        # overwrite initialized weights with pretrained weights
        self.qkv_proj.weight.data = torch.cat((q.weight.data, k.weight.data, v.weight.data), dim=0)
    
    def _post_training(self, model, name):
        q_proj, k_proj, v_proj = torch.split(self.qkv_proj.weight.data, self.out_features, dim=0)

        new_attn = LlamaAttention(self.config)
        new_attn.q_proj.weight.data = q_proj
        new_attn.k_proj.weight.data = k_proj
        new_attn.v_proj.weight.data = v_proj

        set_module_name(model, name, new_attn)

class FusedMLP(torch.nn.Module):
    def __init__(self, config, gate_proj: torch.nn.Linear, up_proj: torch.nn.Linear, down_proj: torch.nn.Linear):
        super().__init__()
        self.config = config
        self.swiglu = SwiGLU(
            in_features=config.hidden_size,
            hidden_features=config.intermediate_size,
            bias=False,
            _pack_weights=True
        )
        # overwrite initialized weights with pretrained weights
        self.swiglu.w12.weight.data = torch.cat((gate_proj.weight.data, up_proj.weight.data), dim=0)
        self.swiglu.w3.weight.data = down_proj.weight.data
    
    def _post_training(self, model, name):
        w1, w2 = torch.split(self.swiglu.w12.weight.data, self.config.intermediate_size, dim=0)

        # Assign the split weights back to the original layers
        new_mlp = LlamaMLP(self.config)
        new_mlp.gate_proj.weight.data = w1
        new_mlp.up_proj.weight.data = w2
        new_mlp.down_proj.weight.data = self.swiglu.w3.weight.data

        set_module_name(model, name, new_mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.swiglu(x)
