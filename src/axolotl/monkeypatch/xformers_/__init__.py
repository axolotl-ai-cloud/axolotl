"""
Fused MLP layer for incrementally improved training efficiency
"""
from collections import OrderedDict

import torch
from torch import nn
from transformers.activations import ACT2FN
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

class FusedMLPv2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.swiglu = SwiGLU(
            in_features=self.hidden_size,
            hidden_features=self.intermediate_size,
            bias=config.mlp_bias,
            _pack_weights=True,
        )
        assert config.hidden_act == "silu"

    def _convert_unpacked_to_packed_state_dict(self, unpacked_state_dict):
        """
        Convert state dict from unpacked format (w1, w2, w3) to packed format (w13, w2).
        """
        packed_state_dict = OrderedDict()

        # Handle w1 and w3 -> w13 conversion for weights
        if 'gate_proj.weight' in unpacked_state_dict and 'up_proj.weight' in unpacked_state_dict:
            gate_proj_weight = unpacked_state_dict['gate_proj.weight']
            up_proj_weight = unpacked_state_dict['up_proj.weight']
            # Concatenate gate and up weights along output dimension (dim=0)
            packed_state_dict['swiglu.w12.weight'] = torch.cat([gate_proj_weight, up_proj_weight], dim=0)

        # Handle w1 and w3 -> w13 conversion for biases (if they exist)
        if 'gate_proj.bias' in unpacked_state_dict and 'up_proj.bias' in unpacked_state_dict:
            gate_proj_bias = unpacked_state_dict['gate_proj.bias']
            up_proj_bias = unpacked_state_dict['up_proj.bias']
            # Concatenate gate and up biases along dimension 0
            packed_state_dict['swiglu.w12.bias'] = torch.cat([gate_proj_bias, up_proj_bias], dim=0)

        # Copy down parameters as-is
        if "down_proj.weight" in unpacked_state_dict:
            packed_state_dict["swiglu.w3.weight"] = unpacked_state_dict['down_proj.weight']
        if "down_proj.bias" in unpacked_state_dict:
            packed_state_dict["swiglu.w3.bias"] = unpacked_state_dict['down_proj.bias']

        for key in ['swiglu.w3.weight', 'swiglu.w3.bias']:
            if key in unpacked_state_dict:
                packed_state_dict[key] = unpacked_state_dict[key]

        # Copy any other parameters that might exist
        excluded_keys = [
            'gate_proj.weight', 'gate_proj.bias',
            'down_proj.weight', 'down_proj.bias',
            'up_proj.weight', 'up_proj.bias',
            'swiglu.w12.weight', 'swiglu.w12.bias',
            'swiglu.w3.weight', 'swiglu.w3.bias',
        ]
        for key, value in unpacked_state_dict.items():
            if key not in excluded_keys:
                packed_state_dict[key] = value

        return packed_state_dict

    def load_state_dict(self, state_dict, strict=True):
        """
        Load state dict, handling both packed (w13) and unpacked (w1, w3) formats.
        """
        # Check if this is an unpacked state dict (has w1 and w3 instead of w13)
        has_unpacked_gate_up = 'gate_proj.weight' in state_dict and 'up_proj.weight' in state_dict
        has_packed_swiglu = 'swiglu.w12.weight' in state_dict

        if has_unpacked_gate_up and not has_packed_swiglu:
            state_dict = self._convert_unpacked_to_packed_state_dict(state_dict)

        return super().load_state_dict(state_dict, strict=strict)

    def state_dict(self, destination=None, prefix='', keep_vars=False, packed=False):
        """
        Return state dict in unpacked format by default for compatibility.
        Set packed=True to get the internal packed format.
        """
        if packed:
            # Return the actual packed state dict
            return super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        else:
            # Return unpacked format for compatibility
            return self.get_unpacked_state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    def get_unpacked_state_dict(self, destination=None, prefix='', keep_vars=False):
        """
        Convert current packed state dict to unpacked format for compatibility.
        """
        # Get the actual packed state dict first
        packed_state_dict = super().state_dict(destination=None, prefix='', keep_vars=keep_vars)

        if destination is None:
            destination = OrderedDict()

        # Handle w13 -> w1 and w3 conversion for weights
        if f'{prefix}swiglu.w12.weight' in packed_state_dict:
            w13_weight = packed_state_dict[f'{prefix}swiglu.w12.weight']
            hidden_dim = w13_weight.shape[0] // 2
            w1_weight, w3_weight = torch.split(w13_weight, hidden_dim, dim=0)
            destination[f'{prefix}gate_proj.weight'] = w1_weight if not keep_vars else w1_weight.detach().requires_grad_(w1_weight.requires_grad)
            destination[f'{prefix}up_proj.weight'] = w3_weight if not keep_vars else w3_weight.detach().requires_grad_(w3_weight.requires_grad)

        # Handle w13 -> w1 and w3 conversion for biases (if they exist)
        if f'{prefix}swiglu.w12.bias' in packed_state_dict:
            w13_bias = packed_state_dict[f'{prefix}swiglu.w12.bias']
            hidden_dim = w13_bias.shape[0] // 2
            w1_bias, w3_bias = torch.split(w13_bias, hidden_dim, dim=0)
            destination[f'{prefix}gate_proj.bias'] = w1_bias if not keep_vars else w1_bias.detach().requires_grad_(w1_bias.requires_grad)
            destination[f'{prefix}up_proj.bias'] = w3_bias if not keep_vars else w3_bias.detach().requires_grad_(w3_bias.requires_grad)

        # Copy w2 parameters as-is
        for param_name in ['weight', 'bias']:
            key = f'{prefix}swiglu.w3.{param_name}'
            if key in packed_state_dict:
                destination[f'{prefix}down_proj.{param_name}'] = packed_state_dict[key]

        # Copy any other parameters
        excluded_prefixes = [f'{prefix}swiglu.w12.', f'{prefix}swiglu.w3.']
        for key, value in packed_state_dict.items():
            if not any(key.startswith(excluded_prefix) for excluded_prefix in excluded_prefixes) and key not in destination:
                destination[key] = value

        return destination

    def forward(self, x):
        return self.swiglu(x)
