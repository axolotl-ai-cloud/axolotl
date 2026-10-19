"""PEFT's ParamWrapper must leave a pre-parametrized expert parameter as it found it."""

import torch
from torch import nn
from torch.nn.utils import parametrize

from axolotl.monkeypatch.moe_quant import patch_peft_target_parameters_matching


class _Packed(nn.Module):
    """Stands in for a quantization parametrization: any transform PEFT did not add."""

    def forward(self, value):
        return value * 2


class _Experts(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(
            torch.randn(2, 16, 16) / 2, requires_grad=False
        )

    def forward(self, x):
        return torch.einsum("bi,eoi->beo", x, self.gate_up_proj)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = _Experts()

    def forward(self, x):
        return self.experts(x)


def test_forward_does_not_leak_lora_proxies_onto_parametrized_experts():
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(0)
    model = _Model()
    parametrize.register_parametrization(
        model.experts, "gate_up_proj", _Packed(), unsafe=True
    )
    patch_peft_target_parameters_matching()
    model = get_peft_model(
        model,
        LoraConfig(r=4, target_modules=[], target_parameters=["experts.gate_up_proj"]),
    )
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_(std=0.1)
    base = model.base_model.model.experts.get_base_layer()
    x = torch.randn(3, 16)

    with torch.no_grad():
        first = model(x)
        second = model(x)

    chain = [type(t).__name__ for t in base.parametrizations["gate_up_proj"]]
    assert chain == ["_Packed"], chain
    torch.testing.assert_close(second, first, rtol=0, atol=0)
