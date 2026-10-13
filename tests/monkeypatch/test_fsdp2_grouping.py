"""Tests for the FSDP2 trainable-dtype grouping rule."""

import torch
from torch import nn

from axolotl.monkeypatch.accelerate.fsdp2_grouping import (
    plan_trainable_dtype_carveout,
)


class _Lora(nn.Module):
    """Shape of a PEFT LoraLayer around a frozen base linear."""

    def __init__(self, base_dtype=torch.bfloat16, base_trainable=False):
        super().__init__()
        self.base_layer = nn.Linear(8, 8, bias=False).to(base_dtype)
        self.base_layer.weight.requires_grad_(base_trainable)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(8, 2, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(2, 8, bias=False)})

    def forward(self, x):
        return self.base_layer(x) + self.lora_B["default"](self.lora_A["default"](x))


class _Layer(nn.Module):
    def __init__(self, **lora_kwargs):
        super().__init__()
        self.q = _Lora(**lora_kwargs)
        self.v = _Lora(**lora_kwargs)
        self.norm = nn.LayerNorm(8).to(torch.bfloat16)
        self.norm.weight.requires_grad_(False)
        self.norm.bias.requires_grad_(False)


def test_fp32_adapters_next_to_frozen_base_need_no_carveout():
    assert plan_trainable_dtype_carveout(_Layer()) == []


def test_frozen_uint8_base_is_ignored():
    layer = _Layer()
    layer.q.base_layer.weight = nn.Parameter(
        torch.zeros(8, 8, dtype=torch.uint8), requires_grad=False
    )
    assert plan_trainable_dtype_carveout(layer) == []


def test_carves_smaller_trainable_dtype_when_base_is_unfrozen():
    layer = _Layer(base_trainable=True)  # bf16 trainable base, fp32 adapters

    carved = plan_trainable_dtype_carveout(layer)

    assert carved == [
        layer.q.lora_A["default"],
        layer.q.lora_B["default"],
        layer.v.lora_A["default"],
        layer.v.lora_B["default"],
    ]


def test_parameter_dict_owners_escalate_to_module_with_forward():
    class _EmbedLora(nn.Module):
        def __init__(self):
            super().__init__()
            self.base_layer = nn.Embedding(16, 8).to(torch.bfloat16)
            self.base_layer.weight.requires_grad_(False)
            self.lora_embedding_A = nn.ParameterDict(
                {"default": nn.Parameter(torch.zeros(2, 16))}
            )
            self.lora_embedding_B = nn.ParameterDict(
                {"default": nn.Parameter(torch.zeros(8, 2))}
            )

        def forward(self, x):
            return self.base_layer(x)

    root = nn.Module()
    root.embed = _EmbedLora()
    root.head = nn.Linear(8, 16, bias=False).to(torch.bfloat16)  # trainable bf16 copy

    assert plan_trainable_dtype_carveout(root) == [root.embed]


def test_already_wrapped_children_are_excluded():
    layer = _Layer(base_trainable=True)
    wrapped = {id(layer.q), id(layer.v)}

    carved = plan_trainable_dtype_carveout(layer, is_wrapped=lambda m: id(m) in wrapped)

    assert carved == []  # only the frozen norm remains in the unit


def test_non_adapter_modules_are_never_carved():
    root = nn.Module()
    root.scale = nn.Parameter(torch.zeros(1))  # fp32 owned by the unit itself
    root.head = nn.Linear(4, 4, bias=False).to(torch.bfloat16)  # trainable bf16 leaf

    # a plain Linear may be read directly (F.linear(x, m.weight)); refuse to split it out
    assert plan_trainable_dtype_carveout(root) == []


def test_adapters_are_carved_even_when_they_are_the_larger_dtype():
    layer = _Layer()  # frozen bf16 bases, fp32 adapters (64 params)
    layer.gate = nn.Linear(2, 2, bias=False).to(
        torch.bfloat16
    )  # unfrozen router, 4 params

    carved = plan_trainable_dtype_carveout(layer)

    assert carved == [
        layer.q.lora_A["default"],
        layer.q.lora_B["default"],
        layer.v.lora_A["default"],
        layer.v.lora_B["default"],
    ]


def test_mixed_owner_on_the_carved_side_blocks_the_split():
    layer = _Layer(base_trainable=True)
    layer.q.lora_A["default"].bias = nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))

    assert plan_trainable_dtype_carveout(layer) == []


def test_unsplittable_mix_returns_nothing():
    root = nn.Module()
    root.w = nn.Parameter(torch.zeros(4))  # fp32 directly on the root
    root.lin = nn.Linear(4, 4).to(torch.bfloat16)
    root.lin.bias = nn.Parameter(torch.zeros(4))  # fp32 inside a bf16 module

    assert plan_trainable_dtype_carveout(root) == []


def test_pinned_modules_are_never_carved():
    layer = _Layer(base_trainable=True)

    carved = plan_trainable_dtype_carveout(
        layer, pinned=[layer.q.lora_A["default"], layer.q.lora_B["default"]]
    )

    # the pinned fp32 modules stay, so fp32 is treated as owned by the unit and the
    # bf16 side would have to go; its owners are LoRA base layers, which cannot move
    assert carved == []


def test_lora_base_layer_is_never_carved():
    layer = _Layer(base_trainable=True)
    layer.scale = nn.Parameter(torch.zeros(1))  # fp32 owned by the unit itself

    # the only way to keep fp32 would be carving the bf16 base layers, which the
    # fused LoRA kernels read directly; refuse rather than create hookless units
    assert plan_trainable_dtype_carveout(layer) == []
