"""Unit tests for ternary (BitNet b1.58 style) QAT."""

from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

from axolotl.utils.callbacks.qat import toggle_fake_quant
from axolotl.utils.quantization_ternary import (
    TernaryFakeQuantizedLinear,
    convert_ternary_model,
    prepare_model_for_ternary_qat,
    quantize_activation,
    ternarize,
)


class TinyModel(nn.Module):
    """Two linear layers, one of which is the LM head."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8, bias=False)
        self.lm_head = nn.Linear(8, 4, bias=False)

    def forward(self, x):
        return self.lm_head(self.proj(x))


def test_prepare_swaps_linears_but_not_lm_head():
    model = TinyModel()
    prepare_model_for_ternary_qat(model)

    assert isinstance(model.proj, TernaryFakeQuantizedLinear)
    assert type(model.lm_head) is nn.Linear  # pylint: disable=unidiomatic-typecheck


def test_forward_uses_ternary_weights_and_straight_through_grads():
    torch.manual_seed(0)
    model = TinyModel()
    x = torch.randn(2, 8)
    prepare_model_for_ternary_qat(model, quantize_activations=False)

    out = model.proj(x)
    assert torch.equal(out, F.linear(x, ternarize(model.proj.weight)))

    out.sum().backward()
    assert torch.allclose(model.proj.weight.grad, x.sum(0).expand(8, 8))


def test_activation_quantization_lands_on_the_int8_grid():
    torch.manual_seed(0)
    x = torch.randn(2, 8)
    scale = x.abs().amax(dim=-1, keepdim=True) / 127

    levels = quantize_activation(x) / scale
    assert torch.allclose(levels, levels.round())
    assert levels.round().abs().max() <= 127


def test_convert_bakes_ternary_weights_into_plain_linear():
    torch.manual_seed(0)
    model = TinyModel()
    x = torch.randn(2, 8)
    prepare_model_for_ternary_qat(model, quantize_activations=False)
    before = model(x)

    convert_ternary_model(model)

    assert type(model.proj) is nn.Linear  # pylint: disable=unidiomatic-typecheck
    assert torch.equal(model(x), before)
    for row in model.proj.weight:
        assert len(torch.unique(row)) <= 3


def test_toggle_fake_quant_restores_the_high_precision_forward():
    torch.manual_seed(0)
    model = TinyModel()
    x = torch.randn(2, 8)
    expected = model(x)

    prepare_model_for_ternary_qat(model)
    model.apply(partial(toggle_fake_quant, enable=False))

    assert torch.equal(model(x), expected)
