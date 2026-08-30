"""Unit tests for ternary (BitNet b1.58 style) QAT."""

from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

from axolotl.utils.callbacks.qat import QATCallback, toggle_fake_quant
from axolotl.utils.dict import DictDefault
from axolotl.utils.quantization_ternary import (
    TernaryFakeQuantizedLinear,
    convert_ternary_model,
    has_tied_output_embedding,
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


class TiedModel(TinyModel):
    """A model whose LM head shares its weight with the embedding table."""

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(4, 8)
        self.lm_head = nn.Linear(8, 4, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


class UntiedModel(TiedModel):
    def __init__(self):
        super().__init__()
        self.lm_head = nn.Linear(8, 4, bias=False)


def test_tied_output_embedding_detection():
    assert has_tied_output_embedding(TiedModel())
    assert not has_tied_output_embedding(UntiedModel())
    assert not has_tied_output_embedding(TinyModel())


def test_tied_embedding_is_not_quantized():
    """torchao is never reached for a tied model, so the LM head stays high precision."""
    model = TiedModel()
    prepare_model_for_ternary_qat(model, quantize_embedding=True)

    assert isinstance(model.proj, TernaryFakeQuantizedLinear)
    assert type(model.embed_tokens) is nn.Embedding  # pylint: disable=unidiomatic-typecheck
    assert type(model.lm_head) is nn.Linear  # pylint: disable=unidiomatic-typecheck


def test_convert_reports_whether_the_model_was_ternary():
    model = TinyModel()
    assert convert_ternary_model(model) is False

    prepare_model_for_ternary_qat(model)
    assert convert_ternary_model(model) is True


class FakeState:
    def __init__(self, global_step):
        self.global_step = global_step


def _quantizers_enabled(model):
    return [
        m.weight_fake_quantizer.enabled
        for m in model.modules()
        if isinstance(m, TernaryFakeQuantizedLinear)
    ]


def _callback_for(after_n_steps):
    model = TinyModel()
    prepare_model_for_ternary_qat(model)
    return QATCallback(DictDefault({"fake_quant_after_n_steps": after_n_steps})), model


def test_fake_quant_toggles_at_the_configured_step():
    callback, model = _callback_for(20)

    callback.on_step_begin(None, FakeState(0), None, model)
    assert _quantizers_enabled(model) == [False]

    callback.on_step_begin(None, FakeState(19), None, model)
    assert _quantizers_enabled(model) == [False]

    callback.on_step_begin(None, FakeState(20), None, model)
    assert _quantizers_enabled(model) == [True]


def test_resume_before_the_switch_keeps_fake_quant_off():
    """Quantizers default to enabled, so a resume mid-warmup must switch them off."""
    callback, model = _callback_for(20)

    callback.on_step_begin(None, FakeState(5), None, model)
    assert _quantizers_enabled(model) == [False]


def test_resume_past_the_switch_enables_fake_quant():
    callback, model = _callback_for(20)

    callback.on_step_begin(None, FakeState(35), None, model)
    assert _quantizers_enabled(model) == [True]


def test_toggle_only_applied_on_transitions():
    callback, model = _callback_for(2)

    applied = []
    model.apply = lambda fn: applied.append(fn)  # type: ignore[method-assign]
    for step in range(6):
        callback.on_step_begin(None, FakeState(step), None, model)

    assert len(applied) == 2  # one disable at step 0, one enable at step 2
