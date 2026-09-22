"""The guard that stops Transformers re-initializing already-quantized weights."""

import sys

import pytest
import torch
from torch import nn
from torch.nn.utils import parametrize
from transformers import PreTrainedModel

from axolotl.monkeypatch.quantized_init import patch_transformers_skip_quantized_init


class _Identity(nn.Module):
    def forward(self, weight):
        return weight


@pytest.fixture(name="initialize")
def _initialize():
    """Return a callable reporting whether Transformers would initialize a module."""
    patch_transformers_skip_quantized_init()
    initialized = []

    class _Model(PreTrainedModel):
        config_class = None

        def _init_weights(self, module):
            initialized.append(module)

    model = _Model.__new__(_Model)

    def run(module):
        initialized.clear()
        PreTrainedModel._initialize_weights(model, module)
        return bool(initialized)

    return run


def test_plain_module_is_still_initialized(initialize):
    """The control: without this, every assertion below passes vacuously."""
    assert initialize(nn.Linear(8, 8)) is True


def test_unrelated_parametrization_is_still_initialized(initialize):
    """Weight norm and friends are not quantized; the guard must not claim them."""
    module = nn.Linear(8, 8, bias=False)
    parametrize.register_parametrization(module, "weight", _Identity(), unsafe=True)
    assert initialize(module) is True


@pytest.mark.parametrize("backend", ["bitsandbytes", "torchao"])
def test_nf4_parametrized_module_is_skipped(initialize, backend):
    """An NF4 packed weight hides behind a parametrization and reads as a missing key."""
    from axolotl.utils.nf4 import (
        BnbNF4Parametrization,
        quantize_bnb_4bit,
        quantize_torchao_nf4,
    )

    weight = torch.randn(64, 64)
    if backend == "torchao":
        pytest.importorskip("torchao")
        data, transform = quantize_torchao_nf4(weight, chunk_size=16384)
    else:
        pytest.importorskip("bitsandbytes")
        data, state = quantize_bnb_4bit(weight)
        transform = BnbNF4Parametrization(state)
    module = nn.Linear(64, 64, bias=False)
    module.weight = nn.Parameter(data, requires_grad=False)
    parametrize.register_parametrization(module, "weight", transform, unsafe=True)
    assert initialize(module) is False
    assert module._is_hf_initialized is True


@pytest.mark.parametrize("bits", [4, 8])
def test_quantized_expert_parametrization_is_skipped(initialize, bits):
    """quantize_moe_experts packs 3-D experts behind bitsandbytes' own parametrizations."""
    bnb = pytest.importorskip("bitsandbytes")
    import bitsandbytes.nn.parametrize  # noqa: F401

    from axolotl.monkeypatch.moe_quant import Bnb8bitParametrization
    from axolotl.utils.nf4 import quantize_bnb_4bit

    weight = torch.randn(4, 32, 64)
    if bits == 4:
        data, state = quantize_bnb_4bit(weight)
        transform = bnb.nn.parametrize.Bnb4bitParametrization(state)
    else:
        data, row_stats, _ = bnb.functional.int8_vectorwise_quant(
            weight.reshape(-1, 64).to(torch.float16)
        )
        transform = Bnb8bitParametrization(row_stats)
    module = nn.Module()
    module.gate_up_proj = nn.Parameter(data, requires_grad=False)
    parametrize.register_parametrization(module, "gate_up_proj", transform, unsafe=True)
    assert initialize(module) is False
    assert module._is_hf_initialized is True


def test_torchao_tensor_module_is_skipped(initialize):
    """A torchao subclass drops the skip flag on .float() and may not implement normal_."""
    mx_tensor = pytest.importorskip("torchao.prototype.mx_formats.mx_tensor")

    module = nn.Linear(32, 32, bias=False)
    module.weight = nn.Parameter(
        mx_tensor.MXTensor.to_mx(
            torch.zeros(32, 32, dtype=torch.bfloat16), torch.float8_e4m3fn, 32
        ),
        requires_grad=False,
    )
    assert initialize(module) is False
    assert module._is_hf_initialized is True


def test_axolotl_parametrizations_survive_a_bitsandbytes_rename(
    initialize, monkeypatch
):
    """A bitsandbytes rename must not take axolotl's own expert parametrization with it."""
    bnb = pytest.importorskip("bitsandbytes")

    import axolotl.monkeypatch.quantized_init as quantized_init
    from axolotl.monkeypatch.moe_quant import Bnb8bitParametrization
    from axolotl.utils.nf4 import quantize_bnb_4bit

    unpatched = getattr(
        PreTrainedModel._initialize_weights,
        "__wrapped__",
        PreTrainedModel._initialize_weights,
    )
    monkeypatch.setattr(PreTrainedModel, "_initialize_weights", unpatched)
    monkeypatch.setitem(sys.modules, "bitsandbytes.nn.parametrize", None)
    quantized_init.patch_transformers_skip_quantized_init()

    weight = torch.randn(4, 32, 64)
    data, row_stats, _ = bnb.functional.int8_vectorwise_quant(
        weight.reshape(-1, 64).to(torch.float16)
    )
    module = nn.Module()
    module.gate_up_proj = nn.Parameter(data, requires_grad=False)
    parametrize.register_parametrization(
        module, "gate_up_proj", Bnb8bitParametrization(row_stats), unsafe=True
    )
    assert initialize(module) is False

    packed, state = quantize_bnb_4bit(weight)
    unreachable = nn.Module()
    unreachable.gate_up_proj = nn.Parameter(packed, requires_grad=False)
    parametrize.register_parametrization(
        unreachable,
        "gate_up_proj",
        sys.modules["bitsandbytes"].nn.parametrize.Bnb4bitParametrization(state),
        unsafe=True,
    )
    assert initialize(unreachable) is True


def test_guard_installed_without_torchao_recognizes_it_later(monkeypatch):
    """A guard installed while torchao was unimportable must not stay blind to it."""
    mx_tensor = pytest.importorskip("torchao.prototype.mx_formats.mx_tensor")
    from transformers import PreTrainedModel

    masked = {
        name: sys.modules[name]
        for name in list(sys.modules)
        if name == "torchao" or name.startswith("torchao.")
    }
    for name in masked:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "torchao", None)
    patch_transformers_skip_quantized_init()
    for name, module in masked.items():
        monkeypatch.setitem(sys.modules, name, module)
    patch_transformers_skip_quantized_init()

    initialized = []

    class _Model(PreTrainedModel):
        config_class = None

        def _init_weights(self, module):
            initialized.append(module)

    module = nn.Linear(32, 32, bias=False)
    module.weight = nn.Parameter(
        mx_tensor.MXTensor.to_mx(
            torch.zeros(32, 32, dtype=torch.bfloat16), torch.float8_e4m3fn, 32
        ),
        requires_grad=False,
    )
    PreTrainedModel._initialize_weights(_Model.__new__(_Model), module)
    assert not initialized
