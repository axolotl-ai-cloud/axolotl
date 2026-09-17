"""The guard that stops Transformers re-initializing already-quantized weights."""

import pytest
import torch
from torch import nn
from torch.nn.utils import parametrize
from transformers import PreTrainedModel

from axolotl.utils.quantization import patch_transformers_skip_quantized_init


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


def test_parametrized_module_is_skipped(initialize):
    """An NF4 packed weight hides behind a parametrization and reads as a missing key."""
    module = nn.Linear(8, 8, bias=False)
    parametrize.register_parametrization(module, "weight", _Identity(), unsafe=True)
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
