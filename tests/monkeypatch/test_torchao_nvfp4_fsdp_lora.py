import weakref

import pytest
import torch

from axolotl.monkeypatch.torchao_nvfp4_fsdp_lora import (
    _install_materialize_mode,
    _materialize_weight,
)


class _Factor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(2, 3))
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return torch.nn.functional.linear(x, self.weight)


def test_materialize_mode_returns_differentiable_clone_without_matrix_probe():
    module = _Factor()
    _install_materialize_mode(module)

    materialized = _materialize_weight(module)

    assert materialized.shape == module.weight.shape
    assert materialized.data_ptr() != module.weight.data_ptr()
    materialized.square().sum().backward()
    assert module.weight.grad is not None
    assert module.calls == 0


def test_materialize_mode_preserves_regular_forward():
    module = _Factor()
    _install_materialize_mode(module)
    inputs = torch.randn(4, 3)

    assert torch.equal(
        module(inputs), torch.nn.functional.linear(inputs, module.weight)
    )
    assert module.calls == 1


def _native_lora(dynamic=False):
    pytest.importorskip("torchao")
    pytest.importorskip("peft")
    from peft import LoraConfig
    from peft.tuners.lora.layer import Linear as LoraLinear
    from torchao.prototype.mx_formats.nvfp4_tensor import (
        NVFP4Tensor,
        QuantizeTensorToNVFP4Kwargs,
    )

    source = torch.randn(32, 32, dtype=torch.bfloat16)
    kwargs = {}
    if dynamic:
        kwargs["act_quant_kwargs"] = QuantizeTensorToNVFP4Kwargs(
            use_dynamic_per_tensor_scale=True
        )
    native = NVFP4Tensor.to_nvfp4(
        source,
        per_tensor_scale=torch.tensor(1.125),
        is_swizzled_scales=False,
        **kwargs,
    )
    base = torch.nn.Linear(32, 32, bias=False, dtype=torch.bfloat16)
    base.weight = torch.nn.Parameter(native, requires_grad=False)
    lora = LoraLinear(
        base,
        adapter_name="default",
        config=LoraConfig(r=8, lora_alpha=16),
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
    )
    with torch.no_grad():
        lora.lora_B["default"].weight.normal_(std=0.1)
    return torch.nn.Sequential(lora), lora


def test_installer_materializes_base_and_factors_through_child_forwards():
    from axolotl.monkeypatch.torchao_nvfp4_fsdp_lora import (
        install_fsdp_native_nvfp4_merge_aware_lora_linears,
    )

    model, lora = _native_lora()
    original = lora.forward

    assert install_fsdp_native_nvfp4_merge_aware_lora_linears(model) == 1
    assert lora._ma_orig_forward.__func__ is original.__func__
    assert hasattr(lora.get_base_layer(), "_axolotl_materialize_orig_forward")
    result = lora(torch.randn(3, 32, dtype=torch.bfloat16))

    assert result.shape == (3, 32)


def test_installer_marks_dynamic_native_weight_unsupported():
    from axolotl.monkeypatch.torchao_nvfp4_fsdp_lora import (
        install_fsdp_native_nvfp4_merge_aware_lora_linears,
    )

    model, lora = _native_lora(dynamic=True)

    assert install_fsdp_native_nvfp4_merge_aware_lora_linears(model) == 0
    assert model._axolotl_merge_aware_unsupported
    assert lora._axolotl_merge_aware_unsupported


def test_fsdp_installer_owner_does_not_register_the_model_as_a_child():
    from axolotl.monkeypatch.torchao_nvfp4_fsdp_lora import (
        install_fsdp_native_nvfp4_merge_aware_lora_linears,
    )

    model, lora = _native_lora()

    assert install_fsdp_native_nvfp4_merge_aware_lora_linears(model) == 1

    owner = lora._axolotl_native_nvfp4_owner
    assert isinstance(owner, weakref.ReferenceType)
    assert owner() is model
    assert "_axolotl_native_nvfp4_owner" not in lora._modules
    model.train()
    assert model.state_dict()
