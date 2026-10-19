import torch

from axolotl.monkeypatch.torchao_nvfp4_dynamic_ste import (
    install_fsdp_native_nvfp4_dynamic_input_stes,
    install_native_nvfp4_dynamic_input_stes,
    native_nvfp4_dynamic_input_ste_preflight,
    native_nvfp4_frozen_dynamic_linear,
    validate_native_nvfp4_dynamic_input_stes,
)


class NVFP4Tensor(torch.Tensor):
    @staticmethod
    def __new__(cls, value):
        return torch.Tensor._make_subclass(cls, value, False)

    @property
    def act_quant_kwargs(self):
        return object()

    def dequantize(self):
        return self.as_subclass(torch.Tensor)


def test_frozen_dynamic_ste_uses_native_forward_and_dense_input_gradient(monkeypatch):
    import axolotl.monkeypatch.torchao_nvfp4_dynamic_ste as bridge

    weight = NVFP4Tensor(torch.tensor([[1.0, -2.0], [3.0, 4.0]]))
    inputs = torch.tensor([[2.0, -1.0]], requires_grad=True)
    calls = []
    original_linear = bridge.F.linear

    def native_linear(value, native, bias):
        calls.append(native)
        return original_linear(value, native.dequantize(), bias)

    monkeypatch.setattr(bridge.F, "linear", native_linear)
    output = native_nvfp4_frozen_dynamic_linear(inputs, None, weight)
    output.sum().backward()

    assert calls == [weight]
    torch.testing.assert_close(inputs.grad, torch.tensor([[4.0, 2.0]]))


def test_plain_installer_wraps_frozen_dynamic_native_linear():
    module = torch.nn.Linear(2, 2, bias=False)
    module.weight = torch.nn.Parameter(NVFP4Tensor(torch.eye(2)), requires_grad=False)
    model = torch.nn.Sequential(module)

    assert install_native_nvfp4_dynamic_input_stes(model) == 1
    assert hasattr(module, "_axolotl_dynamic_nvfp4_ste_orig_forward")


def test_fsdp_installer_composes_materialization_sentinel_for_adapter_base():
    module = torch.nn.Linear(2, 2, bias=False)
    module.weight = torch.nn.Parameter(NVFP4Tensor(torch.eye(2)), requires_grad=False)
    module._axolotl_materialize_orig_forward = module.forward
    model = torch.nn.Sequential(module)

    assert install_fsdp_native_nvfp4_dynamic_input_stes(model) == 1
    materialized = module(_axolotl_materialize_weight=True)
    assert materialized.data_ptr() != module.weight.data_ptr()
    assert model._axolotl_native_nvfp4_dynamic_input_gradients


def test_coverage_validator_rejects_unwrapped_dynamic_native_linear():
    module = torch.nn.Linear(2, 2, bias=False)
    module.weight = torch.nn.Parameter(NVFP4Tensor(torch.eye(2)), requires_grad=False)
    model = torch.nn.Sequential(module)

    assert not validate_native_nvfp4_dynamic_input_stes(model)
    assert not model._axolotl_native_nvfp4_dynamic_input_gradients


def test_preflight_requires_frozen_dynamic_native_linear_target():
    module = torch.nn.Linear(2, 2, bias=False)
    module.weight = torch.nn.Parameter(NVFP4Tensor(torch.eye(2)), requires_grad=False)
    model = torch.nn.Sequential(module)

    assert native_nvfp4_dynamic_input_ste_preflight(model)
    module.weight.requires_grad_(True)
    assert not native_nvfp4_dynamic_input_ste_preflight(model)


class UnknownDynamicOwner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(NVFP4Tensor(torch.eye(2)), requires_grad=False)


def test_coverage_rejects_mixed_known_and_unknown_dynamic_native_weights():
    known = torch.nn.Linear(2, 2, bias=False)
    known.weight = torch.nn.Parameter(NVFP4Tensor(torch.eye(2)), requires_grad=False)
    model = torch.nn.ModuleDict({"known": known, "unknown": UnknownDynamicOwner()})

    assert install_native_nvfp4_dynamic_input_stes(model) == 1
    assert not model._axolotl_native_nvfp4_dynamic_input_gradients
    assert not native_nvfp4_dynamic_input_ste_preflight(model)


def test_coverage_does_not_mark_native_free_model_valid():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False))

    assert not validate_native_nvfp4_dynamic_input_stes(model)
    assert not model._axolotl_native_nvfp4_dynamic_input_gradients


def test_coverage_accepts_packed_zero3_through_materializer_origin():
    class Packed(torch.nn.Module):
        def _zero3_native_forward(self, inputs):
            return inputs

    packed = Packed()
    packed._axolotl_nvfp4_act_quant_kwargs = object()
    packed._axolotl_deepspeed_materialize_orig_forward = packed._zero3_native_forward

    def materialize_forward(self, inputs):
        return self._axolotl_deepspeed_materialize_orig_forward(inputs)

    packed.forward = materialize_forward.__get__(packed, Packed)
    model = torch.nn.Sequential(packed)
    model._axolotl_native_nvfp4_zero3_components = ("packed",)

    assert validate_native_nvfp4_dynamic_input_stes(model)
    assert model._axolotl_native_nvfp4_dynamic_input_gradients
