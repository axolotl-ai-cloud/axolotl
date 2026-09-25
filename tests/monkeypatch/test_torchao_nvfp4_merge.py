"""CPU coverage for native TorchAO NVFP4 merge-aware LoRA helpers."""

import types
import weakref

import pytest
import torch
import torch.nn.functional as F
from torch import nn

pytest.importorskip("torchao")
peft = pytest.importorskip("peft")

from peft import LoraConfig  # noqa: E402
from peft.tuners.lora.layer import Linear as LoraLinear  # noqa: E402
from torchao.prototype.mx_formats.nvfp4_tensor import (  # noqa: E402
    NVFP4Tensor,
    QuantizeTensorToNVFP4Kwargs,
    per_tensor_amax_to_scale,
)

from axolotl.monkeypatch.torchao_nvfp4_merge import (  # noqa: E402
    _native_dynamic_activation,
    capture_native_nvfp4_recipe,
    install_native_nvfp4_merge_aware_lora_linears,
    native_nvfp4_merge_aware_linear,
    quantize_native_effective_weight,
)

IN, OUT, R = 32, 32, 8


def _native_weight(dynamic=False, supplied_scale=False):
    torch.manual_seed(41)
    source = torch.randn(OUT, IN, dtype=torch.bfloat16)
    kwargs = {}
    if dynamic:
        kwargs["act_quant_kwargs"] = QuantizeTensorToNVFP4Kwargs(
            use_dynamic_per_tensor_scale=not supplied_scale
        )
        if supplied_scale:
            kwargs["act_per_tensor_scale"] = torch.tensor(1.75)
    return NVFP4Tensor.to_nvfp4(
        source,
        per_tensor_scale=torch.tensor(1.125),
        is_swizzled_scales=False,
        **kwargs,
    )


def _lora(weight, adapter_dtype=torch.bfloat16):
    base = nn.Linear(IN, OUT, bias=False, dtype=torch.bfloat16)
    base.weight = nn.Parameter(weight, requires_grad=False)
    config = LoraConfig(r=R, lora_alpha=2 * R)
    lora = LoraLinear(
        base,
        adapter_name="default",
        config=config,
        r=R,
        lora_alpha=2 * R,
        lora_dropout=0.0,
    )
    lora.lora_A["default"].to(adapter_dtype)
    lora.lora_B["default"].to(adapter_dtype)
    with torch.no_grad():
        lora.lora_B["default"].weight.copy_(
            torch.randn(OUT, R, dtype=torch.bfloat16) * 0.03
        )
    return nn.Sequential(lora), lora


@pytest.mark.parametrize(
    "dynamic,supplied_scale", [(False, False), (True, False), (True, True)]
)
def test_recipe_roundtrips_native_metadata(dynamic, supplied_scale):
    weight = _native_weight(dynamic, supplied_scale)
    recipe = capture_native_nvfp4_recipe(weight)
    rebuilt = recipe.quantize(weight.dequantize())

    assert recipe.fingerprint() == capture_native_nvfp4_recipe(rebuilt).fingerprint()
    assert torch.equal(rebuilt.qdata, weight.qdata)
    assert torch.equal(rebuilt.scale, weight.scale)
    assert rebuilt.act_quant_kwargs == weight.act_quant_kwargs
    if weight.act_per_tensor_scale is None:
        assert rebuilt.act_per_tensor_scale is None
    else:
        assert torch.equal(rebuilt.act_per_tensor_scale, weight.act_per_tensor_scale)


@pytest.mark.parametrize(
    "dynamic,supplied_scale", [(False, False), (True, False), (True, True)]
)
def test_effective_weight_uses_native_recipe(dynamic, supplied_scale):
    weight = _native_weight(dynamic, supplied_scale)
    a = torch.randn(R, IN, dtype=torch.float32)
    b = torch.randn(OUT, R, dtype=torch.float32)
    scaling = 1.25
    actual = quantize_native_effective_weight(weight, a, b, scaling)
    recipe = capture_native_nvfp4_recipe(weight)
    expected = recipe.quantize(
        (weight.dequantize().float() + (b @ a * scaling).float()).to(weight.orig_dtype)
    )

    assert torch.equal(actual.qdata, expected.qdata)
    assert torch.equal(actual.scale, expected.scale)
    assert actual.act_quant_kwargs == weight.act_quant_kwargs
    if weight.act_per_tensor_scale is not None:
        assert torch.equal(actual.act_per_tensor_scale, weight.act_per_tensor_scale)


@pytest.mark.parametrize("adapter_dtype", [torch.bfloat16, torch.float32])
def test_native_lora_forward_and_ste_gradients_match_snapped_oracle(adapter_dtype):
    model, lora = _lora(_native_weight(), adapter_dtype)
    assert install_native_nvfp4_merge_aware_lora_linears(model) == 1
    x = torch.randn(3, IN, dtype=torch.bfloat16, requires_grad=True)
    actual = lora(x)
    actual.float().square().sum().backward()

    base = lora.get_base_layer()
    a = lora.lora_A["default"].weight.detach().clone().requires_grad_()
    b = lora.lora_B["default"].weight.detach().clone().requires_grad_()
    x_oracle = x.detach().clone().requires_grad_()
    snapped = quantize_native_effective_weight(
        base.weight, a, b, lora.scaling["default"]
    )
    snapped_dense = snapped.dequantize().detach()
    effective = (
        base.weight.dequantize().detach() + (b @ a) * lora.scaling["default"]
    ).to(base.weight.orig_dtype)
    ste = snapped_dense + (effective - effective.detach())
    oracle = F.linear(x_oracle, ste)
    oracle.float().square().sum().backward()

    torch.testing.assert_close(actual, F.linear(x, snapped), rtol=0, atol=0)
    torch.testing.assert_close(x.grad, x_oracle.grad, rtol=0, atol=0)
    torch.testing.assert_close(
        lora.lora_A["default"].weight.grad, a.grad, rtol=0, atol=0
    )
    torch.testing.assert_close(
        lora.lora_B["default"].weight.grad, b.grad, rtol=0, atol=0
    )
    assert not base.weight.requires_grad


def test_dynamic_native_weight_installs_merge_aware_forward():
    model, lora = _lora(_native_weight(dynamic=True))
    assert install_native_nvfp4_merge_aware_lora_linears(model) == 1
    assert hasattr(lora, "_axolotl_native_nvfp4_orig_forward")


def test_installer_owner_does_not_register_the_model_as_a_child():
    model, lora = _lora(_native_weight())

    assert install_native_nvfp4_merge_aware_lora_linears(model) == 1

    owner = lora._axolotl_native_nvfp4_owner
    assert isinstance(owner, weakref.ReferenceType)
    assert owner() is model
    assert "_axolotl_native_nvfp4_owner" not in lora._modules
    model.train()
    assert model.state_dict()


@pytest.mark.parametrize("unsupported", ["dropout", "bias", "variant"])
def test_native_lora_preserves_unsupported_peft_forwards(unsupported):
    model, lora = _lora(_native_weight())
    if unsupported == "dropout":
        lora.lora_dropout["default"] = nn.Dropout(0.1)
    elif unsupported == "bias":
        lora.lora_bias["default"] = True
    else:
        lora.lora_variant["default"] = object()

    original_forward = lora.forward
    assert install_native_nvfp4_merge_aware_lora_linears(model) == 0
    assert lora.forward.__func__ is original_forward.__func__
    assert lora.forward.__self__ is original_forward.__self__
    assert model._axolotl_merge_aware_unsupported
    assert lora._axolotl_merge_aware_unsupported


def test_native_lora_runtime_adapter_names_preserves_peft_forward():
    model, lora = _lora(_native_weight())
    assert install_native_nvfp4_merge_aware_lora_linears(model) == 1
    calls = []

    def original(self, x, *args, **kwargs):
        calls.append((x, args, kwargs))
        return x

    lora._axolotl_native_nvfp4_orig_forward = types.MethodType(original, lora)
    x = torch.randn(3, IN, dtype=torch.bfloat16)
    assert lora(x, adapter_names=["default"] * len(x)) is x
    assert calls[0][2]["adapter_names"] == ["default"] * len(x)
    assert model._axolotl_merge_aware_unsupported
    assert lora._axolotl_merge_aware_unsupported


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="CUDA sm100+ required",
)
@pytest.mark.parametrize("supplied_scale", [False, True])
def test_dynamic_native_lora_matches_native_forward_and_activation_ste(supplied_scale):
    device = torch.device("cuda")
    in_features = out_features = 128
    rank = 8
    torch.manual_seed(79)
    source = torch.randn(out_features, in_features, device=device, dtype=torch.bfloat16)
    weight = NVFP4Tensor.to_nvfp4(
        source,
        per_tensor_scale=per_tensor_amax_to_scale(source.abs().max()),
        act_per_tensor_scale=(
            per_tensor_amax_to_scale(torch.tensor(4.0, device=device))
            if supplied_scale
            else None
        ),
        is_swizzled_scales=True,
        act_quant_kwargs=QuantizeTensorToNVFP4Kwargs(
            use_dynamic_per_tensor_scale=not supplied_scale,
            is_swizzled_scales=True,
        ),
    )
    a = (
        torch.randn(rank, in_features, device=device, dtype=torch.float32) * 0.02
    ).requires_grad_()
    b = (
        torch.randn(out_features, rank, device=device, dtype=torch.float32) * 0.02
    ).requires_grad_()
    x = torch.randn(
        4, in_features, device=device, dtype=torch.bfloat16, requires_grad=True
    )
    scaling = 1.25

    actual = native_nvfp4_merge_aware_linear(x, None, weight, a, b, scaling)
    snapped = quantize_native_effective_weight(weight, a, b, scaling)
    torch.testing.assert_close(actual, F.linear(x, snapped), rtol=0, atol=0)
    grad_output = torch.randn_like(actual)
    actual.backward(grad_output)

    activation = _native_dynamic_activation(x, snapped).dequantize(x.dtype)
    flat_grad = grad_output.reshape(-1, out_features)
    grad_effective = (
        flat_grad.float()
        .transpose(0, 1)
        .matmul(activation.reshape(-1, in_features).float())
    )
    expected_x = grad_output.matmul(snapped.dequantize())
    expected_b = (grad_effective.matmul(a.float().transpose(0, 1)) * scaling).to(
        b.dtype
    )
    expected_a = (b.float().transpose(0, 1).matmul(grad_effective) * scaling).to(
        a.dtype
    )

    torch.testing.assert_close(x.grad, expected_x, rtol=0, atol=0)
    torch.testing.assert_close(a.grad, expected_a, rtol=0, atol=0)
    torch.testing.assert_close(b.grad, expected_b, rtol=0, atol=0)

    optimizer = torch.optim.SGD([a, b], lr=0.05)
    optimizer.step()
    deployed = quantize_native_effective_weight(weight, a, b, scaling)
    trained = native_nvfp4_merge_aware_linear(x.detach(), None, weight, a, b, scaling)
    torch.testing.assert_close(trained, F.linear(x.detach(), deployed), rtol=0, atol=0)
    recipe = capture_native_nvfp4_recipe(weight)
    deployed_recipe = capture_native_nvfp4_recipe(deployed)
    expected_deployed = recipe.quantize(
        (weight.dequantize().float() + b @ a * scaling).to(weight.orig_dtype)
    )
    assert deployed_recipe.fingerprint() == recipe.fingerprint()
    assert torch.equal(deployed.qdata, expected_deployed.qdata)
    assert torch.equal(deployed.scale, expected_deployed.scale)
    assert deployed.act_quant_kwargs == weight.act_quant_kwargs
    assert torch.equal(deployed.per_tensor_scale, weight.per_tensor_scale)
    if supplied_scale:
        assert torch.equal(deployed.act_per_tensor_scale, weight.act_per_tensor_scale)
    else:
        assert deployed.act_per_tensor_scale is None
