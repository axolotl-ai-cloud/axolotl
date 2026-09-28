"""CPU coverage for DeepSpeed-safe native NVFP4 LoRA forwards."""

import pytest
import torch
import torch.nn.functional as functional
from torch import nn

pytest.importorskip("torchao")
pytest.importorskip("peft")

from peft import LoraConfig  # noqa: E402
from peft.tuners.lora.layer import Linear as LoraLinear  # noqa: E402
from torchao.prototype.mx_formats.nvfp4_tensor import (  # noqa: E402
    NVFP4Tensor,
    QuantizeTensorToNVFP4Kwargs,
)

from axolotl.monkeypatch.torchao_deepspeed import (  # noqa: E402
    prepare_native_nvfp4_zero3,
)
from axolotl.monkeypatch.torchao_nvfp4_deepspeed_lora import (  # noqa: E402
    DeepSpeedNativeNVFP4MergeAwareCallback,
    install_deepspeed_native_nvfp4_merge_aware_lora_linears,
)
from axolotl.monkeypatch.torchao_nvfp4_merge import (  # noqa: E402
    quantize_native_effective_weight,
)

IN, OUT, RANK = 32, 32, 8


@pytest.fixture(autouse=True)
def stub_broadcast_filter(monkeypatch):
    import axolotl.monkeypatch.torchao_deepspeed as torchao_deepspeed

    monkeypatch.setattr(
        torchao_deepspeed, "_install_native_nvfp4_broadcast_filter", lambda: None
    )


def _native_weight(dynamic=False):
    torch.manual_seed(725)
    kwargs = {}
    if dynamic:
        kwargs["act_per_tensor_scale"] = torch.tensor(1.25)
        kwargs["act_quant_kwargs"] = QuantizeTensorToNVFP4Kwargs(
            use_dynamic_per_tensor_scale=True
        )
    return NVFP4Tensor.to_nvfp4(
        torch.randn(OUT, IN, dtype=torch.bfloat16),
        per_tensor_scale=torch.tensor(1.125),
        is_swizzled_scales=False,
        **kwargs,
    )


def _lora(native):
    base = nn.Linear(IN, OUT, bias=False, dtype=torch.bfloat16)
    base.weight = nn.Parameter(native, requires_grad=False)
    lora = LoraLinear(
        base,
        adapter_name="default",
        config=LoraConfig(r=RANK, lora_alpha=2 * RANK),
        r=RANK,
        lora_alpha=2 * RANK,
        lora_dropout=0.0,
    )
    lora.lora_A["default"].to(torch.float32)
    lora.lora_B["default"].to(torch.float32)
    with torch.no_grad():
        lora.lora_B["default"].weight.normal_(std=0.1)
    return nn.Sequential(lora), lora


def _oracle(lora, native, inputs):
    a = lora.lora_A["default"].weight.detach().clone().requires_grad_()
    b = lora.lora_B["default"].weight.detach().clone().requires_grad_()
    oracle_inputs = inputs.detach().clone().requires_grad_()
    snapped = quantize_native_effective_weight(native, a, b, lora.scaling["default"])
    effective = (native.dequantize().detach() + b @ a * lora.scaling["default"]).to(
        native.orig_dtype
    )
    ste = snapped.dequantize().detach() + (effective - effective.detach())
    output = functional.linear(oracle_inputs, ste)
    output.float().square().sum().backward()
    return output, oracle_inputs, a, b


@pytest.mark.parametrize("zero3", [False, True])
def test_static_native_lora_matches_snapped_oracle_and_materializes_via_child_forward(
    zero3,
):
    model, lora = _lora(_native_weight())
    native = lora.get_base_layer().weight
    if zero3:
        assert prepare_native_nvfp4_zero3(model, torch.device("cpu"))
    base = lora.get_base_layer()
    calls = {"base": 0, "a": 0, "b": 0}
    base.register_forward_pre_hook(
        lambda *_: calls.__setitem__("base", calls["base"] + 1)
    )
    lora.lora_A["default"].register_forward_pre_hook(
        lambda *_: calls.__setitem__("a", calls["a"] + 1)
    )
    lora.lora_B["default"].register_forward_pre_hook(
        lambda *_: calls.__setitem__("b", calls["b"] + 1)
    )

    assert install_deepspeed_native_nvfp4_merge_aware_lora_linears(model) == 1
    inputs = torch.randn(3, IN, dtype=torch.bfloat16, requires_grad=True)
    actual = lora(inputs)
    actual.float().square().sum().backward()
    expected, expected_inputs, expected_a, expected_b = _oracle(lora, native, inputs)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(inputs.grad, expected_inputs.grad, rtol=0, atol=0)
    torch.testing.assert_close(
        lora.lora_A["default"].weight.grad, expected_a.grad, rtol=0, atol=0
    )
    torch.testing.assert_close(
        lora.lora_B["default"].weight.grad, expected_b.grad, rtol=0, atol=0
    )
    assert calls == {"base": 1, "a": 1, "b": 1}
    assert not base.weight.requires_grad


@pytest.mark.parametrize("zero3", [False, True])
def test_dynamic_native_lora_dispatches_materialized_original_recipe(
    monkeypatch, zero3
):
    model, lora = _lora(_native_weight(dynamic=True))
    native = lora.get_base_layer().weight
    if zero3:
        assert prepare_native_nvfp4_zero3(model, torch.device("cpu"))
    base = lora.get_base_layer()
    calls = {"base": 0, "a": 0, "b": 0}
    base.register_forward_pre_hook(
        lambda *_: calls.__setitem__("base", calls["base"] + 1)
    )
    lora.lora_A["default"].register_forward_pre_hook(
        lambda *_: calls.__setitem__("a", calls["a"] + 1)
    )
    lora.lora_B["default"].register_forward_pre_hook(
        lambda *_: calls.__setitem__("b", calls["b"] + 1)
    )

    assert install_deepspeed_native_nvfp4_merge_aware_lora_linears(model) == 1
    import axolotl.monkeypatch.torchao_nvfp4_deepspeed_lora as bridge

    captured = []

    def primitive(*args):
        captured.append(args)
        return args[0]

    monkeypatch.setattr(bridge, "native_nvfp4_merge_aware_linear", primitive)
    inputs = torch.randn(3, IN, dtype=torch.bfloat16)
    assert lora(inputs) is inputs
    assert len(captured) == 1
    dispatched = captured[0]
    assert dispatched[1] is None
    materialized = dispatched[2]
    assert materialized.act_quant_kwargs == native.act_quant_kwargs
    torch.testing.assert_close(materialized.per_tensor_scale, native.per_tensor_scale)
    torch.testing.assert_close(
        materialized.act_per_tensor_scale, native.act_per_tensor_scale
    )
    torch.testing.assert_close(dispatched[3], lora.lora_A["default"].weight)
    torch.testing.assert_close(dispatched[4], lora.lora_B["default"].weight)
    assert dispatched[5] == lora.scaling["default"]
    assert calls == {"base": 1, "a": 1, "b": 1}
    assert not base.weight.requires_grad


def test_dynamic_input_ste_composes_deepspeed_peft_materialization_sentinel(
    monkeypatch,
):
    from axolotl.monkeypatch.torchao_nvfp4_dynamic_ste import (
        install_deepspeed_native_nvfp4_dynamic_input_stes,
    )

    model, lora = _lora(_native_weight(dynamic=True))
    base = lora.get_base_layer()
    assert install_deepspeed_native_nvfp4_merge_aware_lora_linears(model) == 1
    assert install_deepspeed_native_nvfp4_dynamic_input_stes(model) == 1
    assert hasattr(base, "_axolotl_deepspeed_materialize_orig_forward")
    assert hasattr(base, "_axolotl_dynamic_nvfp4_ste_orig_forward")
    materialized = base(_axolotl_materialize_weight=True)
    assert materialized is not base.weight
    assert materialized.qdata.data_ptr() != base.weight.qdata.data_ptr()
    assert materialized.act_quant_kwargs == base.weight.act_quant_kwargs

    import axolotl.monkeypatch.torchao_nvfp4_deepspeed_lora as bridge

    captured = []
    monkeypatch.setattr(
        bridge,
        "native_nvfp4_merge_aware_linear",
        lambda *args: captured.append(args) or args[0],
    )
    inputs = torch.randn(3, IN, dtype=torch.bfloat16)
    assert lora(inputs) is inputs
    assert captured[0][2].qdata.data_ptr() != base.weight.qdata.data_ptr()
    assert model._axolotl_native_nvfp4_dynamic_input_gradients


@pytest.mark.parametrize("stage", [0, 2])
def test_post_engine_callback_installs_dynamic_ste_for_unpacked_zero_without_merge_aware(
    stage,
):
    from types import SimpleNamespace

    model, lora = _lora(_native_weight(dynamic=True))
    model._axolotl_native_nvfp4_dynamic_input_gradients_requested = "DeepSpeed"
    trainer = SimpleNamespace(
        model_wrapped=SimpleNamespace(
            module=model, zero_optimization_stage=lambda: stage
        )
    )

    DeepSpeedNativeNVFP4MergeAwareCallback(trainer).on_train_begin(None, None, None)

    assert model._axolotl_native_nvfp4_dynamic_input_gradients
    assert hasattr(lora.get_base_layer(), "_axolotl_dynamic_nvfp4_ste_orig_forward")
    assert not hasattr(lora, "_axolotl_deepspeed_native_orig_forward")


def test_post_engine_callback_warns_and_continues_for_unsupported_dynamic_stage():
    from types import SimpleNamespace

    model = SimpleNamespace(
        _axolotl_native_nvfp4_dynamic_input_gradients_requested="DeepSpeed"
    )
    trainer = SimpleNamespace(
        model_wrapped=SimpleNamespace(module=model, zero_optimization_stage=lambda: 4)
    )

    DeepSpeedNativeNVFP4MergeAwareCallback(trainer).on_train_begin(None, None, None)

    assert not model._axolotl_native_nvfp4_dynamic_input_gradients
    assert model._axolotl_merge_aware_unsupported


def test_post_engine_callback_leaves_zero3_packed_dynamic_path_untouched(monkeypatch):
    from types import SimpleNamespace

    model = SimpleNamespace(
        _axolotl_native_nvfp4_dynamic_input_gradients_requested="DeepSpeed"
    )
    trainer = SimpleNamespace(
        model_wrapped=SimpleNamespace(module=model, zero_optimization_stage=lambda: 3)
    )
    monkeypatch.setattr(
        "axolotl.monkeypatch.torchao_nvfp4_dynamic_ste.install_deepspeed_native_nvfp4_dynamic_input_stes",
        lambda _: pytest.fail("ZeRO-3 must retain its packed input-gradient path"),
    )
    monkeypatch.setattr(
        "axolotl.monkeypatch.torchao_nvfp4_dynamic_ste.validate_native_nvfp4_dynamic_input_stes",
        lambda _: True,
    )

    DeepSpeedNativeNVFP4MergeAwareCallback(trainer).on_train_begin(None, None, None)


def test_builder_registers_deepspeed_callback_for_dynamic_input_gradient_request():
    from types import SimpleNamespace

    from axolotl.core.builders.base import TrainerBuilderBase

    class Builder(TrainerBuilderBase):
        def build(self, total_num_steps):
            del total_num_steps

    builder = object.__new__(Builder)
    builder.cfg = SimpleNamespace(plugins=[])
    builder.model = SimpleNamespace(
        _axolotl_native_nvfp4_dynamic_input_gradients_requested="DeepSpeed"
    )
    trainer = object()

    callbacks = builder.get_post_trainer_create_callbacks(trainer)

    assert len(callbacks) == 1
    assert isinstance(callbacks[0], DeepSpeedNativeNVFP4MergeAwareCallback)
    assert callbacks[0].trainer is trainer
