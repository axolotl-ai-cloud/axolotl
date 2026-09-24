"""CPU representation tests for native NVFP4 ZeRO-3 components."""

import pytest
import torch
from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

from axolotl.monkeypatch.torchao_deepspeed import prepare_native_nvfp4_zero3


@pytest.fixture(autouse=True)
def stub_broadcast_filter(monkeypatch):
    import axolotl.monkeypatch.torchao_deepspeed as torchao_deepspeed

    monkeypatch.setattr(
        torchao_deepspeed, "_install_native_nvfp4_broadcast_filter", lambda: None
    )


def _weight(shape=(4, 16)):
    torch.manual_seed(0)
    return torch.nn.Parameter(
        NVFP4Tensor.to_nvfp4(torch.randn(shape, dtype=torch.bfloat16)),
        requires_grad=False,
    )


def _model(weight):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = torch.nn.Linear(16, 4, bias=False, dtype=torch.bfloat16)
            self.second = torch.nn.Linear(16, 4, bias=False, dtype=torch.bfloat16)
            self.first.weight = weight
            self.second.weight = weight

        def forward(self, value):
            return self.first(value) + self.second(value)

    return Model()


def test_zero3_components_preserve_native_output_scale_layout_and_aliases():
    model = _model(_weight())
    value = torch.randn(2, 16, dtype=torch.bfloat16)
    expected = model(value)
    reference_input = value.detach().clone().requires_grad_()
    model(reference_input).float().sum().backward()
    reference_grad = reference_input.grad.clone()
    source = model.first.weight
    qdata = source.qdata.clone()
    scale_bytes = source.scale.view(torch.uint8).clone()

    assert prepare_native_nvfp4_zero3(model, torch.device("cpu"))

    torch.testing.assert_close(model(value), expected, rtol=0, atol=0)
    saved_shapes = []
    actual_input = value.detach().clone().requires_grad_()
    with torch.autograd.graph.saved_tensors_hooks(
        lambda tensor: saved_shapes.append(tuple(tensor.shape)) or tensor,
        lambda tensor: tensor,
    ):
        model(actual_input).float().sum().backward()
    torch.testing.assert_close(actual_input.grad, reference_grad, rtol=0, atol=0)
    assert tuple(source.shape) not in saved_shapes
    assert tuple(reversed(source.shape)) not in saved_shapes
    assert all(
        shape not in {(source.numel(),), (1, source.numel())} for shape in saved_shapes
    )
    assert model.first.weight.is_meta
    assert model.second.weight.is_meta
    assert model.first._axolotl_nvfp4_qdata is model.second._axolotl_nvfp4_qdata
    assert (
        model.first._axolotl_nvfp4_scale_bytes
        is model.second._axolotl_nvfp4_scale_bytes
    )
    assert torch.equal(model.first._axolotl_nvfp4_qdata, qdata)
    assert torch.equal(model.first._axolotl_nvfp4_scale_bytes, scale_bytes)
    assert all(
        parameter.dtype == torch.uint8 and not parameter.requires_grad
        for parameter in model.parameters()
    )
    assert set(model._axolotl_native_nvfp4_deepspeed_names) == {
        "first._axolotl_nvfp4_qdata",
        "first._axolotl_nvfp4_scale_bytes",
        "second._axolotl_nvfp4_qdata",
        "second._axolotl_nvfp4_scale_bytes",
    }
    assert "first._axolotl_nvfp4_per_tensor_scale_bytes" not in model.state_dict()


def test_zero3_components_reject_3d_before_mutating_model():
    model = _model(_weight((2, 4, 16)))
    before = tuple(model.named_parameters(remove_duplicate=False))

    assert not prepare_native_nvfp4_zero3(model, torch.device("cpu"))

    after = tuple(model.named_parameters(remove_duplicate=False))
    assert [name for name, _ in after] == [name for name, _ in before]
    assert all(
        parameter is before[index][1] for index, (_, parameter) in enumerate(after)
    )


def test_zero3_components_preserve_trainable_bias_gradient():
    layer = torch.nn.Linear(16, 4, bias=True, dtype=torch.bfloat16)
    layer.weight = _weight()
    values = [
        torch.randn(shape, dtype=torch.bfloat16)
        for shape in ((16,), (2, 16), (2, 3, 16))
    ]
    expected = []
    for value in values:
        reference_input = value.detach().clone().requires_grad_()
        layer(reference_input).float().sum().backward()
        expected.append((reference_input.grad.clone(), layer.bias.grad.clone()))
        layer.bias.grad = None

    model = torch.nn.Module()
    model.layer = layer
    assert prepare_native_nvfp4_zero3(model, torch.device("cpu"))
    for value, (input_grad, bias_grad) in zip(values, expected, strict=True):
        actual_input = value.detach().clone().requires_grad_()
        model.layer(actual_input).float().sum().backward()
        torch.testing.assert_close(actual_input.grad, input_grad, rtol=0, atol=0)
        torch.testing.assert_close(layer.bias.grad, bias_grad, rtol=0, atol=0)
        layer.bias.grad = None


def test_zero3_preflight_does_not_mutate_valid_weight_before_custom_2d_weight():
    model = torch.nn.Module()
    model.first = torch.nn.Linear(16, 4, bias=False, dtype=torch.bfloat16)
    model.first.weight = _weight()
    model.second = torch.nn.Module()
    model.second.register_parameter("weight", _weight())
    before = tuple(model.named_parameters(remove_duplicate=False))

    assert not prepare_native_nvfp4_zero3(model, torch.device("cpu"))

    after = tuple(model.named_parameters(remove_duplicate=False))
    assert [name for name, _ in after] == [name for name, _ in before]
    assert all(
        parameter is before[index][1] for index, (_, parameter) in enumerate(after)
    )


def test_zero3_components_preserve_non_bf16_per_tensor_scale_bytes():
    scale = torch.tensor(1.0012345, dtype=torch.float32)
    weight = torch.nn.Parameter(
        NVFP4Tensor.to_nvfp4(
            torch.randn(4, 16, dtype=torch.bfloat16), per_tensor_scale=scale
        ),
        requires_grad=False,
    )
    model = _model(weight)
    expected = scale.reshape(-1).view(torch.uint8).clone()

    assert prepare_native_nvfp4_zero3(model, torch.device("cpu"))
    model.to(dtype=torch.bfloat16)

    assert torch.equal(model.first._axolotl_nvfp4_per_tensor_scale_bytes, expected)
    assert model.first._axolotl_nvfp4_per_tensor_scale_bytes.dtype == torch.uint8
    assert torch.equal(
        model.first._axolotl_nvfp4_per_tensor_scale_bytes,
        model.second._axolotl_nvfp4_per_tensor_scale_bytes,
    )
    assert "first._axolotl_nvfp4_per_tensor_scale_bytes" not in model.state_dict()


def test_zero3_components_preserve_shared_linear_module_aliases():
    model = torch.nn.Module()
    model.first = torch.nn.Linear(16, 4, bias=False, dtype=torch.bfloat16)
    model.first.weight = _weight()
    model.second = model.first

    assert prepare_native_nvfp4_zero3(model, torch.device("cpu"))

    assert model.first is model.second
    assert set(model._axolotl_native_nvfp4_deepspeed_names) == {
        "first._axolotl_nvfp4_qdata",
        "first._axolotl_nvfp4_scale_bytes",
        "second._axolotl_nvfp4_qdata",
        "second._axolotl_nvfp4_scale_bytes",
    }
    assert model.first.weight.is_meta


def test_zero3_peft_export_gathers_selected_adapter_parameters_only(monkeypatch):
    import types

    from peft.utils import save_and_load

    from axolotl.monkeypatch.torchao_deepspeed import (
        native_nvfp4_zero3_peft_state_dict,
    )

    model = torch.nn.Module()
    model.register_parameter("base_component", torch.nn.Parameter(torch.ones(4), False))
    model.register_parameter("adapter", torch.nn.Parameter(torch.arange(2.0)))
    model.register_parameter("adapter_alias", model.adapter)
    model.register_parameter(
        "inactive_adapter", torch.nn.Parameter(torch.ones(3), False)
    )
    model._axolotl_native_nvfp4_zero3_components = {"base_component"}
    model.active_adapter = "default"
    model.peft_config = {"default": object()}
    for parameter in model.parameters():
        parameter.ds_id = id(parameter)
        parameter.ds_numel = parameter.numel()

    gathered = []

    class GatheredParameters:
        def __init__(self, parameter):
            self.parameter = parameter

        def __enter__(self):
            gathered.append(self.parameter)
            return self.parameter

        def __exit__(self, *args):
            return False

    def selected(_model, state_dict, adapter_name):
        assert adapter_name == "default"
        return {
            name.replace(".default.", "."): value
            for name, value in state_dict.items()
            if name in {"adapter", "adapter_alias", "inactive_adapter"}
        }

    monkeypatch.setitem(
        __import__("sys").modules,
        "deepspeed",
        types.SimpleNamespace(
            zero=types.SimpleNamespace(GatheredParameters=GatheredParameters)
        ),
    )
    monkeypatch.setattr(save_and_load, "get_peft_model_state_dict", selected)

    state = native_nvfp4_zero3_peft_state_dict(model)

    assert set(state) == {"adapter", "adapter_alias", "inactive_adapter"}
    assert gathered == [model.adapter, model.inactive_adapter]
    assert state["adapter"] is state["adapter_alias"]
    assert all(parameter is not model.base_component for parameter in gathered)
    assert all(value.device.type == "cpu" for value in state.values())

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda destination, value: destination.__setitem__(slice(None), [value, value]),
    )
    gathered.clear()
    assert native_nvfp4_zero3_peft_state_dict(model, collect_on_this_rank=False) == {}
    assert gathered == [model.adapter, model.inactive_adapter]


def test_zero3_peft_export_uses_real_nondefault_modules_to_save_and_bias_keys(
    monkeypatch,
):
    import sys
    import types

    from peft import LoraConfig, get_peft_model
    from peft.utils.save_and_load import get_peft_model_state_dict
    from transformers import LlamaConfig, LlamaForCausalLM

    from axolotl.monkeypatch.torchao_deepspeed import (
        native_nvfp4_zero3_peft_state_dict,
    )

    model = get_peft_model(
        LlamaForCausalLM(
            LlamaConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                attention_bias=True,
                mlp_bias=True,
            )
        ),
        LoraConfig(
            r=2,
            target_modules=["q_proj"],
            modules_to_save=["lm_head"],
            bias="all",
        ),
        adapter_name="custom",
    )
    model.register_parameter(
        "native_component", torch.nn.Parameter(torch.ones(2), False)
    )
    model._axolotl_native_nvfp4_zero3_components = {"native_component"}
    for name, parameter in model.named_parameters():
        if "lora_" in name:
            parameter.requires_grad_(False)
        parameter.ds_id = id(parameter)
        parameter.ds_numel = parameter.numel()

    gathered = []

    class GatheredParameters:
        def __init__(self, parameter):
            self.parameter = parameter

        def __enter__(self):
            gathered.append(self.parameter)
            return self.parameter

        def __exit__(self, *args):
            return False

    monkeypatch.setitem(
        sys.modules,
        "deepspeed",
        types.SimpleNamespace(
            zero=types.SimpleNamespace(GatheredParameters=GatheredParameters)
        ),
    )
    original = {
        name: value
        for name, value in model.state_dict(keep_vars=True).items()
        if name != "native_component"
    }
    expected = get_peft_model_state_dict(
        model, state_dict=original, adapter_name="custom"
    )

    assert any("bias" in name for name in expected)
    state = native_nvfp4_zero3_peft_state_dict(model)

    actual = get_peft_model_state_dict(model, state_dict=state, adapter_name="custom")
    assert actual.keys() == expected.keys()
    assert any("modules_to_save" in name for name in state)
    assert any("lora_A.custom" in name for name in state)
    assert any("bias" in name for name in state)
    assert all(parameter is not model.native_component for parameter in gathered)


def test_zero3_peft_export_synchronizes_rank_zero_key_validation_error(monkeypatch):
    import sys
    import types

    from peft.utils import save_and_load

    from axolotl.monkeypatch.torchao_deepspeed import (
        native_nvfp4_zero3_peft_state_dict,
    )

    model = torch.nn.Module()
    model.register_parameter("base_component", torch.nn.Parameter(torch.ones(2), False))
    model.register_parameter("adapter", torch.nn.Parameter(torch.ones(2)))
    model._axolotl_native_nvfp4_zero3_components = {"base_component"}
    model.active_adapter = "default"
    model.peft_config = {"default": object()}
    model.adapter.ds_id = 1
    model.adapter.ds_numel = 2
    calls = 0

    class GatheredParameters:
        def __init__(self, parameter):
            self.parameter = parameter

        def __enter__(self):
            return self.parameter

        def __exit__(self, *args):
            return False

    def selected(_model, state_dict, adapter_name):
        nonlocal calls
        calls += 1
        return {"adapter": state_dict["adapter"]} if calls == 1 else {}

    def all_gather_object(destination, value):
        destination[:] = [value, value]

    monkeypatch.setitem(
        sys.modules,
        "deepspeed",
        types.SimpleNamespace(
            zero=types.SimpleNamespace(GatheredParameters=GatheredParameters)
        ),
    )
    monkeypatch.setattr(save_and_load, "get_peft_model_state_dict", selected)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)

    with pytest.raises(RuntimeError, match="adapter collection failed"):
        native_nvfp4_zero3_peft_state_dict(model)


def test_save_model_delegates_for_ordinary_model_with_save_on_each_node(monkeypatch):
    import types

    from transformers import Trainer

    from axolotl.core.trainers.mixins.distributed_parallel import (
        DistributedParallelMixin,
    )

    calls = []

    def save_model(self, output_dir=None, _internal_call=False):
        calls.append((output_dir, _internal_call))

    monkeypatch.setattr(Trainer, "save_model", save_model)

    class FakeTrainer(DistributedParallelMixin):
        pass

    trainer = object.__new__(FakeTrainer)
    trainer.model = torch.nn.Linear(2, 2)
    trainer.args = types.SimpleNamespace(save_on_each_node=True, should_save=True)
    trainer.save_model("out", True)

    assert calls == [("out", True)]
