from types import SimpleNamespace

import pytest
import torch

from axolotl.monkeypatch import torchao_tp_lora


class Shard:
    def __init__(self, dim):
        self.dim = dim


class Replicate:
    pass


class Partial:
    pass


def _install_collective_spy(monkeypatch):
    calls = []
    monkeypatch.setattr(torchao_tp_lora.dist, "get_world_size", lambda group: 2)

    def all_reduce(value, *, group):
        calls.append(group)
        value.add_(3)

    monkeypatch.setattr(torchao_tp_lora.dist, "all_reduce", all_reduce)
    return calls


def test_rowwise_reduce_is_identity_in_backward(monkeypatch):
    calls = _install_collective_spy(monkeypatch)
    value = torch.tensor([2.0], requires_grad=True)

    output = torchao_tp_lora._AllReduceForwardIdentityBackward.apply(value, "tp")
    torch.testing.assert_close(output, torch.tensor([5.0]))
    output.sum().backward()

    torch.testing.assert_close(value.grad, torch.ones_like(value))
    assert calls == ["tp"]


def test_colwise_input_reduce_happens_only_in_backward(monkeypatch):
    calls = _install_collective_spy(monkeypatch)
    value = torch.tensor([2.0], requires_grad=True)

    output = torchao_tp_lora._IdentityForwardAllReduceBackward.apply(value, "tp")
    torch.testing.assert_close(output, value.detach())
    output.sum().backward()

    torch.testing.assert_close(value.grad, torch.tensor([4.0]))
    assert calls == ["tp"]


def test_tp_lora_layout_rejects_partial_weight_placement():
    weight = SimpleNamespace(
        placements=(Partial(),), device_mesh=SimpleNamespace(ndim=1)
    )

    with pytest.raises(ValueError, match="output- or input-axis TP shard"):
        torchao_tp_lora._tp_lora_layout(weight, None)


class _Base(torch.nn.Module):
    pass


class _LoRA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base_layer = _Base()
        object.__setattr__(
            self.base_layer,
            "weight",
            SimpleNamespace(
                placements=(Shard(0),),
                shape=(8, 4),
                device_mesh=SimpleNamespace(get_group=lambda: "tp"),
                to_local=lambda: SimpleNamespace(shape=(4, 4)),
            ),
        )
        self.lora_A = torch.nn.ModuleDict(
            {"default": torch.nn.Linear(4, 2, bias=False)}
        )
        self.lora_B = torch.nn.ModuleDict(
            {"default": torch.nn.Linear(2, 4, bias=False)}
        )
        self.lora_dropout = torch.nn.ModuleDict({"default": torch.nn.Identity()})
        self.lora_variant = {}
        self.use_dora = {}


def test_prepare_is_idempotent_and_uses_factor_gradient_hook(monkeypatch):
    model = _LoRA()
    monkeypatch.setattr(torchao_tp_lora, "_is_native_nvfp4_dtensor", lambda _: True)
    monkeypatch.setattr(torchao_tp_lora, "_lora_tp_plan", lambda *_: "colwise")
    monkeypatch.setattr(torchao_tp_lora.dist, "get_world_size", lambda group: 1)

    assert torchao_tp_lora.prepare_native_nvfp4_tp_lora(model)
    hook_count = len(model.lora_A["default"].weight._backward_hooks)
    assert torchao_tp_lora.prepare_native_nvfp4_tp_lora(model)
    assert len(model.lora_A["default"].weight._backward_hooks) == hook_count

    model.lora_A["second"] = torch.nn.Linear(4, 2, bias=False)
    model.lora_B["second"] = torch.nn.Linear(2, 4, bias=False)
    model.lora_dropout["second"] = torch.nn.Identity()
    assert torchao_tp_lora.prepare_native_nvfp4_tp_lora(model)
    assert len(model.lora_A["second"].weight._backward_hooks) == 1


def test_prepare_rejects_unsynchronized_dropout(monkeypatch):
    model = _LoRA()
    model.lora_dropout["default"] = torch.nn.Dropout(0.1)
    monkeypatch.setattr(torchao_tp_lora, "_is_native_nvfp4_dtensor", lambda _: True)
    monkeypatch.setattr(torchao_tp_lora, "_lora_tp_plan", lambda *_: "colwise")

    with pytest.raises(ValueError, match="lora_dropout: 0"):
        torchao_tp_lora.prepare_native_nvfp4_tp_lora(model)


def test_prepare_rejects_gathered_colwise_plan(monkeypatch):
    model = _LoRA()
    monkeypatch.setattr(torchao_tp_lora, "_is_native_nvfp4_dtensor", lambda _: True)
    monkeypatch.setattr(
        torchao_tp_lora, "_lora_tp_plan", lambda *_: "colwise_gather_output"
    )

    with pytest.raises(ValueError, match="colwise and rowwise"):
        torchao_tp_lora.prepare_native_nvfp4_tp_lora(model)


def test_tp_plan_resolver_normalizes_peft_module_prefix():
    base_model = SimpleNamespace(tp_plan={"model.layers.*.self_attn.q_proj": "colwise"})
    model = SimpleNamespace(get_base_model=lambda: base_model)

    assert (
        torchao_tp_lora._lora_tp_plan(
            model, "base_model.model.model.layers.0.self_attn.q_proj"
        )
        == "colwise"
    )
    assert (
        torchao_tp_lora._lora_tp_plan(
            model, "base_model.model.model.layers.0.self_attn.v_proj"
        )
        is None
    )
