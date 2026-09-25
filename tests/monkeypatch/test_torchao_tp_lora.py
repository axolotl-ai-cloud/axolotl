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


def test_prepare_opt_out_keeps_tp_routing_for_ordinary_lora(monkeypatch):
    model = _LoRA()
    monkeypatch.setattr(torchao_tp_lora, "_is_native_nvfp4_dtensor", lambda _: True)
    monkeypatch.setattr(torchao_tp_lora, "_lora_tp_plan", lambda *_: "colwise")
    monkeypatch.setattr(torchao_tp_lora.dist, "get_world_size", lambda group: 1)
    called = []

    def ordinary(module, layout, process_group):
        called.append((module, layout, process_group))
        return module.forward

    monkeypatch.setattr(torchao_tp_lora, "_ordinary_lora_tp_forward", ordinary)

    assert torchao_tp_lora.prepare_native_nvfp4_tp_lora(model, merge_aware=False)
    assert model._axolotl_merge_aware_unsupported
    assert called == [(model, "colwise", "tp")]


def test_prepare_dynamic_quantization_uses_ordinary_tp_fallback(monkeypatch):
    model = _LoRA()
    model.base_layer.weight.to_local = lambda: SimpleNamespace(
        shape=(4, 4), act_quant_kwargs={"scale": "dynamic"}
    )
    monkeypatch.setattr(torchao_tp_lora, "_is_native_nvfp4_dtensor", lambda _: True)
    monkeypatch.setattr(torchao_tp_lora, "_lora_tp_plan", lambda *_: "colwise")
    monkeypatch.setattr(torchao_tp_lora.dist, "get_world_size", lambda group: 1)
    called = []

    def ordinary(module, layout, process_group):
        called.append((module, layout, process_group))
        return module.forward

    monkeypatch.setattr(torchao_tp_lora, "_ordinary_lora_tp_forward", ordinary)

    assert torchao_tp_lora.prepare_native_nvfp4_tp_lora(model)
    assert model._axolotl_merge_aware_unsupported
    assert called == [(model, "colwise", "tp")]


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


def _native_weight(rows=128, columns=64, *, swizzled=False):
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    generator = torch.Generator().manual_seed(rows * 1000 + columns)
    return NVFP4Tensor.to_nvfp4(
        torch.randn(rows, columns, generator=generator),
        per_tensor_scale=torch.tensor(1.0),
        is_swizzled_scales=swizzled,
    )


@pytest.mark.parametrize("shard_dim", [0, 1])
@pytest.mark.parametrize("swizzled", [False, True])
def test_native_effective_weight_tp_quantization_matches_full(shard_dim, swizzled):
    from axolotl.monkeypatch.torchao_nvfp4_merge import quantize_native_effective_weight
    from axolotl.monkeypatch.torchao_tp import materialize_native_nvfp4_tp

    base = _native_weight(swizzled=swizzled)
    rank = 4
    generator = torch.Generator().manual_seed(41)
    lora_a = torch.randn(rank, base.shape[1], generator=generator)
    lora_b = torch.randn(base.shape[0], rank, generator=generator)
    full = quantize_native_effective_weight(base, lora_a, lora_b, 0.25)
    locals_ = []
    for tp_rank in range(2):
        local_base = materialize_native_nvfp4_tp(
            [("weight", base)], {"weight": shard_dim}, tp_rank, 2
        )["weight"]
        local_a = lora_a if shard_dim == 0 else lora_a.chunk(2, dim=1)[tp_rank]
        local_b = lora_b.chunk(2, dim=0)[tp_rank] if shard_dim == 0 else lora_b
        locals_.append(
            quantize_native_effective_weight(local_base, local_a, local_b, 0.25)
        )

    rebuilt = torch.cat([weight.dequantize() for weight in locals_], dim=shard_dim)
    torch.testing.assert_close(rebuilt, full.dequantize(), rtol=0, atol=0)
    torch.testing.assert_close(
        torch.cat([weight.qdata for weight in locals_], dim=shard_dim),
        full.qdata,
        rtol=0,
        atol=0,
    )
    from axolotl.monkeypatch.torchao_tp import _logical_scale_tensor

    torch.testing.assert_close(
        torch.cat(
            [_logical_scale_tensor(weight).scale for weight in locals_], dim=shard_dim
        ),
        _logical_scale_tensor(full).scale,
        rtol=0,
        atol=0,
    )


def test_tp_export_gathers_sharded_factor_and_checks_replica(monkeypatch):
    model = _LoRA()
    model._axolotl_native_nvfp4_tp_lora_prepared = True
    monkeypatch.setattr(torchao_tp_lora, "_is_native_nvfp4_dtensor", lambda _: True)
    monkeypatch.setattr(torchao_tp_lora, "_lora_tp_plan", lambda *_: "colwise")
    monkeypatch.setattr(torchao_tp_lora.dist, "get_world_size", lambda group: 2)

    def all_gather(output, value, *, group):
        output[0].copy_(value)
        if value.shape == model.lora_B["default"].weight.shape:
            output[1].copy_(value + 10)
        else:
            output[1].copy_(value)

    monkeypatch.setattr(torchao_tp_lora.dist, "all_gather", all_gather)
    state = torchao_tp_lora.native_nvfp4_tp_peft_state_dict(
        model, collect_on_this_rank=True
    )

    assert state is not None
    torch.testing.assert_close(
        state["lora_B.default.weight"],
        torch.cat(
            [model.lora_B["default"].weight, model.lora_B["default"].weight + 10]
        ),
    )
    torch.testing.assert_close(
        state["lora_A.default.weight"], model.lora_A["default"].weight
    )


def test_tp_export_rejects_divergent_replicas(monkeypatch):
    model = _LoRA()
    model._axolotl_native_nvfp4_tp_lora_prepared = True
    monkeypatch.setattr(torchao_tp_lora, "_is_native_nvfp4_dtensor", lambda _: True)
    monkeypatch.setattr(torchao_tp_lora, "_lora_tp_plan", lambda *_: "colwise")
    monkeypatch.setattr(torchao_tp_lora.dist, "get_world_size", lambda group: 2)

    def all_gather(output, value, *, group):
        output[0].copy_(value)
        output[1].copy_(value + 1)

    monkeypatch.setattr(torchao_tp_lora.dist, "all_gather", all_gather)
    with pytest.raises(RuntimeError, match="differs across TP ranks"):
        torchao_tp_lora.native_nvfp4_tp_peft_state_dict(
            model, collect_on_this_rank=True
        )


def test_tp_export_gathers_rowwise_lora_a(monkeypatch):
    model = _LoRA()
    object.__setattr__(
        model.base_layer,
        "weight",
        SimpleNamespace(
            placements=(Shard(1),),
            shape=(8, 4),
            device_mesh=SimpleNamespace(get_group=lambda: "tp"),
            to_local=lambda: SimpleNamespace(shape=(8, 2)),
        ),
    )
    model.lora_A["default"] = torch.nn.Linear(2, 2, bias=False)
    model.lora_B["default"] = torch.nn.Linear(2, 8, bias=False)
    model._axolotl_native_nvfp4_tp_lora_prepared = True
    monkeypatch.setattr(torchao_tp_lora, "_is_native_nvfp4_dtensor", lambda _: True)
    monkeypatch.setattr(torchao_tp_lora, "_lora_tp_plan", lambda *_: "rowwise")
    monkeypatch.setattr(torchao_tp_lora.dist, "get_world_size", lambda group: 2)

    def all_gather(output, value, *, group):
        output[0].copy_(value)
        if value.shape == model.lora_A["default"].weight.shape:
            output[1].copy_(value + 10)
        else:
            output[1].copy_(value)

    monkeypatch.setattr(torchao_tp_lora.dist, "all_gather", all_gather)
    state = torchao_tp_lora.native_nvfp4_tp_peft_state_dict(
        model, collect_on_this_rank=True
    )

    torch.testing.assert_close(
        state["lora_A.default.weight"],
        torch.cat(
            [model.lora_A["default"].weight, model.lora_A["default"].weight + 10],
            dim=1,
        ),
    )
    torch.testing.assert_close(
        state["lora_B.default.weight"], model.lora_B["default"].weight
    )


def test_tp_export_ignores_models_without_prepared_native_modules():
    class Plain(torch.nn.Module):
        def named_parameters(self, *args, **kwargs):
            raise AssertionError("ordinary adapter state must not be materialized")

    assert (
        torchao_tp_lora.native_nvfp4_tp_peft_state_dict(
            Plain(), collect_on_this_rank=False
        )
        is None
    )


def test_tp_export_gathers_uneven_output_shards(monkeypatch):
    value = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    monkeypatch.setattr(torchao_tp_lora.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(torchao_tp_lora.dist, "get_world_size", lambda group: 2)
    calls = 0

    def all_gather(output, input_, *, group):
        nonlocal calls
        calls += 1
        if calls == 1:
            output[0].copy_(torch.tensor([3, 2]))
            output[1].copy_(torch.tensor([1, 2]))
            return
        output[0].copy_(input_)
        output[1].zero_()
        output[1][0].fill_(9)

    monkeypatch.setattr(torchao_tp_lora.dist, "all_gather", all_gather)
    gathered = torchao_tp_lora._all_gather_factor(value, "tp", shard_dim=0)

    assert [tuple(item.shape) for item in gathered] == [(3, 2), (1, 2)]
    torch.testing.assert_close(gathered[0], value)
    torch.testing.assert_close(gathered[1], torch.full((1, 2), 9.0))
