"""CPU collective safety checks for native NVFP4 DDP setup."""

from __future__ import annotations

import copy

import pytest
import torch

from axolotl.monkeypatch.torchao_ddp import prepare_native_nvfp4_ddp


def _model(tmp_path):
    pytest.importorskip("torchao.prototype.mx_formats.nvfp4_tensor")
    from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

    from axolotl.utils.quantization import quantize_model, save_quantized_model
    from axolotl.utils.schemas.enums import TorchAOQuantDType

    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
    ).bfloat16()
    quantize_model(model, TorchAOQuantDType.nvfp4)
    save_quantized_model(model, tmp_path / "base")
    model = AutoModelForCausalLM.from_pretrained(
        tmp_path / "base", torch_dtype=torch.bfloat16
    )
    for parameter in model.parameters():
        if type(parameter).__name__ == "NVFP4Tensor":
            parameter.requires_grad_(False)
    return model


def _collectives(monkeypatch, remote):
    broadcasts = []

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def gather(result, value):
        result[:] = [value, remote(value)]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    monkeypatch.setattr(
        torch.distributed,
        "broadcast",
        lambda value, src: broadcasts.append((value, src)),
    )
    return broadcasts


@pytest.mark.parametrize(
    ("field", "replacement"),
    [(0, "missing.native.weight"), (1, (999,)), (4, "different-activation")],
)
def test_native_nvfp4_ddp_rejects_rank_layout_mismatch_before_broadcast(
    tmp_path, monkeypatch, field, replacement
):
    model = _model(tmp_path)

    def remote(value):
        layout, errors = copy.deepcopy(value)
        first = list(layout[0])
        first[field] = replacement
        layout[0] = tuple(first)
        return layout, errors

    broadcasts = _collectives(monkeypatch, remote)
    with pytest.raises(ValueError, match="identical component layouts"):
        prepare_native_nvfp4_ddp(model, torch.device("cpu"))
    assert not broadcasts


def test_native_nvfp4_ddp_rejects_remote_trainable_base_before_broadcast(
    tmp_path, monkeypatch
):
    model = _model(tmp_path)
    broadcasts = _collectives(
        monkeypatch, lambda value: (value[0], ["weight is trainable"])
    )
    with pytest.raises(ValueError, match="frozen base weights"):
        prepare_native_nvfp4_ddp(model, torch.device("cpu"))
    assert not broadcasts


def test_native_nvfp4_ddp_ignores_only_frozen_native_wrappers(tmp_path, monkeypatch):
    model = _model(tmp_path)
    broadcasts = _collectives(monkeypatch, lambda value: value)
    model._ddp_params_and_buffers_to_ignore = {"existing.frozen.parameter"}
    prepare_native_nvfp4_ddp(model, torch.device("cpu"))

    native = {
        name
        for name, parameter in model.named_parameters()
        if type(parameter).__name__ == "NVFP4Tensor"
    }
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    ignored = set(model._ddp_params_and_buffers_to_ignore)
    assert native
    assert "existing.frozen.parameter" in ignored
    assert native <= ignored
    assert not trainable & ignored
    assert broadcasts


def test_native_nvfp4_ddp_preserves_scalar_scale_and_uses_byte_broadcast(
    tmp_path, monkeypatch
):
    model = _model(tmp_path)
    native = next(
        parameter
        for parameter in model.parameters()
        if type(parameter).__name__ == "NVFP4Tensor"
    )
    if getattr(native, "per_tensor_scale", None) is None:
        pytest.skip("TorchAO build does not expose per_tensor_scale")
    original = native.per_tensor_scale
    if original.numel() != 1:
        pytest.skip("TorchAO per_tensor_scale is not scalar")
    if original.ndim:
        native.per_tensor_scale = original.reshape(())
    scalar = native.per_tensor_scale
    before = scalar.clone()
    broadcasts = _collectives(monkeypatch, lambda value: value)
    prepare_native_nvfp4_ddp(model, torch.device("cpu"))
    assert native.per_tensor_scale is scalar
    assert scalar.ndim == 0
    torch.testing.assert_close(scalar, before, rtol=0, atol=0)
    assert broadcasts
    assert all(
        value.dtype is torch.uint8 and value.ndim == 1 and value.is_contiguous()
        for value, _ in broadcasts
    )


@pytest.mark.parametrize(
    "config_key",
    ["tensor_parallel_size", "context_parallel_size", "expert_parallel_size"],
)
def test_distributed_mixin_skips_native_setup_for_parallel_layouts(
    monkeypatch, config_key
):
    import types

    from transformers import Trainer

    from axolotl.core.trainers.mixins.distributed_parallel import (
        DistributedParallelMixin,
    )

    called = []
    monkeypatch.setattr(Trainer, "_wrap_model", lambda _, model, *args, **kwargs: model)
    monkeypatch.setattr(
        "axolotl.monkeypatch.torchao_ddp.prepare_native_nvfp4_ddp",
        lambda *args: called.append(args),
    )
    model = torch.nn.Linear(1, 1)

    class TestTrainer(DistributedParallelMixin):
        pass

    trainer = object.__new__(TestTrainer)
    trainer.accelerator = types.SimpleNamespace(
        distributed_type=types.SimpleNamespace(name="MULTI_GPU"),
        parallelism_config=types.SimpleNamespace(
            tp_enabled=False, cp_enabled=False, dp_shard_enabled=False
        ),
        device=torch.device("cpu"),
    )
    trainer.axolotl_cfg = types.SimpleNamespace(**{config_key: 2})
    assert trainer._wrap_model(model) is model
    assert not called


def _mixin_trainer(distributed_type="MULTI_GPU", **config):
    import types

    from axolotl.core.trainers.mixins.distributed_parallel import (
        DistributedParallelMixin,
    )

    class TestTrainer(DistributedParallelMixin):
        pass

    trainer = object.__new__(TestTrainer)
    trainer.accelerator = types.SimpleNamespace(
        distributed_type=distributed_type,
        parallelism_config=types.SimpleNamespace(
            tp_enabled=False, cp_enabled=False, dp_shard_enabled=False
        ),
        device=torch.device("cpu"),
    )
    trainer.axolotl_cfg = types.SimpleNamespace(**config)
    return trainer


def test_distributed_mixin_dispatches_only_unprepared_pure_ddp(monkeypatch):
    from transformers import Trainer

    called = []
    monkeypatch.setattr(Trainer, "_wrap_model", lambda _, model, *args, **kwargs: model)
    monkeypatch.setattr(
        "axolotl.monkeypatch.torchao_ddp.prepare_native_nvfp4_ddp",
        lambda *args: called.append(args) or True,
    )
    trainer = _mixin_trainer()
    model = torch.nn.Linear(1, 1)
    assert trainer._wrap_model(model) is model
    assert called == [(model, torch.device("cpu"))]
    assert model._axolotl_native_nvfp4_ddp_prepared
    assert trainer._wrap_model(model) is model
    assert len(called) == 1


def test_distributed_mixin_skips_non_ddp(monkeypatch):
    from transformers import Trainer

    called = []
    monkeypatch.setattr(Trainer, "_wrap_model", lambda _, model, *args, **kwargs: model)
    monkeypatch.setattr(
        "axolotl.monkeypatch.torchao_ddp.prepare_native_nvfp4_ddp",
        lambda *args: called.append(args),
    )
    model = torch.nn.Linear(1, 1)
    assert _mixin_trainer("NO")._wrap_model(model) is model
    assert not called


def test_native_nvfp4_deepspeed_filters_only_tagged_frozen_parameters():
    import types

    from axolotl.monkeypatch.torchao_deepspeed import (
        _install_native_nvfp4_broadcast_filter,
    )

    native = type("NVFP4Tensor", (types.SimpleNamespace,), {})(requires_grad=False)
    adapter = types.SimpleNamespace(requires_grad=True)
    remove_duplicate_calls = []

    class Model:
        _axolotl_native_nvfp4_deepspeed_names = {"base.weight"}

        def named_parameters(self, remove_duplicate=True):
            remove_duplicate_calls.append(remove_duplicate)
            return iter((("base.weight", native), ("adapter.weight", adapter)))

    class Engine:
        def __init__(self):
            self.module = Model()
            self.broadcast = []

        def _broadcast_model(self):
            self.broadcast.extend(name for name, _ in self.module.named_parameters())

    _install_native_nvfp4_broadcast_filter(Engine)
    engine = Engine()
    engine._broadcast_model()
    assert engine.broadcast == ["adapter.weight"]
    assert remove_duplicate_calls[:2] == [False, True]
    assert list(engine.module.named_parameters()) == [
        ("base.weight", native),
        ("adapter.weight", adapter),
    ]


def test_native_nvfp4_deepspeed_preserves_unmarked_engine_broadcast():
    from axolotl.monkeypatch.torchao_deepspeed import (
        _install_native_nvfp4_broadcast_filter,
    )

    class Engine:
        def __init__(self):
            self.module = object()
            self.called = 0

        def _broadcast_model(self):
            self.called += 1

    _install_native_nvfp4_broadcast_filter(Engine)
    engine = Engine()
    engine._broadcast_model()
    assert engine.called == 1


def test_distributed_mixin_dispatches_deepspeed_zero12_only(monkeypatch):
    import types

    from transformers import Trainer

    from axolotl.core.trainers.mixins.distributed_parallel import (
        DistributedParallelMixin,
    )

    called = []
    monkeypatch.setattr(Trainer, "_wrap_model", lambda _, model, *args, **kwargs: model)
    monkeypatch.setattr(
        "axolotl.monkeypatch.torchao_deepspeed.prepare_native_nvfp4_deepspeed",
        lambda *args: called.append(args) or True,
    )

    class TestTrainer(DistributedParallelMixin):
        pass

    trainer = object.__new__(TestTrainer)
    trainer.accelerator = types.SimpleNamespace(
        distributed_type=types.SimpleNamespace(name="DEEPSPEED"),
        parallelism_config=types.SimpleNamespace(
            tp_enabled=False, cp_enabled=False, dp_shard_enabled=False
        ),
        device=torch.device("cpu"),
        state=types.SimpleNamespace(
            deepspeed_plugin=types.SimpleNamespace(
                deepspeed_config={"zero_optimization": {"stage": 2}}
            )
        ),
    )
    trainer.axolotl_cfg = types.SimpleNamespace(
        tensor_parallel_size=1, context_parallel_size=1, expert_parallel_size=1
    )
    model = torch.nn.Linear(1, 1)
    assert trainer._wrap_model(model) is model
    assert called == [(model, torch.device("cpu"), 2)]
    assert model._axolotl_native_nvfp4_deepspeed_prepared
