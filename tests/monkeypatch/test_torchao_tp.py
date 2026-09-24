"""CPU checks for native NVFP4 tensor-parallel component slicing."""

import pytest
import torch

import axolotl.monkeypatch.torchao_tp as torchao_tp
from axolotl.monkeypatch.torchao_tp import (
    materialize_native_nvfp4_tp,
    native_nvfp4_tp_shard,
    preflight_native_nvfp4_tp,
)


def _nvfp4(rows=256, columns=128, *, swizzled=False):
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    generator = torch.Generator().manual_seed(rows * 10_000 + columns)
    value = torch.randn(rows, columns, generator=generator, dtype=torch.float32)
    return NVFP4Tensor.to_nvfp4(
        value,
        per_tensor_scale=torch.tensor(1.0),
        is_swizzled_scales=swizzled,
    )


@pytest.mark.parametrize("swizzled", [False, True])
@pytest.mark.parametrize("dim", [0, 1])
def test_native_nvfp4_tp_slice_preserves_component_parity(swizzled, dim):
    tensor = _nvfp4(swizzled=swizzled)
    shards = [native_nvfp4_tp_shard(tensor, dim, rank, 2) for rank in range(2)]
    local = [
        materialize_native_nvfp4_tp([("weight", tensor)], {"weight": dim}, rank, 2)[
            "weight"
        ]
        for rank in range(2)
    ]

    rebuilt = torch.cat([item.dequantize() for item in local], dim=dim)
    torch.testing.assert_close(rebuilt, tensor.dequantize(), rtol=0, atol=0)
    assert all(type(item).__name__ == "NVFP4Tensor" for item in local)
    assert all(item.per_tensor_scale is not tensor.per_tensor_scale for item in local)
    assert all(
        item.qdata.untyped_storage().nbytes() == item.qdata.numel() for item in local
    )
    assert all(
        item.scale.untyped_storage().nbytes() == item.scale.numel() for item in local
    )
    assert (
        [(item.start, item.end) for item in shards] == [(0, 128), (128, 256)]
        if dim == 0
        else [(0, 64), (64, 128)]
    )


def test_native_nvfp4_tp_rejects_unaligned_input_blocks_without_mutation():
    tensor = _nvfp4(columns=80)
    qdata = tensor.qdata.clone()
    scale = tensor.scale.clone()
    with pytest.raises(ValueError, match="align to 16"):
        materialize_native_nvfp4_tp([("weight", tensor)], {"weight": 1}, 0, 2)
    torch.testing.assert_close(tensor.qdata, qdata)
    torch.testing.assert_close(tensor.scale, scale)


def test_native_nvfp4_tp_logicalizes_swizzled_scales_for_output_shards():
    tensor = _nvfp4(rows=192, swizzled=True)
    local = [
        materialize_native_nvfp4_tp([("weight", tensor)], {"weight": 0}, rank, 2)[
            "weight"
        ]
        for rank in range(2)
    ]
    assert all(item.is_swizzled_scales for item in local)
    torch.testing.assert_close(
        torch.cat([item.dequantize() for item in local]),
        tensor.dequantize(),
        rtol=0,
        atol=0,
    )


def test_native_nvfp4_tp_preflight_rejects_alias_placement_disagreement_without_mutation():
    tensor = _nvfp4()
    with pytest.raises(ValueError, match="aliases disagree"):
        preflight_native_nvfp4_tp(
            [("left.weight", tensor), ("right.weight", tensor)],
            {"left.weight": 0, "right.weight": 1},
            0,
            2,
        )
    assert tensor.shape == (256, 128)


def test_native_nvfp4_tp_rejects_invalid_dimension_without_mutation():
    tensor = _nvfp4()
    with pytest.raises(ValueError, match="must be 0 or 1"):
        native_nvfp4_tp_shard(tensor, 2, 0, 2)
    assert tensor.shape == (256, 128)


def test_native_nvfp4_tp_materializes_tied_alias_once_with_owned_storage():
    tensor = _nvfp4()
    shards = materialize_native_nvfp4_tp(
        [("left.weight", tensor), ("right.weight", tensor)],
        {"left.weight": 0, "right.weight": 0},
        0,
        2,
    )
    assert shards["left.weight"] is shards["right.weight"]
    assert (
        shards["left.weight"].qdata.untyped_storage().data_ptr()
        != tensor.qdata.untyped_storage().data_ptr()
    )


@pytest.mark.parametrize("dim, expected", [(-2, 0), (-1, 1)])
def test_native_nvfp4_tp_accepts_standard_negative_dimensions(dim, expected):
    tensor = _nvfp4()
    assert native_nvfp4_tp_shard(tensor, dim, 0, 2).dim == expected


def test_native_nvfp4_tp_allows_block_aligned_swizzled_input_shards():
    tensor = _nvfp4(columns=96, swizzled=True)
    local = [
        materialize_native_nvfp4_tp([("weight", tensor)], {"weight": 1}, rank, 3)[
            "weight"
        ]
        for rank in range(3)
    ]
    assert [item.shape[1] for item in local] == [32, 32, 32]
    torch.testing.assert_close(
        torch.cat([item.dequantize() for item in local], dim=1),
        tensor.dequantize(),
        rtol=0,
        atol=0,
    )


def test_native_nvfp4_tp_preflight_does_not_slice_before_later_failure(monkeypatch):
    valid = _nvfp4()
    invalid = _nvfp4(columns=80)
    calls = []
    original = torchao_tp.slice_native_nvfp4_tp

    def record(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(torchao_tp, "slice_native_nvfp4_tp", record)
    with pytest.raises(ValueError, match="align to 16"):
        materialize_native_nvfp4_tp(
            [("valid.weight", valid), ("invalid.weight", invalid)],
            {"valid.weight": 0, "invalid.weight": 1},
            0,
            2,
        )
    assert calls == []


class _Mesh:
    def __init__(self, rank, size):
        self.rank = rank
        self.world_size = size

    def get_local_rank(self):
        return self.rank

    def size(self):
        return self.world_size


def test_native_nvfp4_tp_converter_shards_after_deserialization():
    from concurrent.futures import Future
    from types import SimpleNamespace

    from transformers.core_model_loading import ConversionOps, WeightConverter

    events = []
    tensor = _nvfp4(rows=64, columns=64, swizzled=True)

    class DeserializeNative(ConversionOps):
        def convert(self, input_dict, full_layer_name, **kwargs):
            events.append(("deserialize", next(iter(input_dict.values()))[0].shape))
            return {full_layer_name: tensor}

    base = WeightConverter("_weight_qdata", "weight", [DeserializeNative()])
    converter = torchao_tp._native_nvfp4_weight_converter(base, _Mesh(1, 2))
    component = converter.distributed_operation.shard_tensor(tensor.qdata, device="cpu")
    events.append(("component", component.shape))
    future = Future()
    future.set_result(component)
    converter.add_tensor("proj.weight", "proj._weight_qdata", "_weight_qdata", future)
    expected = native_nvfp4_tp_shard(tensor, 0, 1, 2, name="proj.weight")
    model = SimpleNamespace(
        tp_plan={"proj": "colwise"},
        _axolotl_native_nvfp4_tp_manifest={"proj.weight": expected},
    )

    loaded = converter.convert("proj.weight", model=model)["proj.weight"]

    assert events == [
        ("component", tensor.qdata.shape),
        ("deserialize", tensor.qdata.shape),
    ]
    assert type(loaded).__name__ == "NVFP4Tensor"
    assert loaded.shape == (32, 64)
    assert loaded.qdata.untyped_storage().nbytes() == loaded.qdata.numel()
    assert converter.distributed_operation.dim == 0


def test_native_nvfp4_tp_converter_rejects_shape_that_disagrees_with_manifest():
    from concurrent.futures import Future
    from types import SimpleNamespace

    from transformers.core_model_loading import ConversionOps, WeightConverter

    tensor = _nvfp4(rows=64, columns=64)

    class DeserializeNative(ConversionOps):
        def convert(self, input_dict, full_layer_name, **kwargs):
            return {full_layer_name: tensor}

    base = WeightConverter("_weight_qdata", "weight", [DeserializeNative()])
    converter = torchao_tp._native_nvfp4_weight_converter(base, _Mesh(0, 2))
    future = Future()
    future.set_result(tensor.qdata)
    converter.add_tensor("proj.weight", "proj._weight_qdata", "_weight_qdata", future)
    model = SimpleNamespace(
        tp_plan={"proj": "colwise"},
        _axolotl_native_nvfp4_tp_manifest={
            "proj.weight": torchao_tp.NativeNVFP4TPShard("proj.weight", 0, 0, 16)
        },
    )

    with pytest.raises(ValueError, match="changed after preflight"):
        converter.convert("proj.weight", model=model)


def test_native_nvfp4_tp_context_preflights_before_replacing_converters(monkeypatch):
    from types import SimpleNamespace

    from transformers.core_model_loading import ConversionOps, WeightConverter
    from transformers.quantizers.quantizer_torchao import TorchAoHfQuantizer

    class Identity(ConversionOps):
        def convert(self, input_dict, **kwargs):
            return input_dict

    events = []
    base = WeightConverter("_weight_qdata", "weight", [Identity()])
    quantizer = object.__new__(TorchAoHfQuantizer)
    quantizer.quantization_config = SimpleNamespace(
        quant_type=type("NVFP4WeightOnlyConfig", (), {})()
    )
    parameter = torch.nn.Parameter(torch.empty(64, 64, device="meta"))

    class Model:
        tp_plan = {"proj": "rowwise"}

        def named_parameters(self, remove_duplicate=False):
            assert not remove_duplicate
            return [("proj.weight", parameter)]

    def get_weight_conversions(_):
        events.append("converters")
        return [base]

    def preprocess(_, model, **kwargs):
        del model, kwargs
        events.append("preprocess")

    monkeypatch.setattr(
        TorchAoHfQuantizer, "get_weight_conversions", get_weight_conversions
    )
    monkeypatch.setattr(
        TorchAoHfQuantizer,
        "_process_model_before_weight_loading",
        preprocess,
    )
    monkeypatch.setattr(
        TorchAoHfQuantizer,
        "param_needs_quantization",
        lambda _, __, ___: True,
    )

    with torchao_tp.native_nvfp4_tp_checkpoint_loading(_Mesh(1, 2)):
        quantizer._process_model_before_weight_loading(Model())
        converters = quantizer.get_weight_conversions()

    assert events == ["preprocess", "converters"]
    assert type(converters[0]).__name__ == "NativeNVFP4WeightConverter"
    assert converters[0].distributed_operation.device_mesh.world_size == 2


def test_native_nvfp4_tp_detects_saved_hf_quantization_config():
    from torchao.prototype.mx_formats import NVFP4WeightOnlyConfig
    from transformers import TorchAoConfig

    from axolotl.loaders.model import _is_native_nvfp4_quantization_config

    serialized = TorchAoConfig(NVFP4WeightOnlyConfig()).to_dict()
    assert serialized["quant_type"]["default"]["_type"] == "NVFP4WeightOnlyConfig"
    assert _is_native_nvfp4_quantization_config(serialized)
    assert not _is_native_nvfp4_quantization_config({"quant_method": "torchao"})


def test_native_nvfp4_tp_converter_uses_real_torchao_deserializer():
    from concurrent.futures import Future
    from types import SimpleNamespace

    from torchao.prototype.safetensors.safetensors_support import (
        flatten_tensor_state_dict,
    )
    from transformers.core_model_loading import WeightConverter
    from transformers.integrations.torchao import TorchAoDeserialize

    tensor = _nvfp4(rows=64, columns=64, swizzled=True)
    flattened, metadata = flatten_tensor_state_dict({"proj.weight": tensor})
    module = torch.nn.Module()
    module.proj = torch.nn.Linear(64, 64, bias=False)
    quantizer = SimpleNamespace(metadata=metadata)
    source_patterns = [key.removeprefix("proj.") for key in flattened]
    base = WeightConverter(source_patterns, "weight", [TorchAoDeserialize(quantizer)])
    converter = torchao_tp._native_nvfp4_weight_converter(base, _Mesh(1, 2))
    for key, value in flattened.items():
        future = Future()
        component = converter.distributed_operation.shard_tensor(value, device="cpu")
        future.set_result(component)
        converter.add_tensor("proj.weight", key, key.removeprefix("proj."), future)
    expected = native_nvfp4_tp_shard(tensor, 0, 1, 2, name="proj.weight")
    module.tp_plan = {"proj": "colwise"}
    module._axolotl_native_nvfp4_tp_manifest = {"proj.weight": expected}

    loaded = converter.convert("proj.weight", model=module)["proj.weight"]

    assert type(loaded).__name__ == "NVFP4Tensor"
    assert loaded.shape == (32, 64)
    assert loaded.is_swizzled_scales
    torch.testing.assert_close(
        loaded.dequantize(), tensor.dequantize()[32:], rtol=0, atol=0
    )


def test_native_nvfp4_metadata_detection_is_exact_and_recursive():
    assert torchao_tp._metadata_has_nvfp4(
        {"_type": "NVFP4Tensor", "_data": {"block_size": 16}}
    )
    assert torchao_tp._metadata_has_nvfp4(
        [{"_type": "Tensor"}, {"nested": {"_type": "NVFP4Tensor"}}]
    )
    assert not torchao_tp._metadata_has_nvfp4(
        {"_type": "Tensor", "description": "NVFP4Tensor text only"}
    )


def test_native_nvfp4_dtensor_interception_delegates_unrelated_bf16(monkeypatch):
    from transformers.core_model_loading import DtensorShardOperation

    calls = []

    def original(operation, tensor, *args, **kwargs):
        calls.append(tensor)
        return "bf16-shard"

    monkeypatch.setattr(DtensorShardOperation, "shard_tensor", original)
    native = object.__new__(DtensorShardOperation)
    native._axolotl_native_nvfp4_component = True
    bf16 = object.__new__(DtensorShardOperation)
    bf16._axolotl_native_nvfp4_component = False
    scalar = torch.tensor(1.0)
    dense = torch.ones(16, 16, dtype=torch.bfloat16)
    with torchao_tp.native_nvfp4_tp_checkpoint_loading(_Mesh(0, 2)):
        assert DtensorShardOperation.shard_tensor(native, scalar).item() == 1.0
        assert DtensorShardOperation.shard_tensor(bf16, dense) == "bf16-shard"
    assert calls == [dense]


def test_native_nvfp4_tp_converter_uses_dtensor_destination_for_meta_target(
    monkeypatch,
):
    from concurrent.futures import Future

    from transformers.core_model_loading import ConversionOps, WeightConverter

    tensor = _nvfp4(rows=64, columns=64)

    class DeserializeNative(ConversionOps):
        def convert(self, input_dict, full_layer_name, **kwargs):
            return {full_layer_name: tensor}

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(64, 64, bias=False, device="meta")
            self.tp_plan = {"proj": "colwise"}
            self._axolotl_native_nvfp4_tp_manifest = {
                "proj.weight": native_nvfp4_tp_shard(
                    tensor, 0, 0, 2, name="proj.weight"
                )
            }

    model = Model()
    destination = torch.device("cpu")
    base = WeightConverter("_weight_qdata", "weight", [DeserializeNative()])
    converter = torchao_tp._native_nvfp4_weight_converter(
        base, _Mesh(0, 2), {id(model.proj.weight): destination}
    )
    future = Future()
    future.set_result(tensor.qdata)
    converter.add_tensor("proj.weight", "proj._weight_qdata", "_weight_qdata", future)
    moves = []
    original_to = type(tensor).to

    def record_to(value, *args, **kwargs):
        moves.append(args[0] if args else kwargs.get("device"))
        return original_to(value, *args, **kwargs)

    monkeypatch.setattr(type(tensor), "to", record_to)
    loaded = converter.convert("proj.weight", model=model)["proj.weight"]

    assert model.proj.weight.device.type == "meta"
    assert loaded.device == destination
    assert destination in moves


def test_native_nvfp4_tp_context_preserves_classmethod_descriptor_and_subclass_dispatch(
    monkeypatch,
):
    import inspect

    from transformers import PreTrainedModel

    calls = []

    class Model:
        _axolotl_native_nvfp4_tp_manifest = {}

        def named_parameters(self, remove_duplicate=False):
            return []

    def original(cls, model, *args, **kwargs):
        calls.append(cls)
        return model

    monkeypatch.setattr(
        PreTrainedModel, "maybe_distribute_model", classmethod(original)
    )
    descriptor = inspect.getattr_static(PreTrainedModel, "maybe_distribute_model")

    class Child(PreTrainedModel):
        pass

    Child.maybe_distribute_model(Model())
    with pytest.raises(RuntimeError, match="cleanup"):
        with torchao_tp.native_nvfp4_tp_checkpoint_loading(_Mesh(0, 2)):
            Child.maybe_distribute_model(Model())
            raise RuntimeError("cleanup")
    Child.maybe_distribute_model(Model())

    assert calls == [Child, Child, Child]
    assert (
        inspect.getattr_static(PreTrainedModel, "maybe_distribute_model") is descriptor
    )
    assert "maybe_distribute_model" not in Child.__dict__


def test_native_nvfp4_tp_context_tags_detached_native_state_dict_values(monkeypatch):
    from transformers import PreTrainedModel
    from transformers.core_model_loading import DtensorShardOperation

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(64, 64, bias=False)
            self._axolotl_native_nvfp4_tp_manifest = {"proj.weight": object()}

    class Child(PreTrainedModel):
        pass

    model = Model()
    monkeypatch.setattr(
        PreTrainedModel,
        "maybe_distribute_model",
        classmethod(lambda cls, model, *args, **kwargs: model),
    )
    monkeypatch.setattr(DtensorShardOperation, "__init__", lambda operation, _: None)

    with torchao_tp.native_nvfp4_tp_checkpoint_loading(_Mesh(0, 2)):
        Child.maybe_distribute_model(model)
        detached = model.state_dict()["proj.weight"]
        operation = object.__new__(DtensorShardOperation)
        DtensorShardOperation.__init__(operation, detached)
        assert id(detached) != id(model.proj.weight)
        assert operation._axolotl_native_nvfp4_component

    assert "state_dict" not in model.__dict__


def test_model_loader_ordinary_load_allows_null_tensor_parallel_size(monkeypatch):
    import builtins
    from types import SimpleNamespace

    from axolotl.loaders.model import ModelLoader

    calls = []

    class Loader:
        @staticmethod
        def from_pretrained(base_model, **kwargs):
            calls.append((base_model, kwargs))
            return "model"

    original_import = builtins.__import__

    def reject_torchao_tp(name, *args, **kwargs):
        if name == "axolotl.monkeypatch.torchao_tp":
            raise AssertionError("ordinary loading must not import the native TP hook")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_torchao_tp)
    model_loader = object.__new__(ModelLoader)
    model_loader.auto_model_loader = Loader
    model_loader.base_model = "ordinary"
    model_loader.model_config = SimpleNamespace(quantization_config=None)
    model_loader.model_kwargs = {}
    model_loader.cfg = SimpleNamespace(
        tensor_parallel_size=None,
        trust_remote_code=None,
    )

    assert model_loader._load_model_from_pretrained() == "model"
    assert calls == [
        ("ordinary", {"config": model_loader.model_config, "trust_remote_code": False})
    ]


def test_model_loader_uses_native_tp_hook_for_native_nvfp4(monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace

    import axolotl.monkeypatch.torchao_tp as torchao_tp
    from axolotl.loaders.model import ModelLoader

    calls = []

    class Loader:
        @staticmethod
        def from_pretrained(base_model, **kwargs):
            calls.append(("load", base_model, kwargs))
            return "model"

    @contextmanager
    def native_hook(mesh):
        calls.append(("enter", mesh))
        yield
        calls.append(("exit", mesh))

    monkeypatch.setattr(torchao_tp, "native_nvfp4_tp_checkpoint_loading", native_hook)
    model_loader = object.__new__(ModelLoader)
    model_loader.auto_model_loader = Loader
    model_loader.base_model = "native"
    model_loader.model_config = SimpleNamespace(
        quantization_config={"_type": "NVFP4WeightOnlyConfig"}
    )
    model_loader.model_kwargs = {}
    model_loader.device_mesh = object()
    model_loader.cfg = SimpleNamespace(
        tensor_parallel_size=2,
        trust_remote_code=None,
    )

    assert model_loader._load_model_from_pretrained() == "model"
    assert calls == [
        ("enter", model_loader.device_mesh),
        (
            "load",
            "native",
            {"config": model_loader.model_config, "trust_remote_code": False},
        ),
        ("exit", model_loader.device_mesh),
    ]
