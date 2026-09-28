"""Validation of tensor_parallel_size combinations."""

from types import SimpleNamespace

import pytest
import torch

from axolotl.utils.config import validate_config
from axolotl.utils.dict import DictDefault


def _loader(**cfg):
    from axolotl.loaders.model import ModelLoader

    loader = object.__new__(ModelLoader)
    loader.model = object()
    loader.reference_model = False
    loader.cfg = DictDefault(
        dict(adapter="lora", lora_model_dir=None, rl=None, merge_lora=False) | cfg
    )
    return loader


class Shard:
    def __init__(self, dim):
        self.dim = dim


class _LoRA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base_layer = torch.nn.Module()
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


class TestTensorParallelValidation:
    """TP composes with FSDP/CP but not with adapters or expert parallelism."""

    def test_rejects_adapter_on_bf16_base(self):
        from axolotl.loaders.model import check_tensor_parallel_adapter_support

        with pytest.raises(ValueError, match="native NVFP4"):
            check_tensor_parallel_adapter_support(False)

    def test_allows_adapter_on_native_nvfp4_base(self):
        from axolotl.loaders.model import check_tensor_parallel_adapter_support

        check_tensor_parallel_adapter_support(True)

    def test_rejects_when_no_lora_target_was_prepared(self):
        from axolotl.loaders.model import check_tensor_parallel_adapter_support
        from axolotl.monkeypatch.torchao_tp_lora import NativeNvfp4TpLoraResult

        with pytest.raises(ValueError, match="native NVFP4"):
            check_tensor_parallel_adapter_support(NativeNvfp4TpLoraResult())

    def test_rejects_tp_sharded_targets_that_are_not_nvfp4(self):
        """A partially quantized base: the bf16 targets still in the tp_plan would get plain
        PEFT and train shard-locally, even though the NVFP4 ones were prepared."""
        from axolotl.loaders.model import check_tensor_parallel_adapter_support
        from axolotl.monkeypatch.torchao_tp_lora import NativeNvfp4TpLoraResult

        result = NativeNvfp4TpLoraResult(
            prepared=("layers.0.q_proj",), skipped=("layers.0.o_proj",)
        )
        assert result
        with pytest.raises(ValueError, match="layers.0.o_proj"):
            check_tensor_parallel_adapter_support(result)

    def test_prepare_reports_tp_sharded_non_nvfp4_targets(self, monkeypatch):
        from axolotl.loaders.model import check_tensor_parallel_adapter_support
        from axolotl.monkeypatch import torchao_tp_lora

        model = torch.nn.Module()
        model.q_proj = _LoRA()
        model.o_proj = _LoRA()
        model.o_proj.base_layer.weight.dtype = torch.bfloat16
        nvfp4 = model.q_proj.base_layer.weight

        monkeypatch.setattr(
            torchao_tp_lora, "_is_native_nvfp4_dtensor", lambda w: w is nvfp4
        )
        monkeypatch.setattr(torchao_tp_lora, "_is_tp_dtensor", lambda w: True)
        monkeypatch.setattr(torchao_tp_lora, "_lora_tp_plan", lambda *_: "colwise")
        monkeypatch.setattr(torchao_tp_lora.dist, "get_world_size", lambda group: 1)

        result = torchao_tp_lora.prepare_native_nvfp4_tp_lora(model)
        assert result.prepared == ("q_proj",)
        assert result.skipped == ("o_proj",)
        assert model.q_proj._axolotl_native_nvfp4_tp_lora_prepared
        assert not hasattr(model.o_proj, "_axolotl_native_nvfp4_tp_lora_prepared")
        with pytest.raises(ValueError, match="o_proj"):
            check_tensor_parallel_adapter_support(result)

    def test_prepare_ignores_unsharded_non_nvfp4_targets(self, monkeypatch):
        from axolotl.monkeypatch import torchao_tp_lora

        model = torch.nn.Module()
        model.q_proj = _LoRA()
        monkeypatch.setattr(
            torchao_tp_lora, "_is_native_nvfp4_dtensor", lambda w: False
        )

        result = torchao_tp_lora.prepare_native_nvfp4_tp_lora(model)
        assert not result
        assert result.skipped == ()

    def test_tp_dtensor_predicate_uses_the_tp_mesh_dim(self):
        from torch.distributed.tensor import DTensor

        from axolotl.monkeypatch.torchao_tp_lora import _is_tp_dtensor

        class _FakeDTensor:
            __class__ = DTensor  # isinstance() honours __class__

            def __init__(self, **mesh):
                self.device_mesh = SimpleNamespace(**mesh)

        assert _is_tp_dtensor(_FakeDTensor(mesh_dim_names=("dp_shard", "tp")))
        assert _is_tp_dtensor(_FakeDTensor(mesh_dim_names=None, ndim=1))
        assert not _is_tp_dtensor(_FakeDTensor(mesh_dim_names=("dp_shard",)))
        assert not _is_tp_dtensor(torch.zeros(2))
        assert not _is_tp_dtensor(None)

    @pytest.mark.parametrize("rl", ["dpo", "ipo", "kto"])
    def test_rejects_adapter_on_config_only_rl_path(self, rl, monkeypatch):
        """TRL wraps the TP-sharded model itself, after the loader could prepare it."""
        from axolotl.loaders import model as model_module

        monkeypatch.setattr(
            model_module,
            "load_adapter",
            lambda *a, **k: pytest.fail("load_adapter must not run"),
        )
        loader = _loader(rl=rl, tensor_parallel_size=2)
        with pytest.raises(ValueError, match=f"rl: {rl}"):
            loader._build_adapters()

    def test_rl_guard_allows_tp_without_adapter_and_adapter_without_tp(self):
        from axolotl.loaders.model import check_tensor_parallel_rl_adapter_support

        check_tensor_parallel_rl_adapter_support(
            DictDefault(adapter=None, rl="dpo", tensor_parallel_size=2)
        )
        check_tensor_parallel_rl_adapter_support(
            DictDefault(adapter="lora", rl="dpo", tensor_parallel_size=1)
        )
        check_tensor_parallel_rl_adapter_support(
            DictDefault(adapter="lora", rl="dpo", tensor_parallel_size=None)
        )

    def test_rl_merge_lora_path_still_runs_the_nvfp4_guard(self, monkeypatch):
        from axolotl.loaders import model as model_module
        from axolotl.monkeypatch import torchao_tp_lora

        adapted, config = object(), object()
        monkeypatch.setattr(
            model_module, "load_adapter", lambda *a, **k: (adapted, config)
        )
        monkeypatch.setattr(
            torchao_tp_lora, "prepare_native_nvfp4_tp_lora", lambda *a, **k: False
        )
        loader = _loader(rl="dpo", merge_lora=True, tensor_parallel_size=2)
        with pytest.raises(ValueError, match="native NVFP4"):
            loader._build_adapters()


class TestExpertParallelLoraReinit:
    """The per-module seeded re-draw exists for expert parallelism; other runs keep PEFT's init."""

    @staticmethod
    def _spy(monkeypatch):
        from axolotl.loaders import adapter as adapter_module

        calls = []
        monkeypatch.setattr(
            adapter_module,
            "reinit_lora_from_seed",
            lambda model, seed: calls.append((model, seed)) or 1,
        )
        return calls

    def test_skipped_without_expert_parallelism(self, monkeypatch):
        calls = self._spy(monkeypatch)
        for cfg in ({}, {"expert_parallel_size": 1}, {"expert_parallel_size": None}):
            loader = _loader(seed=42, **cfg)
            loader._reinit_expert_parallel_lora(SimpleNamespace(init_lora_weights=True))
        assert calls == []

    def test_runs_under_expert_parallelism_with_the_config_seed(self, monkeypatch):
        calls = self._spy(monkeypatch)
        loader = _loader(seed=42, expert_parallel_size=2)
        loader._reinit_expert_parallel_lora(SimpleNamespace(init_lora_weights=True))
        assert calls == [(loader.model, 42)]

    def test_unseeded_run_uses_the_process_seed_not_zero(self, monkeypatch):
        calls = self._spy(monkeypatch)
        loader = _loader(seed=None, expert_parallel_size=2)
        loader._reinit_expert_parallel_lora(SimpleNamespace(init_lora_weights=True))
        assert calls == [(loader.model, torch.initial_seed())]
        assert calls[0][1] != 0

    def test_skipped_for_loaded_or_value_dependent_adapters(self, monkeypatch):
        calls = self._spy(monkeypatch)
        loader = _loader(seed=1, expert_parallel_size=2, lora_model_dir="/adapter")
        loader._reinit_expert_parallel_lora(SimpleNamespace(init_lora_weights=True))
        loader = _loader(seed=1, expert_parallel_size=2)
        loader._reinit_expert_parallel_lora(
            SimpleNamespace(init_lora_weights="gaussian")
        )
        loader._reinit_expert_parallel_lora(None)
        assert calls == []

    def test_rejects_expert_parallel(self, min_base_cfg):
        from axolotl.integrations.base import PluginManager

        plugin = "axolotl.integrations.expert_parallel.ExpertParallelPlugin"
        manager = PluginManager.get_instance()
        manager.register(plugin)
        try:
            cfg = (
                DictDefault(
                    tensor_parallel_size=2, expert_parallel_size=2, plugins=[plugin]
                )
                | min_base_cfg
            )
            with pytest.raises(ValueError, match="expert_parallel_size"):
                validate_config(cfg)
        finally:
            manager.plugins.pop(plugin, None)

    def test_allows_full_parameter(self, min_base_cfg):
        cfg = DictDefault(tensor_parallel_size=2) | min_base_cfg
        validate_config(cfg)

    def test_disables_cpu_ram_efficient_loading(self, min_base_cfg):
        cfg = (
            DictDefault(
                tensor_parallel_size=2,
                fsdp_version=2,
                fsdp_config={"cpu_ram_efficient_loading": True},
            )
            | min_base_cfg
        )
        out = validate_config(cfg)
        assert out.fsdp_config.cpu_ram_efficient_loading is False
