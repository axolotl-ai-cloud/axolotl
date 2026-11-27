"""CPU tests for the ``optmoe`` ``nvfp4_lora`` precision seam: config and model
qualification, the PEFT adapter layout mapping, the activation-scale lookup and the
requant cache's optimizer hook."""

# ruff: noqa: F811  (fixtures imported from test_optmoe_backend are reused by name)

import json
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from axolotl.integrations.expert_parallel import (
    ExpertParallelPlugin,
    experts_fn,
    nvfp4_lora,
    plugin as plugin_mod,
)

from .test_optmoe_backend import (  # noqa: F401
    _cfg,
    _FakeRuntime,
    _qwen3_model,
    _restore_ep_state,
    cuda_capability,
    optmoe_package,
)

HIDDEN, INTER, EXPERTS = 512, 1024, 4


def _lora_cfg(**kw):
    defaults = dict(
        adapter="lora",
        use_sonicmoe=True,
        lora_dropout=0,
        lora_target_parameters=["experts.gate_up_proj", "experts.down_proj"],
        nvfp4_merge_aware=False,
        base_model="nvidia/Qwen3-30B-A3B-NVFP4",
    )
    defaults.update(kw)
    return _cfg(**defaults)


@pytest.fixture
def nvfp4_base(monkeypatch):
    monkeypatch.setattr(nvfp4_lora, "base_is_nvfp4_modelopt", lambda cfg: True)


class _FakeNVFP4Tensor(torch.Tensor):
    pass


_FakeNVFP4Tensor.__name__ = "NVFP4Tensor"


def _nvfp4_param(num_experts, rows, cols, *, swizzled=False, scalar_pts=False):
    param = torch.nn.Parameter(
        torch.Tensor._make_subclass(
            _FakeNVFP4Tensor, torch.empty(num_experts, rows, cols)
        ),
        requires_grad=False,
    )
    param.qdata = torch.zeros(num_experts, rows, cols // 2, dtype=torch.uint8)
    param.scale = torch.zeros(num_experts, rows, cols // 16, dtype=torch.float8_e4m3fn)
    param.per_tensor_scale = torch.ones(1 if scalar_pts else num_experts, 1, 1)
    param.is_swizzled_scales = swizzled
    param.block_size = 16
    return param


def _nvfp4_model(**param_kw):
    model = _qwen3_model(hidden=HIDDEN, intermediate=INTER, num_experts=EXPERTS)
    experts = model.mlp.experts
    experts.gate_up_proj = _nvfp4_param(EXPERTS, 2 * INTER, HIDDEN, **param_kw)
    experts.down_proj = _nvfp4_param(EXPERTS, HIDDEN, INTER, **param_kw)
    return model


class _FakeParamWrapper(torch.nn.Module):
    def __init__(self, base_layer, parameter_name, scaling):
        super().__init__()
        self.base_layer = base_layer
        self.parameter_name = parameter_name
        self.lora_A = torch.nn.ParameterDict()
        self.scaling = {"default": scaling}
        self.active_adapters = ["default"]


def _adapt(model, gate_up=2.0, down=2.0):
    """Nest PEFT-style wrappers over ``model.mlp.experts`` and return the bare module."""
    experts = model.mlp.experts
    wrapped = experts
    for name, scaling in (("gate_up_proj", gate_up), ("down_proj", down)):
        if scaling is not None:
            wrapped = _FakeParamWrapper(wrapped, name, scaling)
    model.mlp.experts = wrapped
    return experts


@pytest.fixture
def nvfp4_lora_package(optmoe_package, monkeypatch):
    """``optmoe.nvfp4_lora`` stand-ins: a requant cache with the real invalidation
    contract and an experts class recording its constructor arguments."""

    class RequantCache:
        def __init__(self, mode=None):
            self.mode = mode or "layer"
            self.version = 0
            self.entries = {}

        def invalidate(self):
            self.version += 1
            self.entries.clear()

        def stats(self):
            return dict(mode=self.mode, version=self.version)

    class Nvfp4LoraExperts:
        instances = []

        def __init__(
            self, q1, s1, pts1, q2, s2, pts2, *, scaling, input_scale, backend
        ):
            self.args = (q1, s1, pts1, q2, s2, pts2)
            self.scaling = float(scaling)
            self.input_scale = input_scale
            self.backend = backend
            self.experts, n1, k1 = q1.shape
            self.hidden, self.inter = 2 * k1, n1 // 2
            Nvfp4LoraExperts.instances.append(self)

        def transposed_base_bytes(self):
            return 0

    module = types.ModuleType("optmoe.nvfp4_lora")
    module.RequantCache = RequantCache
    module.Nvfp4LoraExperts = Nvfp4LoraExperts
    monkeypatch.setitem(sys.modules, "optmoe.nvfp4_lora", module)
    optmoe_package.nvfp4_lora = module
    yield module
    nvfp4_lora.set_state(None)


class TestConfigReason:
    def test_qualifying_config_allows_fused(self, optmoe_package, nvfp4_base):
        for precision in ("auto", "nvfp4_lora"):
            cfg = _lora_cfg(expert_parallel_optmoe_precision=precision)
            assert ExpertParallelPlugin._optmoe_fused_config_reason(cfg) is None
            assert ExpertParallelPlugin._optmoe_fused_possible(cfg)

    @pytest.mark.parametrize(
        "extra,match",
        [
            ({"adapter": "qlora"}, "single lora adapter"),
            ({"use_sonicmoe": False}, "use_sonicmoe: true"),
            ({"lora_dropout": 0.05}, "lora_dropout: 0"),
            ({"lora_target_parameters": ["experts.down_proj"]}, "gate_up_proj"),
            ({"lora_target_parameters": None}, "lora_target_parameters"),
            ({"nvfp4_merge_aware": True}, "nvfp4_merge_aware is on"),
            ({"nvfp4_merge_aware": None}, None),
        ],
    )
    def test_disqualifying_options(
        self,
        optmoe_package,
        nvfp4_base,
        extra,
        match,
    ):
        cfg = _lora_cfg(expert_parallel_optmoe_precision="nvfp4_lora", **extra)
        reason = ExpertParallelPlugin._optmoe_fused_config_reason(cfg)
        if match is None:
            assert reason is None
        else:
            assert match in reason
            # ``auto`` with an adapter falls back to the dispatch mode with the same cause
            cfg = _lora_cfg(**extra)
            auto_reason = ExpertParallelPlugin._optmoe_fused_config_reason(cfg)
            assert "the fused kernels cannot train it" in auto_reason
            assert match in auto_reason

    def test_non_nvfp4_base_is_refused(self, optmoe_package):
        reason = ExpertParallelPlugin._optmoe_fused_config_reason(
            _lora_cfg(expert_parallel_optmoe_precision="nvfp4_lora")
        )
        assert "modelopt NVFP4 base" in reason
        # ``auto`` trains the adapter on the merged BF16 experts instead
        assert ExpertParallelPlugin._optmoe_fused_config_reason(_lora_cfg()) is None

    def test_other_precisions_keep_the_adapter_rule(
        self,
        optmoe_package,
        nvfp4_base,
    ):
        reason = ExpertParallelPlugin._optmoe_fused_config_reason(
            _lora_cfg(expert_parallel_optmoe_precision="mxfp8")
        )
        assert "packed NVFP4 base trains LoRA only as nvfp4_lora" in reason

    def test_precision_without_adapter_is_refused(self, optmoe_package):
        reason = ExpertParallelPlugin._optmoe_fused_config_reason(
            _cfg(expert_parallel_optmoe_precision="nvfp4_lora")
        )
        assert "needs adapter: lora" in reason

    def test_forced_fused_falls_back_or_raises(self, optmoe_package):
        cfg = _lora_cfg(
            expert_parallel_optmoe_mode="fused",
            expert_parallel_optmoe_fused_lora=False,
        )
        assert ExpertParallelPlugin._resolve_backend(cfg) == "torch"
        cfg = _lora_cfg(
            expert_parallel_optmoe_mode="fused",
            expert_parallel_optmoe_precision="nvfp4_lora",
            expert_parallel_fallback_on_unsupported=False,
        )
        with pytest.raises(ValueError, match="modelopt NVFP4 base"):
            ExpertParallelPlugin._resolve_backend(cfg)


class TestModelResolution:
    def test_nvfp4_experts_resolve_the_precision(
        self,
        optmoe_package,
        nvfp4_base,
        cuda_capability,
    ):
        cuda_capability((10, 0))
        cfg = _lora_cfg()
        plugin = ExpertParallelPlugin()
        plugin.post_model_build(cfg, _nvfp4_model())
        assert cfg.expert_parallel_backend == "optmoe"
        assert cfg.expert_parallel_optmoe_mode == "fused"
        assert cfg.expert_parallel_optmoe_precision == "nvfp4_lora"
        assert experts_fn.get_optmoe_mode() == "fused"
        assert plugin._optmoe_geometry == plugin_mod.OptMoEGeometry(
            EXPERTS, HIDDEN, INTER, 2, None, "nvfp4_lora"
        )

    def test_scalar_per_tensor_scale_is_accepted(
        self,
        optmoe_package,
        nvfp4_base,
        cuda_capability,
    ):
        cuda_capability((10, 0))
        geometry, reason = ExpertParallelPlugin._optmoe_model_reason(
            _lora_cfg(), _nvfp4_model(scalar_pts=True)
        )
        assert reason is None and geometry.precision == "nvfp4_lora"

    @pytest.mark.parametrize(
        "capability,model_kw,plain,match",
        [
            ((9, 0), {}, False, "requires SM100"),
            ((10, 0), {"swizzled": True}, False, "swizzled"),
            ((10, 0), {}, True, "not a packed NVFP4Tensor"),
        ],
    )
    def test_unsupported_experts_take_dispatch_or_fall_back(
        self,
        optmoe_package,
        nvfp4_base,
        cuda_capability,
        caplog,
        capability,
        model_kw,
        plain,
        match,
    ):
        cuda_capability(capability)
        build = _qwen3_model if plain else _nvfp4_model
        kwargs = dict(hidden=HIDDEN, intermediate=INTER) if plain else model_kw
        _, reason = ExpertParallelPlugin._optmoe_model_reason(
            _lora_cfg(), build(**kwargs)
        )
        assert match in reason

        cfg = _lora_cfg()  # auto: the peer dispatch mode carries the adapter
        ExpertParallelPlugin().post_model_build(cfg, build(**kwargs))
        assert cfg.expert_parallel_backend == "optmoe"
        assert cfg.expert_parallel_optmoe_mode == "dispatch"
        assert cfg.expert_parallel_optmoe_precision == "auto"

        cfg = _lora_cfg(
            expert_parallel_optmoe_mode="fused",
            expert_parallel_optmoe_precision="nvfp4_lora",
        )
        with caplog.at_level("WARNING"):
            ExpertParallelPlugin().post_model_build(cfg, build(**kwargs))
        assert match in caplog.text
        assert cfg.expert_parallel_backend == "torch"

    def test_full_fine_tune_geometry_has_no_precision(
        self,
        optmoe_package,
        cuda_capability,
    ):
        cuda_capability((10, 0))
        geometry, reason = ExpertParallelPlugin._optmoe_model_reason(
            _cfg(), _qwen3_model()
        )
        assert reason is None and geometry.precision is None


class TestAdapterDiscovery:
    def test_nested_wrappers_yield_the_experts_module(self):
        model = _nvfp4_model()
        experts = _adapt(model)
        assert list(nvfp4_lora.expert_adapter_scalings(model)) == [(experts, 2.0)]

    @pytest.mark.parametrize(
        "scalings", [(2.0, 1.0), (2.0, None), (None, 2.0), (None, None)]
    )
    def test_unqualified_adapters_yield_nothing(self, scalings):
        model = _nvfp4_model()
        _adapt(model, *scalings)
        assert list(nvfp4_lora.expert_adapter_scalings(model)) == []

    def test_inactive_adapter_yields_nothing(self):
        model = _nvfp4_model()
        _adapt(model)
        model.mlp.experts.active_adapters = []
        assert list(nvfp4_lora.expert_adapter_scalings(model)) == []


class TestAdapterMapping:
    def test_peft_layouts_map_to_optmoe_views(self):
        e, r, h, i = 3, 2, 8, 6
        a_per_expert = torch.randn(e, r, h)
        b_per_expert = torch.randn(e, 2 * i, r)
        lora_a = a_per_expert.reshape(e * r, h)  # expert-major rows
        lora_b = b_per_expert.permute(1, 2, 0).reshape(2 * i, r * e)  # rank-major cols
        a, b = nvfp4_lora.map_adapter(lora_a, lora_b, e, r, h)
        assert torch.equal(a, a_per_expert)
        assert torch.equal(b, b_per_expert)
        assert b.is_contiguous()

    def test_global_adapter_is_sliced_to_local_experts(self):
        e_global, e_local, r, h, out = 4, 2, 2, 8, 6
        experts = SimpleNamespace(
            num_experts=e_local, num_experts_global=e_global, local_expert_offset=2
        )
        lora_a = torch.arange(e_global * r * h, dtype=torch.float32).reshape(
            e_global * r, h
        )
        lora_b = torch.arange(out * r * e_global, dtype=torch.float32).reshape(
            out, r * e_global
        )
        wrapper = SimpleNamespace(r={"default": r}, active_adapters=["default"])
        a, b, n, rank = nvfp4_lora.local_expert_lora(lora_a, lora_b, experts, wrapper)
        assert (n, rank) == (e_local, r)
        assert torch.equal(a, lora_a[4:8])
        assert torch.equal(
            b, lora_b.reshape(out, r, e_global)[:, :, 2:].reshape(out, -1)
        )

        wrapper._ep_lora_sharded = True
        local_a, local_b = (
            lora_a[4:8],
            lora_b.reshape(out, r, e_global)[:, :, 2:].reshape(out, -1),
        )
        same = nvfp4_lora.local_expert_lora(local_a, local_b, experts, wrapper)
        assert same[0] is local_a and same[1] is local_b

    def test_forward_hands_mapped_factors_to_the_runtime(
        self, nvfp4_lora_package, monkeypatch
    ):
        e, r, h, i, tokens = 2, 2, 256, 256, 5
        experts = SimpleNamespace(
            num_experts=e, num_experts_global=e, local_expert_offset=0
        )
        a1 = torch.randn(e * r, h, requires_grad=True)
        b1 = torch.randn(2 * i, r * e, requires_grad=True)
        a2 = torch.randn(e * r, i, requires_grad=True)
        b2 = torch.randn(h, r * e, requires_grad=True)
        experts._sonicmoe_lora = {
            "gate_up_proj": (a1, b1, 2.0),
            "down_proj": (a2, b2, 2.0),
        }
        experts._sonicmoe_lora_wrappers = {}
        layer = SimpleNamespace(hidden=h, inter=i, scaling=2.0)
        state = SimpleNamespace(cache=object(), layer=lambda experts_, scaling: layer)
        monkeypatch.setattr(nvfp4_lora, "get_state", lambda: state)
        calls = []

        def forward_lora(x, idx, w, *factors):
            calls.append((factors[:-1], factors[-1]))
            return x * 2

        runtime = SimpleNamespace(precision="nvfp4_lora", forward_lora=forward_lora)
        x = torch.randn(tokens, h)
        out = nvfp4_lora.forward(
            runtime,
            experts,
            x,
            torch.zeros(tokens, 2, dtype=torch.int64),
            torch.ones(tokens, 2),
        )
        assert torch.equal(out, x * 2)
        ((ma1, mb1, ma2, mb2, got_layer), cache) = calls[0]
        assert got_layer is layer and cache is state.cache
        assert ma1.shape == (e, r, h) and mb1.shape == (e, 2 * i, r)
        assert ma2.shape == (e, r, i) and mb2.shape == (e, h, r)
        # the views stay on the autograd graph of the PEFT parameters
        (mb1.sum() + ma2.sum()).backward()
        assert b1.grad is not None and a2.grad is not None

    def test_forward_requires_the_fastpath_factors(self, nvfp4_lora_package):
        experts = SimpleNamespace(num_experts=2, _sonicmoe_lora=None)
        with pytest.raises(RuntimeError, match="received no LoRA factors"):
            nvfp4_lora.forward(
                SimpleNamespace(precision="nvfp4_lora"),
                experts,
                torch.zeros(1, 4),
                torch.zeros(1, 2, dtype=torch.int64),
                torch.ones(1, 2),
            )

    def test_fused_backend_routes_nvfp4_lora(self, monkeypatch):
        routed = []
        monkeypatch.setattr(
            nvfp4_lora, "forward", lambda *args: routed.append(args) or args[2]
        )
        runtime = SimpleNamespace(precision="nvfp4_lora")
        experts_fn.set_optmoe_runtime(runtime)
        x = torch.zeros(3, 4)
        out = experts_fn._OptMoEFusedBackend.forward(
            object(),
            x,
            torch.zeros(3, 2, dtype=torch.int64),
            torch.ones(3, 2),
            num_experts_global=4,
            num_local_experts=2,
        )
        assert out is x and routed and routed[0][0] is runtime


def _write_checkpoint(tmp_path, scales):
    from safetensors.torch import save_file

    tensors = {
        f"model.layers.{layer}.mlp.experts.{expert}.{proj}.input_scale": torch.tensor(
            value
        )
        for (layer, expert, proj), value in scales.items()
    }
    tensors["model.layers.0.mlp.experts.0.gate_proj.weight"] = torch.zeros(
        2, 2, dtype=torch.uint8
    )
    save_file(tensors, str(tmp_path / "model-00001-of-00001.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": dict.fromkeys(tensors, "model-00001-of-00001.safetensors")}
        )
    )
    return str(tmp_path)


class TestActivationScales:
    SCALES = {
        (0, e, proj): base + 0.01 * e
        for e in range(4)
        for proj, base in (("gate_proj", 0.2), ("up_proj", 0.3), ("down_proj", 0.5))
    }

    def test_reads_modelopt_input_scales(self, tmp_path):
        repo = _write_checkpoint(tmp_path, self.SCALES)
        scales = nvfp4_lora.read_input_scales(repo)
        assert len(scales) == len(self.SCALES)
        assert scales[(0, 3, "up_proj")] == pytest.approx(0.33)

    def test_scales_are_read_at_the_weights_revision(self, monkeypatch):
        import huggingface_hub

        seen = []

        def cached(repo_id, filename, revision=None):
            seen.append(("cache", filename, revision))
            return None

        def download(repo_id, filename, revision=None):
            seen.append(("hub", filename, revision))
            raise OSError("offline")

        monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", cached)
        monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
        assert nvfp4_lora.read_input_scales("org/model", "pinned") == {}
        assert seen and {revision for _, _, revision in seen} == {"pinned"}

    def test_missing_index_defaults_to_unit_scales(self, tmp_path, caplog):
        with caplog.at_level("WARNING"):
            assert nvfp4_lora.read_input_scales(str(tmp_path)) == {}
        assert "default to 1.0" in caplog.text

    def test_layer_builds_with_local_expert_scales(
        self, tmp_path, nvfp4_lora_package, monkeypatch
    ):
        repo = _write_checkpoint(tmp_path, self.SCALES)
        state = nvfp4_lora.Nvfp4LoraState(repo_id=repo)
        model = _nvfp4_model()
        experts = model.mlp.experts
        experts.num_local_experts = 2
        experts.local_expert_offset = 2
        assert (
            nvfp4_lora.tag_layers(
                torch.nn.ModuleDict({"layers": torch.nn.ModuleList([model])})
            )
            == 1
        )
        assert experts._optmoe_layer_index == 0
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: 0)
        monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
        gate_up = experts.gate_up_proj
        monkeypatch.setattr(
            nvfp4_lora.Nvfp4LoraState,
            "_device_ok",
            staticmethod(lambda device: True),
            raising=False,
        )
        layer = state.layer(experts, scaling=2.0)
        assert state.layer(experts, scaling=2.0) is layer
        q1, s1, pts1, q2, s2, pts2 = layer.args
        assert q1 is gate_up.qdata and s1 is gate_up.scale
        assert q2 is experts.down_proj.qdata
        assert pts1.shape == (EXPERTS,) and pts2.shape == (EXPERTS,)
        pts_x, pts_h = layer.input_scale
        torch.testing.assert_close(pts_x, torch.tensor([0.32, 0.33]))
        torch.testing.assert_close(pts_h, torch.tensor([0.52, 0.53]))
        state.close()

    def test_layer_without_scales_uses_unit(self, nvfp4_lora_package, monkeypatch):
        state = nvfp4_lora.Nvfp4LoraState(repo_id=None)
        experts = _nvfp4_model().mlp.experts
        experts.num_local_experts = EXPERTS
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: 0)
        monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
        monkeypatch.setattr(
            nvfp4_lora.Nvfp4LoraState,
            "_device_ok",
            staticmethod(lambda device: True),
            raising=False,
        )
        assert state.layer(experts, scaling=1.0).input_scale == (1.0, 1.0)
        state.close()


class TestPrepareLayers:
    @pytest.fixture
    def on_gpu(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: 0)
        monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
        monkeypatch.setattr(
            nvfp4_lora.Nvfp4LoraState, "_device_ok", staticmethod(lambda device: True)
        )

    def test_layers_are_built_and_registered(self, nvfp4_lora_package, on_gpu):
        registered = []
        nvfp4_lora_package.register_layer = lambda entry, cache: registered.append(
            (entry, cache)
        )
        state = nvfp4_lora.Nvfp4LoraState(repo_id=None)
        model = _nvfp4_model()
        experts = _adapt(model)
        experts.num_local_experts = EXPERTS
        assert state.prepare_layers(model) == 1
        layer = state.layers[id(experts)]
        assert registered == [(layer, state.cache)]
        assert state.layer(experts, scaling=2.0) is layer
        assert len(registered) == 1
        state.close()

    def test_package_without_register_layer_still_builds(
        self, nvfp4_lora_package, on_gpu
    ):
        state = nvfp4_lora.Nvfp4LoraState(repo_id=None)
        model = _nvfp4_model()
        experts = _adapt(model)
        experts.num_local_experts = EXPERTS
        assert state.prepare_layers(model) == 1
        state.close()

    def test_experts_off_the_gpu_are_left_for_the_first_forward(
        self, nvfp4_lora_package
    ):
        state = nvfp4_lora.Nvfp4LoraState(repo_id=None)
        model = _nvfp4_model()
        _adapt(model)
        assert state.prepare_layers(model) == 0 and not state.layers
        state.close()

    def test_bf16_experts_are_skipped(self, nvfp4_lora_package, on_gpu):
        state = nvfp4_lora.Nvfp4LoraState(repo_id=None)
        model = _qwen3_model(hidden=HIDDEN, intermediate=INTER, num_experts=EXPERTS)
        _adapt(model)
        assert state.prepare_layers(model) == 0
        state.close()


class TestRuntimeLifecycle:
    def test_state_follows_the_runtime(
        self,
        monkeypatch,
        nvfp4_lora_package,
        nvfp4_base,
    ):
        monkeypatch.setattr(
            ExpertParallelPlugin,
            "_resolve_ep_group",
            staticmethod(lambda cfg: object()),
        )
        monkeypatch.setattr(
            nvfp4_lora, "read_input_scales", lambda repo, revision=None: {}
        )
        cfg = _lora_cfg(
            expert_parallel_optmoe_precision="nvfp4_lora",
            activation_checkpointing={"enabled": False},
        )
        plugin = ExpertParallelPlugin()
        plugin._optmoe_geometry = plugin_mod.OptMoEGeometry(
            EXPERTS, HIDDEN, INTER, 2, None, "nvfp4_lora"
        )
        model = _nvfp4_model()
        plugin.post_model_load(cfg, model)
        (runtime,) = _FakeRuntime.instances
        assert runtime.kwargs["precision"] == "nvfp4_lora"
        state = nvfp4_lora.get_state()
        assert state is not None and nvfp4_lora.active()

        # every optimizer step invalidates the merged weights, including under
        # gradient accumulation (no step, no invalidation)
        param = torch.nn.Parameter(torch.zeros(2))
        optimizer = torch.optim.SGD([param], lr=0.1)
        param.grad = torch.ones(2)
        assert state.cache.version == 0
        optimizer.step()
        assert state.cache.version == 1
        optimizer.zero_grad()
        param.grad = torch.ones(2)
        optimizer.step()
        assert state.cache.version == 2

        plugin.post_train_unload(cfg)
        assert nvfp4_lora.get_state() is None and not nvfp4_lora.active()
        version = state.cache.version
        param.grad = torch.ones(2)
        optimizer.step()
        assert state.cache.version == version  # hook removed

    def test_full_fine_tune_builds_no_state(self, monkeypatch, nvfp4_lora_package):
        monkeypatch.setattr(
            ExpertParallelPlugin,
            "_resolve_ep_group",
            staticmethod(lambda cfg: object()),
        )
        cfg = _cfg(activation_checkpointing={"enabled": False})
        plugin = ExpertParallelPlugin()
        plugin._optmoe_geometry = plugin_mod.OptMoEGeometry(
            EXPERTS, HIDDEN, INTER, 2, None
        )
        plugin.post_model_load(cfg, torch.nn.Linear(2, 2))
        assert nvfp4_lora.get_state() is None
        plugin.post_train_unload(cfg)


class TestEfficientLoadingMetaMove:
    """cpu_ram_efficient_loading moves the model to meta before FSDP wrapping and refills it
    from rank 0; the frozen packed experts are this rank's EP slice outside FSDP, so they
    must skip that round trip."""

    def test_frozen_experts_stay_real(self, monkeypatch):
        monkeypatch.setattr(nvfp4_lora, "active", lambda: True)
        model = _nvfp4_model()
        experts = model.mlp.experts
        gate_up, down = experts.gate_up_proj, experts.down_proj
        kept = nvfp4_lora.detach_frozen_experts(model)
        assert {attr for _m, attr, _p in kept} == {"gate_up_proj", "down_proj"}
        model = model.to(torch.device("meta"))
        nvfp4_lora.reattach_frozen_experts(kept)
        assert experts.gate_up_proj is gate_up and experts.down_proj is down
        assert not gate_up.is_meta and not down.is_meta
        assert all(
            p.is_meta
            for n, p in model.named_parameters()
            if not n.endswith(("experts.gate_up_proj", "experts.down_proj"))
        )

    def test_nothing_detached_without_nvfp4_lora(self, monkeypatch):
        monkeypatch.setattr(nvfp4_lora, "active", lambda: False)
        assert nvfp4_lora.detach_frozen_experts(_nvfp4_model()) == []

    def test_meta_experts_before_wrapping_raise(self, monkeypatch):
        monkeypatch.setattr(nvfp4_lora, "active", lambda: True)
        model = _nvfp4_model()
        experts = model.mlp.experts
        experts.gate_up_proj = torch.nn.Parameter(
            torch.Tensor._make_subclass(
                _FakeNVFP4Tensor, torch.empty(EXPERTS, 2 * INTER, HIDDEN, device="meta")
            ),
            requires_grad=False,
        )
        with pytest.raises(RuntimeError, match="on meta before FSDP wrapping"):
            nvfp4_lora.detach_frozen_experts(model)
