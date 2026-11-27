"""CPU tests for the ``optmoe`` fused LoRA precisions (``bf16_lora`` / ``mxfp8_lora``):
config and model qualification, the dispatch fallback selector, routing of the fused
forward and the merged-weight cache's per-process state."""

# ruff: noqa: F811  (fixtures imported from test_optmoe_backend are reused by name)

import os
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from axolotl.integrations.expert_parallel import (
    ExpertParallelPlugin,
    experts_fn,
    fused_lora,
)

from .test_optmoe_backend import (  # noqa: F401
    _cfg,
    _qwen3_model,
    _restore_ep_state,
    cuda_capability,
    optmoe_package,
)
from .test_optmoe_nvfp4_lora import _adapt, _lora_cfg, _nvfp4_model


def _bf16_lora_cfg(**kw):
    return _lora_cfg(base_model="Qwen/Qwen3-30B-A3B", **kw)


class TestConfigReason:
    @pytest.mark.parametrize("precision", ["auto", "bf16", "mxfp8", "mxfp8_lora"])
    def test_qualifying_config_allows_fused(self, optmoe_package, precision):
        cfg = _bf16_lora_cfg(expert_parallel_optmoe_precision=precision)
        assert ExpertParallelPlugin._optmoe_fused_config_reason(cfg) is None
        assert ExpertParallelPlugin._optmoe_fused_possible(cfg)

    @pytest.mark.parametrize(
        "extra,match",
        [
            ({"adapter": "qlora"}, "single lora adapter"),
            ({"peft_use_dora": True}, "DoRA"),
            ({"lora_dropout": 0.1}, "lora_dropout: 0"),
            ({"use_sonicmoe": False}, "use_sonicmoe: true"),
            ({"lora_target_parameters": ["experts.gate_up_proj"]}, "down_proj"),
            (
                {
                    "lora_target_parameters": [
                        "layers.0.mlp.experts.gate_up_proj",
                        "layers.0.mlp.experts.down_proj",
                    ]
                },
                "every layer",
            ),
        ],
    )
    def test_unsupported_adapters_keep_the_dispatch_mode(
        self, optmoe_package, extra, match
    ):
        reason = ExpertParallelPlugin._optmoe_fused_config_reason(
            _bf16_lora_cfg(expert_parallel_optmoe_precision="mxfp8", **extra)
        )
        assert "the fused kernels cannot train it" in reason and match in reason
        explicit = ExpertParallelPlugin._optmoe_fused_config_reason(
            _bf16_lora_cfg(expert_parallel_optmoe_precision="mxfp8_lora", **extra)
        )
        assert match in explicit

    def test_selector_restores_the_dispatch_fallback(self, optmoe_package):
        cfg = _bf16_lora_cfg(
            expert_parallel_optmoe_precision="mxfp8",
            expert_parallel_optmoe_fused_lora=False,
        )
        reason = ExpertParallelPlugin._optmoe_fused_config_reason(cfg)
        assert "trains full expert weights only" in reason
        assert not ExpertParallelPlugin._optmoe_fused_possible(cfg)

    def test_lora_precision_without_adapter_is_refused(self, optmoe_package):
        reason = ExpertParallelPlugin._optmoe_fused_config_reason(
            _cfg(expert_parallel_optmoe_precision="bf16_lora")
        )
        assert "needs adapter: lora" in reason


class TestModelResolution:
    @pytest.mark.parametrize(
        "capability,precision,resolved",
        [
            ((10, 0), "auto", "mxfp8_lora"),
            ((10, 0), "mxfp8", "mxfp8_lora"),
            ((10, 0), "bf16", "bf16_lora"),
            ((9, 0), "auto", "bf16_lora"),
            ((9, 0), "fp8", "fp8_lora"),
        ],
    )
    def test_bf16_experts_resolve_the_lora_precision(
        self, optmoe_package, cuda_capability, capability, precision, resolved
    ):
        cuda_capability(capability)
        cfg = _bf16_lora_cfg(expert_parallel_optmoe_precision=precision)
        ExpertParallelPlugin().post_model_build(cfg, _qwen3_model())
        assert cfg.expert_parallel_backend == "optmoe"
        assert cfg.expert_parallel_optmoe_mode == "fused"
        assert cfg.expert_parallel_optmoe_precision == resolved

    @pytest.mark.parametrize(
        "capability,precision,build,match",
        [
            ((9, 0), "mxfp8", _qwen3_model, "requires SM100"),
            ((10, 0), "fp8", _qwen3_model, "requires SM90"),
            ((10, 0), "mxfp8", _nvfp4_model, "packed NVFP4"),
            (
                (9, 0),
                "bf16",
                lambda: _qwen3_model(intermediate=384),
                "multiples of 256",
            ),
        ],
    )
    def test_unsupported_take_the_dispatch_mode(
        self, optmoe_package, cuda_capability, capability, precision, build, match
    ):
        cuda_capability(capability)
        cfg = _bf16_lora_cfg(expert_parallel_optmoe_precision=precision)
        _, reason = ExpertParallelPlugin._optmoe_model_reason(cfg, build())
        assert match in reason
        ExpertParallelPlugin().post_model_build(cfg, build())
        assert cfg.expert_parallel_optmoe_mode == "dispatch"


class TestForward:
    def test_fused_backend_routes_the_lora_precisions(self, monkeypatch):
        routed = []
        monkeypatch.setattr(
            fused_lora, "forward", lambda *args: routed.append(args) or args[2]
        )
        runtime = SimpleNamespace(precision="mxfp8_lora")
        experts_fn.set_optmoe_runtime(runtime)
        experts = _qwen3_model(num_experts=4).mlp.experts
        x = torch.zeros(3, experts.gate_up_proj.shape[-1])
        out = experts_fn._OptMoEFusedBackend.forward(
            experts,
            x,
            torch.zeros(3, 2, dtype=torch.int64),
            torch.ones(3, 2),
            num_experts_global=4,
            num_local_experts=4,
        )
        assert out is x and routed[0][0] is runtime
        assert routed[0][5] is experts.gate_up_proj
        assert routed[0][6] is experts.down_proj

    def test_forward_hands_base_and_mapped_factors(self, monkeypatch):
        e, r, h, i, tokens = 2, 2, 256, 128, 5
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
        layer = SimpleNamespace(scaling=2.0)
        state = SimpleNamespace(cache=object(), layer=lambda experts_, scaling: layer)
        monkeypatch.setattr(fused_lora, "get_state", lambda: state)
        calls = []

        def forward_lora(x, idx, w, *factors, w1, w2):
            calls.append((factors, w1, w2))
            return x

        w1, w2 = torch.zeros(e, 2 * i, h), torch.zeros(e, h, i)
        runtime = SimpleNamespace(precision="bf16_lora", forward_lora=forward_lora)
        x = torch.randn(tokens, h)
        fused_lora.forward(
            runtime,
            experts,
            x,
            torch.zeros(tokens, 2, dtype=torch.int64),
            torch.ones(tokens, 2),
            w1,
            w2,
        )
        (ma1, mb1, ma2, mb2, got_layer, cache), got_w1, got_w2 = calls[0]
        assert got_w1 is w1 and got_w2 is w2
        assert got_layer is layer and cache is state.cache
        assert ma1.shape == (e, r, h) and mb1.shape == (e, 2 * i, r)
        assert ma2.shape == (e, r, i) and mb2.shape == (e, h, r)

    def test_mismatched_scalings_are_refused(self, monkeypatch):
        experts = SimpleNamespace(num_experts=1, num_experts_global=1)
        experts._sonicmoe_lora = {
            "gate_up_proj": (torch.zeros(1, 4), torch.zeros(4, 1), 2.0),
            "down_proj": (torch.zeros(1, 2), torch.zeros(4, 1), 1.0),
        }
        monkeypatch.setattr(fused_lora, "get_state", lambda: SimpleNamespace())
        with pytest.raises(ValueError, match="share one LoRA scaling"):
            fused_lora.forward(
                SimpleNamespace(),
                experts,
                torch.zeros(1, 4),
                None,
                None,
                torch.zeros(1, 4, 4),
                torch.zeros(1, 4, 2),
            )


@pytest.fixture
def lora_package(optmoe_package, monkeypatch):
    """``optmoe.lora`` stand-ins recording the layers registered with the cache."""

    class MergedWeightCache:
        def __init__(self):
            self.mode, self.version = "layer", 0

        def invalidate(self):
            self.version += 1

    class LoraExperts:
        def __init__(self, scaling):
            self.scaling = float(scaling)

    lora = types.ModuleType("optmoe.lora")
    lora.MergedWeightCache = MergedWeightCache
    lora.LoraExperts = LoraExperts
    lora.registered = []
    lora.register_layer = lambda entry, cache: lora.registered.append((entry, cache))
    monkeypatch.setitem(sys.modules, "optmoe.lora", lora)
    optmoe_package.lora = lora
    yield lora
    fused_lora.set_state(None)


class TestState:
    def _built(self, cuda_capability, monkeypatch, model, **kw):
        monkeypatch.setattr(
            ExpertParallelPlugin, "_resolve_ep_group", staticmethod(lambda cfg: None)
        )
        cuda_capability((10, 0))
        cfg = _bf16_lora_cfg(**kw)
        plugin = ExpertParallelPlugin()
        plugin.post_model_build(cfg, model)
        plugin._build_optmoe_runtime(cfg)
        return plugin, cfg

    def test_runtime_build_creates_and_unload_clears_the_state(
        self, lora_package, cuda_capability, monkeypatch
    ):
        plugin, cfg = self._built(
            cuda_capability,
            monkeypatch,
            _qwen3_model(),
            expert_parallel_optmoe_lora_merged_cache="all",
        )
        state = fused_lora.get_state()
        assert state is not None
        assert os.environ["OPTMOE_LORA_MERGED_CACHE"] == "all"
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
        optimizer.step()
        assert state.cache.version == 1
        plugin.post_train_unload(cfg)
        assert fused_lora.get_state() is None

    def test_lora_forward_ops_are_saved_under_sac(
        self, lora_package, cuda_capability, monkeypatch
    ):
        from torch.utils.checkpoint import CheckpointPolicy

        from axolotl.monkeypatch.selective_checkpointing import registered_save_policy

        self._built(cuda_capability, monkeypatch, _qwen3_model())
        for name in ("optmoe::lora_forward", "optmoe::nvfp4_lora_forward"):
            op = SimpleNamespace(name=lambda name=name: name)
            assert registered_save_policy(op, {}) == CheckpointPolicy.MUST_SAVE

    def test_layers_register_once(self, lora_package):
        state = fused_lora.FusedLoraState()
        model = _qwen3_model()
        experts = _adapt(model)
        assert state.prepare_layers(model) == 1
        layer = state.layers[id(experts)]
        assert lora_package.registered == [(layer, state.cache)]
        assert state.layer(experts, 2.0) is layer
        assert len(lora_package.registered) == 1
        state.close()

    def test_close_drops_the_package_registry_entries(self, lora_package):
        lora_package._LAYERS = {}

        def register(entry, cache):
            entry.key = len(lora_package._LAYERS)
            lora_package._LAYERS[entry.key] = (entry, cache)

        lora_package.register_layer = register
        other = SimpleNamespace(key=99)
        lora_package._LAYERS[99] = (other, None)
        state = fused_lora.FusedLoraState()
        model = _qwen3_model()
        _adapt(model)
        assert state.prepare_layers(model) == 1
        assert len(lora_package._LAYERS) == 2
        state.close()
        assert list(lora_package._LAYERS) == [99]

    def test_package_without_register_layer_still_builds(self, lora_package):
        del lora_package.register_layer
        state = fused_lora.FusedLoraState()
        model = _qwen3_model()
        _adapt(model)
        assert state.prepare_layers(model) == 1
        state.close()

    def test_train_begin_registers_the_adapted_layers(
        self, lora_package, cuda_capability, monkeypatch
    ):
        model = _qwen3_model()
        plugin, cfg = self._built(cuda_capability, monkeypatch, model, lora_r=16)
        experts = _adapt(model)
        accelerator = SimpleNamespace(
            state=SimpleNamespace(fsdp_plugin=None), unwrap_model=lambda m: m
        )
        trainer = SimpleNamespace(model=model, accelerator=accelerator)
        (callback,) = plugin.add_callbacks_post_trainer(cfg, trainer)
        assert lora_package.registered == []
        callback.on_train_begin(None, None, None)
        state = fused_lora.get_state()
        assert lora_package.registered == [(state.layers[id(experts)], state.cache)]

    def test_prepared_layers_trace_without_graph_breaks(self, lora_package):
        e, r, h, i = 2, 2, 256, 128
        state = fused_lora.FusedLoraState()
        fused_lora.set_state(state)
        layers = []
        for _ in range(12):
            experts = SimpleNamespace(
                num_experts=e, num_experts_global=e, local_expert_offset=0
            )
            experts._sonicmoe_lora = {
                "gate_up_proj": (torch.randn(e * r, h), torch.randn(2 * i, r * e), 2.0),
                "down_proj": (torch.randn(e * r, i), torch.randn(h, r * e), 2.0),
            }
            experts._sonicmoe_lora_wrappers = {}
            state.layer(experts, 2.0)
            layers.append(experts)
        runtime = SimpleNamespace(
            precision="bf16_lora",
            forward_lora=lambda x, idx, w, a1, b1, a2, b2, layer, cache, w1, w2: (
                x + a1.sum() + b2.sum() + layer.scaling
            ),
        )
        idx, w = torch.zeros(5, 2, dtype=torch.int64), torch.ones(5, 2)
        w1, w2 = torch.zeros(e, 2 * i, h), torch.zeros(e, h, i)

        def run(x):
            for experts in layers:
                x = fused_lora.forward(runtime, experts, x, idx, w, w1, w2)
            return x

        x = torch.randn(5, h)
        compiled = torch.compile(run, backend="eager", fullgraph=True)
        torch.testing.assert_close(compiled(x), run(x))
