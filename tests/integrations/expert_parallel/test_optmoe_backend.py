"""CPU tests for the ``optmoe`` expert-parallel backend seam (fused and dispatch modes)."""

import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from axolotl.integrations.expert_parallel import (
    ExpertParallelArgs,
    ExpertParallelPlugin,
    experts_fn,
    plugin as plugin_mod,
)


def _package_api():
    spec = importlib.util.find_spec("optmoe")
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations))) / "api.py"


# resolved before any test replaces ``optmoe`` in sys.modules
PACKAGE_API = _package_api()


def _qwen3_model(hidden=512, intermediate=1024, hidden_act="silu", num_experts=4):
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

    config = Qwen3MoeConfig(
        hidden_size=hidden,
        moe_intermediate_size=intermediate,
        num_experts=num_experts,
        num_experts_per_tok=2,
        hidden_act=hidden_act,
    )
    model = torch.nn.Module()
    model.config = config
    model.mlp = Qwen3MoeSparseMoeBlock(config)
    return model


def _cfg(**kw):
    defaults = dict(
        expert_parallel_size=2,
        expert_parallel_backend="optmoe",
        expert_parallel_optmoe_mode="auto",
        expert_parallel_optmoe_precision="auto",
        expert_parallel_fallback_on_unsupported=True,
        expert_parallel_save_dispatch=False,
        expert_parallel_token_capacity=None,
        expert_parallel_dispatch_chunks=1,
        experts_implementation=None,
        adapter=None,
        dp_shard_size=None,
        dp_replicate_size=None,
        context_parallel_size=None,
        sequence_len=1000,
        micro_batch_size=2,
        eval_batch_size=None,
        activation_checkpointing={"enabled": True},
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


class _FakeRuntime:
    forward_op = "optmoe::persistent_forward"
    instances: list["_FakeRuntime"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.destroyed = False
        precision = kwargs.get("precision", "auto")
        self.precision = "bf16" if precision == "auto" else precision
        self.max_tokens_per_rank = kwargs.get("max_tokens_per_rank")
        self.prepared = []
        _FakeRuntime.instances.append(self)

    @staticmethod
    def token_alignment():
        return 128

    def forward(self, x, topk_idx, topk_weights, w1, w2):
        self.calls.append((x, topk_idx, topk_weights, w1, w2))
        return x * 2

    def prepare(self, **kwargs):
        self.prepared.append(kwargs)

    def destroy(self):
        self.destroyed = True


class _FakeDispatchRuntime:
    """Single-rank stand-in for ``OptMoEDispatchRuntime``: the fixed-capacity DeepEP
    layout of one rank (``capacity`` rows, sentinel rows past the batch), plain torch so
    the seam can be compiled around it."""

    dispatch_op = "optmoe::peer_dispatch"
    combine_op = "optmoe::peer_combine"
    instances: list["_FakeDispatchRuntime"] = []

    def __init__(self, capacity=8, **kwargs):
        self.kwargs = kwargs
        self.capacity = kwargs.get("max_tokens_per_rank", capacity)
        self.max_tokens_per_rank = self.capacity
        self.receive_rows = self.capacity
        self.calls = []
        self.destroyed = False
        _FakeDispatchRuntime.instances.append(self)

    def dispatch(self, x, topk_idx, topk_weights, num_local_experts):
        self.calls.append(("dispatch", x, topk_idx, topk_weights, num_local_experts))
        pad = self.capacity - x.shape[0]
        # rank 0 of the layout: experts [0, E_local) are local, the rest are remote (-1)
        local = torch.where(
            topk_idx < num_local_experts, topk_idx, torch.full_like(topk_idx, -1)
        )
        recv_idx = torch.nn.functional.pad(local, (0, 0, 0, pad), value=-1)
        recv_w = torch.nn.functional.pad(topk_weights, (0, 0, 0, pad))
        recv_w = recv_w.masked_fill(recv_idx < 0, 0)
        handle = ((recv_idx >= 0).any(-1), topk_idx, num_local_experts)
        return torch.nn.functional.pad(x, (0, 0, 0, pad)), recv_idx, recv_w, handle

    def combine(self, local_out, handle):
        rows, topk_idx, num_local_experts = handle
        self.calls.append(("combine", local_out, rows, topk_idx, num_local_experts))
        return local_out[: topk_idx.shape[0]]

    def destroy(self):
        self.destroyed = True


@pytest.fixture
def optmoe_package(monkeypatch):
    """The real package API (pure Python) with a fake runtime class, installed as
    ``optmoe`` so the plugin's imports and ``find_spec`` resolve to it."""
    if PACKAGE_API is None or not PACKAGE_API.exists():
        pytest.skip("the optmoe package is not installed")
    spec = importlib.util.spec_from_file_location("_optmoe_api", PACKAGE_API)
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    module = types.ModuleType("optmoe")
    module.__spec__ = importlib.machinery.ModuleSpec("optmoe", None)
    for name in (
        "DEFAULT_RESERVED_SMS",
        "MODES",
        "check_nccl_env",
        "dispatch_unsupported_reason",
        "lora_precision",
        "nccl_env",
        "resolve_precision",
        "unsupported_reason",
    ):
        setattr(module, name, getattr(api, name))
    module.OptMoERuntime = _FakeRuntime
    module.OptMoEDispatchRuntime = _FakeDispatchRuntime
    monkeypatch.setitem(sys.modules, "optmoe", module)
    _FakeRuntime.instances = []
    _FakeDispatchRuntime.instances = []
    return module


@pytest.fixture
def no_package(monkeypatch):
    monkeypatch.setattr(plugin_mod, "find_spec", lambda name: None)


@pytest.fixture
def cuda_capability(monkeypatch):
    def set_capability(capability):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: capability is not None)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)

    set_capability((9, 0))
    return set_capability


@pytest.fixture(autouse=True)
def _restore_ep_state():
    from axolotl.integrations.expert_parallel.torch_dispatch import set_ep_group
    from axolotl.monkeypatch.selective_checkpointing import clear_registered_saves

    yield
    experts_fn.set_backend(None)
    experts_fn.set_local_implementation("grouped_mm")
    experts_fn.set_optmoe_runtime(None)
    experts_fn.set_optmoe_mode("fused")
    experts_fn.set_valid_token_mask(None)
    experts_fn.set_token_capacity(None)
    set_ep_group(None)
    clear_registered_saves()


def _sharded_experts():
    experts = _qwen3_model(num_experts=4).mlp.experts
    experts.num_experts_global = 4
    experts.num_local_experts = 2
    return experts


class TestFusedForward:
    @staticmethod
    def _experts():
        return _sharded_experts()

    def test_bypasses_local_implementation(self, monkeypatch):
        runtime = _FakeRuntime()
        experts_fn.set_optmoe_runtime(runtime)
        experts_fn.set_optmoe_mode("fused")

        def forbidden(*args, **kwargs):
            raise AssertionError("the fused backend must not run the local path")

        monkeypatch.setattr(experts_fn, "_run_local", forbidden)
        monkeypatch.setattr(experts_fn, "resolve_local_implementation", forbidden)
        experts = self._experts()
        x = torch.randn(3, 512)
        ids = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.int32)
        weights = torch.rand(3, 2, dtype=torch.bfloat16)
        y = experts_fn._ep_forward(experts, x, ids, weights, backend="optmoe")

        assert y.dtype == torch.float32  # cast back after the BF16 kernel
        (rx, ridx, rw, w1, w2) = runtime.calls[0]
        assert rx.dtype == torch.bfloat16
        assert ridx.dtype == torch.int64 and ridx.tolist() == ids.tolist()
        assert rw.dtype == torch.float32
        assert w1 is experts.gate_up_proj and w2 is experts.down_proj
        torch.testing.assert_close(y, (x.bfloat16() * 2).float())

    def test_padding_sentinel_and_capacity_apply_first(self):
        runtime = _FakeRuntime()
        experts_fn.set_optmoe_runtime(runtime)
        experts_fn.set_valid_token_mask(torch.tensor([True, False, True, True]))
        experts_fn.set_token_capacity(2)
        ids = torch.tensor([[0, 1], [0, 1], [0, 2], [0, 3]])
        weights = torch.tensor([[0.6, 0.4], [0.9, 0.1], [0.7, 0.3], [0.5, 0.5]])
        experts_fn._ep_forward(
            self._experts(),
            torch.randn(4, 512, dtype=torch.bfloat16),
            ids,
            weights,
            backend="optmoe",
        )
        _, ridx, rw, _, _ = runtime.calls[0]
        assert ridx[1].tolist() == [-1, -1]  # padding row
        # expert 0 keeps its two highest-weight real tokens; token 3 drops it and its
        # surviving weight is rescaled to the original gate sum
        assert ridx[:, 0].tolist() == [0, -1, 0, -1]
        torch.testing.assert_close(rw[3], torch.tensor([0.0, 1.0]))

    def test_unbuilt_runtime_raises(self):
        with pytest.raises(RuntimeError, match="no runtime built"):
            experts_fn._ep_forward(
                self._experts(),
                torch.randn(1, 512),
                torch.zeros(1, 2, dtype=torch.long),
                torch.ones(1, 2),
                backend="optmoe",
            )

    def test_set_backend_accepts_optmoe(self):
        experts_fn.set_backend("optmoe")
        assert experts_fn.get_backend() == "optmoe"

    def test_mode_override_and_validation(self):
        experts_fn.set_optmoe_mode("dispatch")
        assert experts_fn.get_optmoe_mode() == "dispatch"
        with pytest.raises(ValueError, match="unknown optmoe mode"):
            experts_fn.set_optmoe_mode("auto")
        assert experts_fn._OptMoEBackend.resolve() is experts_fn._OptMoEDispatchBackend
        assert (
            experts_fn._OptMoEBackend.resolve("fused") is experts_fn._OptMoEFusedBackend
        )


class TestDispatchForward:
    """The dispatch mode is a regular dispatch/combine backend around ``_run_local``."""

    @staticmethod
    def _inputs(tokens=3):
        x = torch.randn(tokens, 512)
        ids = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.int32)[:tokens]
        weights = torch.rand(tokens, 2, dtype=torch.bfloat16)
        return x, ids, weights

    def test_runs_the_configured_local_kernel(self, monkeypatch):
        runtime = _FakeDispatchRuntime(capacity=8)
        experts_fn.set_optmoe_runtime(runtime)
        experts_fn.set_optmoe_mode("dispatch")
        seen = {}

        def local(experts, recv_x, recv_idx, recv_w):
            seen.update(x=recv_x, idx=recv_idx, w=recv_w)
            return recv_x * 3

        monkeypatch.setattr(
            experts_fn, "resolve_local_implementation", lambda name=None: local
        )
        experts = _sharded_experts()
        x, ids, weights = self._inputs()
        y = experts_fn._ep_forward(experts, x, ids, weights, backend="optmoe")

        assert runtime.calls[0][0] == "dispatch" and runtime.calls[1][0] == "combine"
        _, rx, ridx, rw, e_local = runtime.calls[0]
        assert rx.dtype == torch.bfloat16 and ridx.dtype == torch.int64
        assert rw.dtype == torch.float32 and e_local == 2
        # the local kernel sees the fixed-capacity rows plus the per-expert pad rows,
        # with the dispatch's -1 slots rewritten to E_local / weight 0
        assert seen["x"].shape == (8 + 2, 512)
        assert seen["idx"][3:8].eq(2).all() and seen["w"][3:8].eq(0).all()
        assert seen["idx"][:3].tolist() == [[0, 1], [1, 2], [2, 2]]
        assert seen["w"][2].eq(0).all() and seen["w"][1, 1] == 0
        assert seen["w"][0].gt(0).all() and seen["w"][1, 0] > 0
        _, local_out, rows, topk_idx, _ = runtime.calls[1]
        assert local_out.shape == (8, 512)
        # token 2's experts [2, 3] are both remote, so its receive row is unrouted
        assert rows.dtype == torch.bool and rows.tolist() == [True, True] + [False] * 6
        assert topk_idx.tolist() == ids.tolist()
        assert y.dtype == torch.float32
        torch.testing.assert_close(y, (x.bfloat16() * 3).float())

    def test_padding_sentinel_and_capacity_apply_first(self, monkeypatch):
        runtime = _FakeDispatchRuntime(capacity=8)
        experts_fn.set_optmoe_runtime(runtime)
        experts_fn.set_optmoe_mode("dispatch")
        monkeypatch.setattr(
            experts_fn,
            "resolve_local_implementation",
            lambda name=None: lambda experts, rx, ri, rw: rx,
        )
        experts_fn.set_valid_token_mask(torch.tensor([True, False, True, True]))
        experts_fn.set_token_capacity(2)
        ids = torch.tensor([[0, 1], [0, 1], [0, 2], [0, 3]])
        weights = torch.tensor([[0.6, 0.4], [0.9, 0.1], [0.7, 0.3], [0.5, 0.5]])
        experts_fn._ep_forward(
            _sharded_experts(),
            torch.randn(4, 512, dtype=torch.bfloat16),
            ids,
            weights,
            backend="optmoe",
        )
        _, _, ridx, rw, _ = runtime.calls[0]
        assert ridx[1].tolist() == [-1, -1]
        assert ridx[:, 0].tolist() == [0, -1, 0, -1]
        torch.testing.assert_close(rw[3], torch.tensor([0.0, 1.0]))

    def test_fused_mode_bypasses_local_kernel(self, monkeypatch):
        experts_fn.set_optmoe_runtime(_FakeRuntime())
        experts_fn.set_optmoe_mode("dispatch")

        def forbidden(*args, **kwargs):
            raise AssertionError("fused mode must not run the local path")

        monkeypatch.setattr(experts_fn, "_run_local", forbidden)
        x, ids, weights = self._inputs()
        y = experts_fn._ep_forward(
            _sharded_experts(), x, ids, weights, backend="optmoe", optmoe_mode="fused"
        )
        torch.testing.assert_close(y, (x.bfloat16() * 2).float())

    def test_compiles_fullgraph(self, monkeypatch):
        from transformers.integrations.moe import grouped_mm_experts_forward

        runtime = _FakeDispatchRuntime(capacity=8)
        experts_fn.set_optmoe_runtime(runtime)
        experts_fn.set_optmoe_mode("dispatch")
        monkeypatch.setattr(
            experts_fn,
            "resolve_local_implementation",
            lambda name=None: grouped_mm_experts_forward,
        )
        experts = _sharded_experts().bfloat16()
        x, ids, weights = self._inputs()
        x = x.bfloat16().requires_grad_(True)
        weights = weights.float().requires_grad_(True)
        eager = experts_fn._ep_forward(experts, x, ids, weights, backend="optmoe")
        torch._dynamo.reset()
        compiled = torch.compile(experts_fn._ep_forward, fullgraph=True)
        y = compiled(experts, x, ids, weights, backend="optmoe")
        torch.testing.assert_close(y, eager)
        y.float().sum().backward()
        assert x.grad is not None and weights.grad is not None
        assert experts.gate_up_proj.grad is not None


class TestArgs:
    def test_backend_and_precision_keys(self):
        args = ExpertParallelArgs(
            expert_parallel_backend="optmoe",
            expert_parallel_optmoe_precision="fp8",
        )
        assert args.expert_parallel_backend == "optmoe"
        assert args.expert_parallel_optmoe_precision == "fp8"
        assert ExpertParallelArgs().expert_parallel_optmoe_precision == "auto"
        assert (
            ExpertParallelArgs(
                expert_parallel_optmoe_precision="nvfp4_lora"
            ).expert_parallel_optmoe_precision
            == "nvfp4_lora"
        )
        with pytest.raises(ValueError):
            ExpertParallelArgs(expert_parallel_optmoe_precision="fp4")

    @pytest.mark.parametrize("mode", ["auto", "fused", "dispatch"])
    def test_mode_key(self, mode):
        args = ExpertParallelArgs(
            expert_parallel_backend="optmoe", expert_parallel_optmoe_mode=mode
        )
        assert args.expert_parallel_optmoe_mode == mode
        assert ExpertParallelArgs().expert_parallel_optmoe_mode == "auto"
        with pytest.raises(ValueError):
            ExpertParallelArgs(expert_parallel_optmoe_mode="peer")


class TestBackendResolution:
    @pytest.mark.parametrize("installed", [True, False])
    def test_auto_never_selects_optmoe(self, monkeypatch, installed):
        monkeypatch.setattr(
            plugin_mod, "find_spec", lambda name: object() if installed else None
        )
        cfg = _cfg(expert_parallel_backend="auto")
        assert ExpertParallelPlugin._resolve_backend(cfg) == (
            "deep_ep" if installed else "torch"
        )
        assert ExpertParallelPlugin().get_distributed_env(cfg) == {}

    def test_explicit_optmoe(self, optmoe_package):
        cfg = _cfg()
        assert ExpertParallelPlugin._resolve_backend(cfg) == "optmoe"
        assert cfg.expert_parallel_backend == "optmoe"

    def test_missing_package_falls_back(self, no_package, caplog):
        cfg = _cfg()
        with caplog.at_level("WARNING"):
            assert ExpertParallelPlugin._resolve_backend(cfg) == "torch"
        assert cfg.expert_parallel_backend == "torch"
        assert "not installed" in caplog.text

    def test_missing_package_raises_without_fallback(self, no_package):
        cfg = _cfg(expert_parallel_fallback_on_unsupported=False)
        with pytest.raises(ImportError, match="not installed"):
            ExpertParallelPlugin._resolve_backend(cfg)

    @pytest.mark.parametrize(
        "extra,match",
        [
            ({"adapter": "lora"}, "adapter"),
            ({"adapter": "qlora", "load_in_4bit": True}, "adapter"),
            ({"load_in_8bit": True}, "load_in_8bit"),
            ({"gptq": True}, "gptq"),
            ({"model_quantization_config": "Mxfp4Config"}, "model_quantization_config"),
        ],
    )
    def test_config_reasons_force_fused_to_fall_back(
        self, optmoe_package, extra, match
    ):
        cfg = _cfg(expert_parallel_optmoe_mode="fused", **extra)
        assert ExpertParallelPlugin._resolve_backend(cfg) == "torch"
        cfg = _cfg(
            expert_parallel_optmoe_mode="fused",
            expert_parallel_fallback_on_unsupported=False,
            **extra,
        )
        with pytest.raises(ValueError, match=match):
            ExpertParallelPlugin._resolve_backend(cfg)

    @pytest.mark.parametrize("mode", ["auto", "dispatch"])
    @pytest.mark.parametrize("extra", [{"adapter": "lora"}, {"load_in_8bit": True}])
    def test_config_reasons_keep_optmoe_in_other_modes(
        self, optmoe_package, mode, extra
    ):
        cfg = _cfg(
            expert_parallel_optmoe_mode=mode,
            expert_parallel_fallback_on_unsupported=False,
            **extra,
        )
        assert ExpertParallelPlugin._resolve_backend(cfg) == "optmoe"
        assert not ExpertParallelPlugin._optmoe_fused_possible(cfg)

    def test_forced_fused_lora_with_sonicmoe_lands_on_torch_sonicmoe(
        self, monkeypatch, optmoe_package
    ):
        registered = []
        monkeypatch.setattr(
            "axolotl.integrations.kernels.libs.sonicmoe.experts.register_sonicmoe_experts",
            lambda: registered.append("sonicmoe"),
        )
        cfg = _cfg(
            adapter="lora", use_sonicmoe=True, expert_parallel_optmoe_mode="fused"
        )
        ExpertParallelPlugin().pre_model_load(cfg)
        assert cfg.expert_parallel_backend == "torch"
        assert experts_fn.get_backend() == "torch"
        assert experts_fn.get_local_implementation() == "sonicmoe"
        assert registered == ["sonicmoe"]

    def test_auto_lora_with_sonicmoe_picks_dispatch_mode(
        self, monkeypatch, optmoe_package, cuda_capability
    ):
        monkeypatch.setattr(
            "axolotl.integrations.kernels.libs.sonicmoe.experts.register_sonicmoe_experts",
            lambda: None,
        )
        monkeypatch.delenv("NCCL_MAX_CTAS", raising=False)
        cfg = _cfg(adapter="lora", use_sonicmoe=True)
        plugin = ExpertParallelPlugin()
        plugin.pre_model_load(cfg)  # no NCCL_MAX_CTAS needed: fused is already out
        assert cfg.expert_parallel_backend == "optmoe"
        assert experts_fn.get_optmoe_mode() == "dispatch"
        assert experts_fn.get_local_implementation() == "sonicmoe"
        plugin.post_model_build(cfg, _qwen3_model())
        assert cfg.expert_parallel_backend == "optmoe"
        assert cfg.expert_parallel_optmoe_mode == "dispatch"
        assert experts_fn.get_backend() == "optmoe"
        assert plugin._optmoe_dispatch_geometry == plugin_mod.OptMoEDispatchGeometry(
            hidden=512, top_k=2
        )


class TestModelFallback:
    @pytest.mark.parametrize(
        "build,mutate,match",
        [
            ({"hidden": 384}, None, "multiple of 256"),
            ({"intermediate": 1000}, None, "multiple of 128"),
            ({"hidden_act": "gelu"}, None, "SwiGLU"),
            ({}, lambda e: setattr(e, "has_bias", True), "biases"),
            ({}, lambda e: setattr(e, "limit", 7.0), "clamped"),
            ({}, lambda e: setattr(e, "alpha", 1.702), "alpha"),
            ({}, lambda e: setattr(e, "has_gate", False), "not gated"),
            ({}, lambda e: setattr(e, "is_transposed", True), "transposed"),
            ({}, lambda e: setattr(e, "has_post_expert_norm", True), "post-expert"),
        ],
    )
    def test_unsupported_model_falls_back_to_torch_in_fused_mode(
        self, optmoe_package, cuda_capability, caplog, build, mutate, match
    ):
        model = _qwen3_model(**build)
        if mutate is not None:
            mutate(model.mlp.experts)
        cfg = _cfg(expert_parallel_optmoe_mode="fused")
        with caplog.at_level("WARNING"):
            ExpertParallelPlugin().post_model_build(cfg, model)
        assert match in caplog.text
        assert cfg.expert_parallel_backend == "torch"
        assert experts_fn.get_backend() == "torch"

        model = _qwen3_model(**build)
        if mutate is not None:
            mutate(model.mlp.experts)
        with pytest.raises(ValueError, match=match):
            ExpertParallelPlugin().post_model_build(
                _cfg(
                    expert_parallel_optmoe_mode="fused",
                    expert_parallel_fallback_on_unsupported=False,
                ),
                model,
            )

    @pytest.mark.parametrize(
        "build,mutate,match",
        [
            ({"hidden": 384}, None, "multiple of 256"),
            ({"hidden_act": "gelu"}, None, "SwiGLU"),
            ({}, lambda e: setattr(e, "has_bias", True), "biases"),
            ({}, lambda e: setattr(e, "limit", 7.0), "clamped"),
            ({}, lambda e: setattr(e, "has_gate", False), "not gated"),
        ],
    )
    def test_unsupported_model_uses_dispatch_mode_in_auto(
        self, optmoe_package, cuda_capability, caplog, build, mutate, match
    ):
        model = _qwen3_model(**build)
        if mutate is not None:
            mutate(model.mlp.experts)
        cfg = _cfg(expert_parallel_fallback_on_unsupported=False)
        plugin = ExpertParallelPlugin()
        with caplog.at_level("INFO"):
            plugin.post_model_build(cfg, model)
        assert match in caplog.text and "peer dispatch mode" in caplog.text
        assert cfg.expert_parallel_backend == "optmoe"
        assert cfg.expert_parallel_optmoe_mode == "dispatch"
        assert experts_fn.get_backend() == "optmoe"
        assert experts_fn.get_optmoe_mode() == "dispatch"
        assert plugin._optmoe_geometry is None
        assert plugin._optmoe_dispatch_geometry.top_k == 2

    def test_forced_dispatch_skips_fused_checks(
        self, optmoe_package, cuda_capability, caplog
    ):
        model = _qwen3_model(hidden_act="gelu")
        cfg = _cfg(
            expert_parallel_optmoe_mode="dispatch",
            expert_parallel_fallback_on_unsupported=False,
        )
        with caplog.at_level("INFO"):
            ExpertParallelPlugin().post_model_build(cfg, model)
        assert "SwiGLU" not in caplog.text
        assert cfg.expert_parallel_optmoe_mode == "dispatch"
        assert experts_fn.get_optmoe_mode() == "dispatch"

    @pytest.mark.parametrize(
        "capability,match",
        [(None, "CUDA is unavailable"), ((7, 0), "sm_70")],
    )
    def test_dispatch_unsupported_falls_back_to_torch(
        self, optmoe_package, cuda_capability, caplog, capability, match
    ):
        cuda_capability(capability)
        cfg = _cfg()
        with caplog.at_level("WARNING"):
            ExpertParallelPlugin().post_model_build(cfg, _qwen3_model())
        assert "peer dispatch" in caplog.text and match in caplog.text
        assert cfg.expert_parallel_backend == "torch"
        assert experts_fn.get_backend() == "torch"
        with pytest.raises(ValueError, match=match):
            ExpertParallelPlugin().post_model_build(
                _cfg(expert_parallel_fallback_on_unsupported=False), _qwen3_model()
            )

    @pytest.mark.parametrize("capability", [(8, 0), (12, 0)])
    def test_auto_dispatch_on_hardware_without_fused_kernels(
        self, optmoe_package, cuda_capability, capability
    ):
        cuda_capability(capability)
        cfg = _cfg(expert_parallel_fallback_on_unsupported=False)
        ExpertParallelPlugin().post_model_build(cfg, _qwen3_model())
        assert cfg.expert_parallel_backend == "optmoe"
        assert cfg.expert_parallel_optmoe_mode == "dispatch"

    @pytest.mark.parametrize(
        "capability,precision,match",
        [
            (None, "auto", "CUDA is unavailable"),
            ((8, 0), "auto", "< SM90"),
            ((12, 0), "auto", "sm_120"),
            ((9, 0), "mxfp8", "requires SM100"),
            ((10, 0), "fp8", "requires SM90"),
            ((9, 0), "fp8_bwd", "not implemented"),
        ],
    )
    def test_hardware_and_precision_reasons(
        self, optmoe_package, cuda_capability, capability, precision, match
    ):
        cuda_capability(capability)
        cfg = _cfg(expert_parallel_optmoe_precision=precision)
        geometry, reason = ExpertParallelPlugin._optmoe_model_reason(
            cfg, _qwen3_model()
        )
        assert geometry is None and match in reason

    def test_fp32_plain_gate_override_is_accepted(
        self, optmoe_package, cuda_capability
    ):
        cuda_capability((9, 0))
        model = _qwen3_model()
        experts = model.mlp.experts

        def _apply_gate(self, gate_up):
            gate, up = gate_up.chunk(2, dim=-1)
            return (self.act_fn(gate.float()) * up.float()).to(gate.dtype)

        experts.__class__ = type(
            "BeamLikeExperts", (type(experts),), {"_apply_gate": _apply_gate}
        )
        cfg = _cfg()
        ExpertParallelPlugin().post_model_build(cfg, model)
        assert cfg.expert_parallel_backend == "optmoe"
        assert experts_fn.get_backend() == "optmoe"

    def test_different_gate_override_falls_back(
        self, optmoe_package, cuda_capability, caplog
    ):
        cuda_capability((9, 0))
        model = _qwen3_model()
        experts = model.mlp.experts

        def _apply_gate(self, gate_up):
            gate, up = gate_up.chunk(2, dim=-1)
            return torch.nn.functional.gelu(gate) * up

        experts.__class__ = type(
            "GeluGateExperts", (type(experts),), {"_apply_gate": _apply_gate}
        )
        cfg = _cfg(expert_parallel_optmoe_mode="fused")
        with caplog.at_level("WARNING"):
            ExpertParallelPlugin().post_model_build(cfg, model)
        assert "custom gated activation" in caplog.text
        assert cfg.expert_parallel_backend == "torch"

    @pytest.mark.parametrize("capability", [(9, 0), (10, 0)])
    def test_supported_model_keeps_backend(
        self, optmoe_package, cuda_capability, capability
    ):
        from axolotl.monkeypatch.selective_checkpointing import has_registered_saves

        cuda_capability(capability)
        cfg = _cfg()
        plugin = ExpertParallelPlugin()
        plugin.post_model_build(cfg, _qwen3_model())
        assert cfg.expert_parallel_backend == "optmoe"
        assert cfg.expert_parallel_optmoe_mode == "fused"
        assert experts_fn.get_backend() == "optmoe"
        assert experts_fn.get_optmoe_mode() == "fused"
        assert plugin._optmoe_geometry == plugin_mod.OptMoEGeometry(
            num_experts=4, hidden=512, intermediate=1024, top_k=2, activation_clamp=None
        )
        assert has_registered_saves()


class TestRuntimeLifecycle:
    @staticmethod
    def _built(cfg, monkeypatch):
        group = object()
        monkeypatch.setattr(
            ExpertParallelPlugin, "_resolve_ep_group", staticmethod(lambda cfg: group)
        )
        plugin = ExpertParallelPlugin()
        plugin._optmoe_geometry = plugin_mod.OptMoEGeometry(8, 512, 1024, 2, None)
        cfg.activation_checkpointing = {"enabled": False}
        plugin.post_model_load(cfg, torch.nn.Linear(2, 2))
        return plugin, group

    @pytest.mark.parametrize(
        "extra,tokens",
        [
            ({}, 2048),  # 1000 * 2 rounded up to 128
            ({"eval_batch_size": 3}, 3072),
            ({"context_parallel_size": 2}, 1024),
            ({"sequence_len": 4096, "micro_batch_size": 1}, 4096),
        ],
    )
    def test_built_once_and_destroyed(self, monkeypatch, optmoe_package, extra, tokens):
        from torch.utils.checkpoint import CheckpointPolicy

        from axolotl.monkeypatch.selective_checkpointing import registered_save_policy

        cfg = _cfg(**extra)
        plugin, group = self._built(cfg, monkeypatch)
        (runtime,) = _FakeRuntime.instances
        assert runtime.kwargs == {
            "group": group,
            "num_experts": 8,
            "max_tokens_per_rank": tokens,
            "top_k": 2,
            "hidden": 512,
            "intermediate": 1024,
            "precision": "auto",
            "reserved_sms": 16,
            "activation_clamp": None,
        }
        assert experts_fn.get_optmoe_runtime() is runtime
        op = SimpleNamespace(name=lambda: "optmoe::persistent_forward")
        assert registered_save_policy(op, {}) == CheckpointPolicy.MUST_SAVE

        plugin.post_model_load(cfg, torch.nn.Linear(2, 2))
        assert len(_FakeRuntime.instances) == 1

        plugin.post_train_unload(cfg)
        assert runtime.destroyed
        with pytest.raises(RuntimeError, match="no runtime built"):
            experts_fn.get_optmoe_runtime()
        plugin.post_train_unload(cfg)

    @staticmethod
    def _trainer(model, param_dtype=None, fsdp_version=2):
        fsdp_plugin = SimpleNamespace(
            fsdp_version=fsdp_version,
            mixed_precision_policy=SimpleNamespace(param_dtype=param_dtype),
        )
        accelerator = SimpleNamespace(
            state=SimpleNamespace(fsdp_plugin=fsdp_plugin), unwrap_model=lambda m: m
        )
        return SimpleNamespace(model=model, accelerator=accelerator)

    @pytest.mark.parametrize(
        "param_dtype,fsdp_version,expected",
        [
            (None, 2, torch.float32),
            (torch.bfloat16, 2, torch.bfloat16),
            (torch.bfloat16, 1, torch.float32),
        ],
    )
    def test_prepared_at_train_begin(
        self, monkeypatch, optmoe_package, param_dtype, fsdp_version, expected
    ):
        cfg = _cfg()
        plugin, _ = self._built(cfg, monkeypatch)
        (runtime,) = _FakeRuntime.instances
        trainer = self._trainer(_qwen3_model(), param_dtype, fsdp_version)
        (callback,) = plugin.add_callbacks_post_trainer(cfg, trainer)
        assert runtime.prepared == []
        callback.on_train_begin(None, None, None)
        assert runtime.prepared == [{"num_tokens": 2000, "weight_dtype": expected}]

    def test_prepared_with_lora_rank(self, monkeypatch, optmoe_package):
        cfg = _cfg(lora_r=16)
        plugin, _ = self._built(cfg, monkeypatch)
        (runtime,) = _FakeRuntime.instances
        runtime.precision = "bf16_lora"
        trainer = self._trainer(_qwen3_model(), torch.bfloat16)
        (callback,) = plugin.add_callbacks_post_trainer(cfg, trainer)
        callback.on_train_begin(None, None, None)
        assert runtime.prepared == [
            {"num_tokens": 2000, "lora_rank": 16, "lora_dtype": torch.bfloat16}
        ]

    def test_no_callback_without_a_fused_runtime(self, monkeypatch, optmoe_package):
        cfg = _cfg(expert_parallel_backend="torch")
        plugin, _ = self._built(cfg, monkeypatch)
        assert plugin.add_callbacks_post_trainer(cfg, self._trainer(None)) == []

    def test_torch_fallback_builds_nothing(self, monkeypatch, optmoe_package):
        self._built(_cfg(expert_parallel_backend="torch"), monkeypatch)
        assert _FakeRuntime.instances == []
        assert _FakeDispatchRuntime.instances == []

    @pytest.mark.parametrize("save_dispatch", [False, True])
    @pytest.mark.parametrize(
        "compact,bounded", [(False, False), (True, False), (True, True)]
    )
    def test_dispatch_runtime_lifecycle(
        self, monkeypatch, optmoe_package, save_dispatch, compact, bounded
    ):
        from torch.utils.checkpoint import CheckpointPolicy

        from axolotl.monkeypatch.selective_checkpointing import registered_save_policy

        group = object()
        monkeypatch.setattr(
            ExpertParallelPlugin, "_resolve_ep_group", staticmethod(lambda cfg: group)
        )
        cfg = _cfg(
            expert_parallel_optmoe_mode="dispatch",
            expert_parallel_save_dispatch=save_dispatch,
            expert_parallel_optmoe_compact=compact,
            expert_parallel_optmoe_bounded=bounded,
            activation_checkpointing={"enabled": False},
        )
        plugin = ExpertParallelPlugin()
        plugin._optmoe_dispatch_geometry = plugin_mod.OptMoEDispatchGeometry(512, 2)
        plugin.post_model_load(cfg, torch.nn.Linear(2, 2))
        assert _FakeRuntime.instances == []
        (runtime,) = _FakeDispatchRuntime.instances
        assert runtime.kwargs == {
            "group": group,
            "max_tokens_per_rank": 2000,  # no kernel token alignment in dispatch mode
            "top_k": 2,
            "hidden": 512,
            "compact": compact,
            "bounded": bounded,
        }
        assert experts_fn.get_optmoe_runtime() is runtime
        for name in ("optmoe::peer_dispatch", "optmoe::peer_combine"):
            policy = registered_save_policy(SimpleNamespace(name=lambda n=name: n), {})
            # unregistered ops follow the layer policy, i.e. the transfers are replayed
            assert policy == (CheckpointPolicy.PREFER_SAVE if save_dispatch else None)

        plugin.post_model_load(cfg, torch.nn.Linear(2, 2))
        assert len(_FakeDispatchRuntime.instances) == 1
        plugin.post_train_unload(cfg)
        assert runtime.destroyed
        with pytest.raises(RuntimeError, match="no runtime built"):
            experts_fn.get_optmoe_runtime()


class TestDistributedEnv:
    @pytest.mark.parametrize("replicate,ctas", [(None, "8"), (1, "8"), (2, "5")])
    def test_optmoe_exports_nccl_max_ctas(self, optmoe_package, replicate, ctas):
        env = ExpertParallelPlugin().get_distributed_env(
            _cfg(dp_replicate_size=replicate)
        )
        assert env == {"NCCL_MAX_CTAS": ctas}

    @pytest.mark.parametrize(
        "extra",
        [
            {"expert_parallel_backend": "torch"},
            {"expert_parallel_backend": "deep_ep"},
            {"expert_parallel_backend": "auto"},
            {"expert_parallel_size": 1},
            {"adapter": "lora"},
            {"load_in_4bit": True, "expert_parallel_optmoe_mode": "auto"},
            {"expert_parallel_optmoe_mode": "dispatch"},
        ],
    )
    def test_other_backends_export_nothing(self, optmoe_package, extra):
        assert ExpertParallelPlugin().get_distributed_env(_cfg(**extra)) == {}

    def test_dispatch_mode_needs_no_nccl_max_ctas(self, monkeypatch, optmoe_package):
        monkeypatch.delenv("NCCL_MAX_CTAS", raising=False)
        cfg = _cfg(expert_parallel_optmoe_mode="dispatch")
        ExpertParallelPlugin().pre_model_load(cfg)
        assert experts_fn.get_backend() == "optmoe"
        assert experts_fn.get_optmoe_mode() == "dispatch"

    def test_missing_package_exports_nothing(self, no_package):
        assert ExpertParallelPlugin().get_distributed_env(_cfg()) == {}

    @pytest.mark.parametrize("value,ok", [("8", True), ("16", False)])
    def test_pre_model_load_checks_user_value(
        self, monkeypatch, optmoe_package, value, ok
    ):
        monkeypatch.setenv("NCCL_MAX_CTAS", value)
        cfg = _cfg()
        if ok:
            ExpertParallelPlugin().pre_model_load(cfg)
            assert experts_fn.get_backend() == "optmoe"
            assert cfg.experts_implementation == "expert_parallel"
        else:
            with pytest.raises(RuntimeError, match="NCCL_MAX_CTAS"):
                ExpertParallelPlugin().pre_model_load(cfg)
