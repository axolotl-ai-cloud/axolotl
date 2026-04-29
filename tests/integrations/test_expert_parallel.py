"""Tests for the Expert-Parallel (DeepEP) integration."""

import os
from importlib.util import find_spec

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from axolotl.integrations.expert_parallel import (
    ExpertParallelArgs,
    ExpertParallelPlugin,
)
from axolotl.integrations.expert_parallel.experts_fn import (
    REGISTRY,
    kernel_to_registered_name,
    register_all,
)
from axolotl.integrations.expert_parallel.shard import (
    _detect_experts_modules,
    shard_expert_weights,
)


def _build_qwen3moe_block(num_experts: int = 16, top_k: int = 4):
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

    cfg = Qwen3MoeConfig(
        hidden_size=512,
        moe_intermediate_size=1024,
        num_experts=num_experts,
        num_experts_per_tok=top_k,
    )
    return Qwen3MoeSparseMoeBlock(cfg)


# --------------------------------------------------------------------------- #
# Args validation
# --------------------------------------------------------------------------- #


class TestExpertParallelArgs:
    def test_defaults(self):
        a = ExpertParallelArgs()
        assert a.expert_parallel_size == 1
        assert a.expert_parallel_backend == "deep_ep"
        assert a.expert_parallel_fallback_on_unsupported is True

    def test_enabled(self):
        a = ExpertParallelArgs(expert_parallel_size=2)
        assert a.expert_parallel_size == 2

    def test_invalid_size(self):
        with pytest.raises(ValueError):
            ExpertParallelArgs(expert_parallel_size=0)


class TestKernelInference:
    """Plugin auto-composes with user's chosen local kernel."""

    def _infer(self, **cfg_kwargs):
        from types import SimpleNamespace

        return ExpertParallelPlugin._infer_local_kernel(SimpleNamespace(**cfg_kwargs))

    def test_use_scattermoe_picks_scattermoe(self):
        assert self._infer(use_scattermoe=True) == "scattermoe"

    def test_experts_implementation_scattermoe_alone_does_NOT_pick_scattermoe(self):
        # use_scattermoe is the source of truth; bare experts_implementation=scattermoe
        # without the master flag falls through to the default kernel.
        assert self._infer(experts_implementation="scattermoe") == "grouped_mm"

    def test_use_scattermoe_overrides_experts_implementation_eager(self):
        # If kernels validator hasn't run yet, use_scattermoe still wins.
        assert (
            self._infer(use_scattermoe=True, experts_implementation="eager")
            == "scattermoe"
        )

    def test_grouped_mm_picks_grouped_mm(self):
        assert self._infer(experts_implementation="grouped_mm") == "grouped_mm"

    def test_batched_mm_picks_grouped_mm(self):
        assert self._infer(experts_implementation="batched_mm") == "grouped_mm"

    def test_eager_picks_eager(self):
        assert self._infer(experts_implementation="eager") == "eager"

    def test_default_picks_grouped_mm(self):
        assert self._infer() == "grouped_mm"

    def test_use_sonicmoe_falls_back_with_warning(self, caplog):
        # SonicMoE isn't a generic kernel; falls back to grouped_mm.
        with caplog.at_level("WARNING"):
            assert self._infer(use_sonicmoe=True) == "grouped_mm"
        assert any("SonicMoE" in m for m in caplog.messages)


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #


class TestRegistration:
    def test_kernel_name_mapping(self):
        assert kernel_to_registered_name("eager") == "deep_ep"
        assert kernel_to_registered_name("grouped_mm") == "deep_ep_grouped_mm"
        assert kernel_to_registered_name("scattermoe") == "deep_ep_scattermoe"

    def test_register_all_idempotent(self):
        from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS

        register_all()
        register_all()  # should not error
        for name in REGISTRY:
            fn = ALL_EXPERTS_FUNCTIONS.get_interface(name, None)
            assert fn is not None, f"{name} not registered"

    def test_whitelist_patch_accepts_deep_ep_names(self):
        from transformers.modeling_utils import PreTrainedModel

        register_all()
        m = PreTrainedModel.__new__(PreTrainedModel)

        class _Cfg:
            _experts_implementation = "deep_ep_grouped_mm"

        m.config = _Cfg()
        for name in REGISTRY:
            assert PreTrainedModel.get_correct_experts_implementation(m, name) == name

    def test_whitelist_patch_rejects_garbage(self):
        from transformers.modeling_utils import PreTrainedModel

        register_all()
        m = PreTrainedModel.__new__(PreTrainedModel)
        m.config = type("_C", (), {"_experts_implementation": "garbage"})()
        with pytest.raises(ValueError):
            PreTrainedModel.get_correct_experts_implementation(m, "garbage")


# --------------------------------------------------------------------------- #
# Module detection
# --------------------------------------------------------------------------- #


class TestExpertModuleDetection:
    def test_detects_qwen3moe_experts(self):
        block = _build_qwen3moe_block()
        found = list(_detect_experts_modules(block))
        assert len(found) == 1
        name, module = found[0]
        assert module is block.experts
        assert module.gate_up_proj.dim() == 3
        assert module.down_proj.dim() == 3

    def test_skips_non_3d_modules(self):
        # A regular linear layer should not be detected as Experts
        m = torch.nn.Linear(8, 8)
        m.gate_up_proj = m.weight  # 2D, doesn't match
        m.down_proj = m.weight
        found = list(_detect_experts_modules(m))
        assert len(found) == 0


# --------------------------------------------------------------------------- #
# Sharding (single-rank == no-op)
# --------------------------------------------------------------------------- #


class TestShardingSingleRank:
    """At world_size=1, sharding is a no-op."""

    def setup_method(self):
        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29555")
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            dist.init_process_group(backend="gloo", rank=0, world_size=1)

    def test_no_op_at_world_size_1(self):
        block = _build_qwen3moe_block(num_experts=16)
        original_shape = tuple(block.experts.gate_up_proj.shape)
        n = shard_expert_weights(block, dist.group.WORLD)
        assert n == 0
        assert tuple(block.experts.gate_up_proj.shape) == original_shape

    def test_none_group_no_op(self):
        block = _build_qwen3moe_block(num_experts=16)
        n = shard_expert_weights(block, None)
        assert n == 0


# --------------------------------------------------------------------------- #
# Plugin lifecycle (no DeepEP needed for these)
# --------------------------------------------------------------------------- #


class TestPluginLifecycle:
    def test_get_input_args(self):
        plugin = ExpertParallelPlugin()
        assert (
            plugin.get_input_args()
            == "axolotl.integrations.expert_parallel.ExpertParallelArgs"
        )

    def test_pre_model_load_disabled_is_noop(self):
        plugin = ExpertParallelPlugin()

        class _Cfg:
            expert_parallel_enabled = False

        plugin.pre_model_load(_Cfg())  # should not raise / not register anything new

    def _ep_cfg(self, **kw):
        from types import SimpleNamespace

        defaults = dict(
            expert_parallel_size=2,
            expert_parallel_fallback_on_unsupported=True,
            experts_implementation=None,
        )
        defaults.update(kw)
        return SimpleNamespace(**defaults)

    @pytest.mark.skipif(find_spec("deep_ep") is None, reason="deep_ep not installed")
    def test_pre_model_load_default_picks_grouped_mm(self):
        cfg = self._ep_cfg()
        ExpertParallelPlugin().pre_model_load(cfg)
        assert cfg.experts_implementation == "deep_ep_grouped_mm"

    @pytest.mark.skipif(find_spec("deep_ep") is None, reason="deep_ep not installed")
    def test_pre_model_load_use_scattermoe_auto_composes(self):
        cfg = self._ep_cfg(use_scattermoe=True, experts_implementation="scattermoe")
        ExpertParallelPlugin().pre_model_load(cfg)
        assert cfg.experts_implementation == "deep_ep_scattermoe"

    @pytest.mark.skipif(find_spec("deep_ep") is None, reason="deep_ep not installed")
    def test_pre_model_load_overrides_existing_eager(self):
        cfg = self._ep_cfg(experts_implementation="eager")
        ExpertParallelPlugin().pre_model_load(cfg)
        assert cfg.experts_implementation == "deep_ep"

    def test_disabled_by_default(self):
        """expert_parallel_size=1 (default) means EP is off — pre_model_load no-ops."""
        cfg = self._ep_cfg(expert_parallel_size=1)
        ExpertParallelPlugin().pre_model_load(cfg)
        assert cfg.experts_implementation is None

    def test_pre_model_load_enabled_no_deep_ep_falls_back(self, monkeypatch):
        import axolotl.integrations.expert_parallel.plugin as plugin_mod

        monkeypatch.setattr(plugin_mod, "find_spec", lambda name: None)
        cfg = self._ep_cfg()
        ExpertParallelPlugin().pre_model_load(cfg)
        assert cfg.experts_implementation is None

    def test_pre_model_load_no_fallback_raises(self, monkeypatch):
        import axolotl.integrations.expert_parallel.plugin as plugin_mod

        monkeypatch.setattr(plugin_mod, "find_spec", lambda name: None)
        cfg = self._ep_cfg(expert_parallel_fallback_on_unsupported=False)
        with pytest.raises(ImportError):
            ExpertParallelPlugin().pre_model_load(cfg)


# --------------------------------------------------------------------------- #
# Mesh-axis topology — rank assignments for EP+FSDP composition.
#
# The 4-rank case (ep=2 × dp_shard=2) is what we'll actually run on a 4× A100
# box. Tested here in single-process by spawning gloo workers; no GPUs needed.
# --------------------------------------------------------------------------- #


def _ep_topology_worker(rank, world_size, ep_size, dp_shard_size, port, q):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    try:
        from types import SimpleNamespace

        cfg = SimpleNamespace(
            expert_parallel_size=ep_size,
            dp_shard_size=dp_shard_size,
            tensor_parallel_size=1,
            context_parallel_size=1,
        )
        ep_group = ExpertParallelPlugin._resolve_ep_group(cfg)
        ep_ranks = sorted(dist.get_process_group_ranks(ep_group))
        # Also peek at the dp_shard slice if a 2D mesh was built.
        mesh = ExpertParallelPlugin._device_mesh
        dp_ranks = (
            sorted(dist.get_process_group_ranks(mesh["dp_shard"].get_group()))
            if mesh is not None and "dp_shard" in mesh.mesh_dim_names
            else None
        )
        q.put((rank, ep_ranks, dp_ranks))
    finally:
        dist.destroy_process_group()
        ExpertParallelPlugin._device_mesh = None


def _ep_topology_worker_expects_error(
    rank, world_size, ep_size, dp_shard_size, port, q
):
    """Variant that captures any ValueError from `_resolve_ep_group`.

    Module-level so `mp.get_context("spawn")` can pickle it.
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        from types import SimpleNamespace

        cfg = SimpleNamespace(
            expert_parallel_size=ep_size,
            dp_shard_size=dp_shard_size,
            tensor_parallel_size=1,
            context_parallel_size=1,
        )
        err = None
        try:
            ExpertParallelPlugin._resolve_ep_group(cfg)
        except ValueError as e:
            err = str(e)
        q.put((rank, err))
    finally:
        dist.destroy_process_group()
        ExpertParallelPlugin._device_mesh = None


def _spawn_topology_check(world_size, ep_size, dp_shard_size, port_base):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [
        ctx.Process(
            target=_ep_topology_worker,
            args=(r, world_size, ep_size, dp_shard_size, port_base, q),
        )
        for r in range(world_size)
    ]
    for p in procs:
        p.start()
    results = [q.get(timeout=120) for _ in range(world_size)]
    for p in procs:
        p.join(timeout=20)
        assert p.exitcode == 0, f"worker exited with {p.exitcode}"
    return sorted(results, key=lambda r: r[0])


class TestMeshTopology:
    """The 4-rank EP+FSDP composition rank assignments."""

    def test_world4_ep2_dp2_orthogonal(self):
        """At world=4 with ep=2 and dp_shard=2, EP groups must be strided
        ({0,2}, {1,3}) and dp_shard groups contiguous ({0,1}, {2,3}).
        """
        # Use large random-ish port base to avoid collision with anything else.
        results = _spawn_topology_check(
            world_size=4, ep_size=2, dp_shard_size=2, port_base=37610
        )
        # Build per-rank groupings from results.
        ep_groups_by_rank = {r: tuple(eps) for r, eps, _ in results}
        dp_groups_by_rank = {r: tuple(dps) for r, _, dps in results}

        # EP groups (strided): {0,2} and {1,3}
        assert ep_groups_by_rank[0] == (0, 2), ep_groups_by_rank
        assert ep_groups_by_rank[1] == (1, 3), ep_groups_by_rank
        assert ep_groups_by_rank[2] == (0, 2), ep_groups_by_rank
        assert ep_groups_by_rank[3] == (1, 3), ep_groups_by_rank

        # dp_shard groups (contiguous, matches accelerate): {0,1} and {2,3}
        assert dp_groups_by_rank[0] == (0, 1), dp_groups_by_rank
        assert dp_groups_by_rank[1] == (0, 1), dp_groups_by_rank
        assert dp_groups_by_rank[2] == (2, 3), dp_groups_by_rank
        assert dp_groups_by_rank[3] == (2, 3), dp_groups_by_rank

    def test_world4_ep4_dp1_uses_world(self):
        """ep_size == world_size short-circuits to dist.group.WORLD."""
        results = _spawn_topology_check(
            world_size=4, ep_size=4, dp_shard_size=1, port_base=37710
        )
        for rank, ep_ranks, dp_ranks in results:
            assert ep_ranks == [0, 1, 2, 3], (rank, ep_ranks)
            assert dp_ranks is None  # no 2D mesh built

    def test_world4_ep2_dp1_invalid_product_raises(self):
        """ep<world without dp_shard filling the rest must raise (product mismatch)."""
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        procs = [
            ctx.Process(
                target=_ep_topology_worker_expects_error,
                args=(r, 4, 2, 1, 37810, q),
            )
            for r in range(4)
        ]
        for p in procs:
            p.start()
        results = [q.get(timeout=60) for _ in range(4)]
        for p in procs:
            p.join(timeout=10)
        for rank, err in results:
            assert err is not None, (
                f"rank {rank} did not raise; expected product mismatch"
            )
            assert "must equal" in err.lower() or "world_size" in err.lower(), err

    def test_mesh_axis_product_mismatch_raises(self):
        """world=4 with ep=2*dp=4 (product 8 != 4) raises clearly."""
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        procs = [
            ctx.Process(
                target=_ep_topology_worker_expects_error,
                args=(r, 4, 2, 4, 37910, q),
            )
            for r in range(4)
        ]
        for p in procs:
            p.start()
        results = [q.get(timeout=60) for _ in range(4)]
        for p in procs:
            p.join(timeout=10)
        for rank, err in results:
            assert err is not None, f"rank {rank} did not raise"
            assert "must equal" in err.lower() or "world_size" in err.lower(), err
