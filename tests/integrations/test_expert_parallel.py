"""Tests for the Expert-Parallel (DeepEP) integration."""

import os
import queue as queue_mod
import socket
import time
from datetime import timedelta
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
from axolotl.integrations.expert_parallel.plugin import expert_shard_axis
from axolotl.integrations.expert_parallel.shard import (
    _detect_experts_modules,
    _slice_expert_lora_param,
    ep_adapter_load_local_shard,
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

    def test_use_sonicmoe_picks_sonicmoe(self):
        assert self._infer(use_sonicmoe=True) == "sonicmoe"


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #


class TestRegistration:
    def test_kernel_name_mapping(self):
        assert kernel_to_registered_name("eager") == "deep_ep"
        assert kernel_to_registered_name("grouped_mm") == "deep_ep_grouped_mm"
        assert kernel_to_registered_name("scattermoe") == "deep_ep_scattermoe"
        assert kernel_to_registered_name("sonicmoe") == "deep_ep_sonicmoe"

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

    def teardown_method(self):
        if dist.is_initialized():
            dist.destroy_process_group()

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
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=120),
    )

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
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=120),
    )
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


def _ep_cp_topology_worker(rank, world_size, ep_size, cp_size, dp_shard_size, port, q):
    """EP × CP (× dp_shard) topology probe: returns this rank's ep / cp / dp_shard group members."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(
        backend="gloo", rank=rank, world_size=world_size, timeout=timedelta(seconds=120)
    )
    try:
        from types import SimpleNamespace

        cfg = SimpleNamespace(
            expert_parallel_size=ep_size,
            dp_shard_size=dp_shard_size,
            tensor_parallel_size=1,
            context_parallel_size=cp_size,
        )
        ep_group = ExpertParallelPlugin._resolve_ep_group(cfg)
        ep_ranks = sorted(dist.get_process_group_ranks(ep_group))
        cp_group = ExpertParallelPlugin._resolve_cp_group(cfg)
        cp_ranks = (
            sorted(dist.get_process_group_ranks(cp_group))
            if cp_group is not None
            else None
        )
        mesh = ExpertParallelPlugin._device_mesh
        dp_ranks = (
            sorted(dist.get_process_group_ranks(mesh["dp_shard"].get_group()))
            if mesh is not None and "dp_shard" in (mesh.mesh_dim_names or ())
            else None
        )
        q.put((rank, ep_ranks, cp_ranks, dp_ranks))
    finally:
        dist.destroy_process_group()
        ExpertParallelPlugin._device_mesh = None
        ExpertParallelPlugin._device_mesh = None


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _collect_worker_results(procs, q, world_size, timeout=120):
    """Collect one result per worker, bailing out early if a worker dies.

    A bare ``q.get(timeout=...)`` blocks the full timeout even after a worker has
    crashed without reporting; here we stop once all workers have exited so an
    infra failure surfaces as a clear assertion instead of a silent hang.
    """
    results = []
    deadline = time.monotonic() + timeout
    while len(results) < world_size and time.monotonic() < deadline:
        try:
            results.append(q.get(timeout=5))
        except queue_mod.Empty:
            if all(not p.is_alive() for p in procs):
                break
    for p in procs:
        p.join(timeout=20)
    exitcodes = [p.exitcode for p in procs]
    assert len(results) == world_size, (
        f"only {len(results)}/{world_size} workers reported; exitcodes={exitcodes}"
    )
    assert all(code == 0 for code in exitcodes), f"worker exitcodes={exitcodes}"
    return results


def _spawn_topology_check(world_size, ep_size, dp_shard_size):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = _find_free_port()
    procs = [
        ctx.Process(
            target=_ep_topology_worker,
            args=(r, world_size, ep_size, dp_shard_size, port, q),
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


def _spawn_ep_cp_check(world_size, ep_size, cp_size, dp_shard_size=1):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = _find_free_port()
    procs = [
        ctx.Process(
            target=_ep_cp_topology_worker,
            args=(r, world_size, ep_size, cp_size, dp_shard_size, port, q),
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

    def test_world4_ep2_cp2_orthogonal(self):
        """world=4, ep=2 × cp=2: experts shard on `ep`, sequence on `cp`. The two axes must be
        orthogonal — every rank is in exactly one ep group and one cp group that intersect only at
        that rank, and the groups tile the 4 ranks. (DeepEP not required — pure group topology.)"""
        results = _spawn_ep_cp_check(world_size=4, ep_size=2, cp_size=2)
        ep = {r: tuple(e) for r, e, c, d in results}
        cp = {r: tuple(c) for r, e, c, d in results}
        for r in range(4):
            assert len(ep[r]) == 2 and len(cp[r]) == 2, (ep, cp)
            assert set(ep[r]) & set(cp[r]) == {r}, (
                r,
                ep,
                cp,
            )  # orthogonal: meet only at self
        assert len(set(ep.values())) == 2 and len(set(cp.values())) == 2, (ep, cp)
        # union of all ep groups (and cp groups) tiles the world
        assert set().union(*ep.values()) == {0, 1, 2, 3}
        assert set().union(*cp.values()) == {0, 1, 2, 3}

    def test_world4_ep2_dp2_orthogonal(self):
        """At world=4 with ep=2 and dp_shard=2, EP groups must be strided
        ({0,2}, {1,3}) and dp_shard groups contiguous ({0,1}, {2,3}).
        """
        results = _spawn_topology_check(world_size=4, ep_size=2, dp_shard_size=2)
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
        results = _spawn_topology_check(world_size=4, ep_size=4, dp_shard_size=1)
        for rank, ep_ranks, dp_ranks in results:
            assert ep_ranks == [0, 1, 2, 3], (rank, ep_ranks)
            assert dp_ranks is None  # no 2D mesh built

    def _spawn_expects_error(self, ep_size, dp_shard_size, world_size=4):
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        port = _find_free_port()
        procs = [
            ctx.Process(
                target=_ep_topology_worker_expects_error,
                args=(r, world_size, ep_size, dp_shard_size, port, q),
            )
            for r in range(world_size)
        ]
        for p in procs:
            p.start()
        return _collect_worker_results(procs, q, world_size)

    def test_world4_ep2_dp1_invalid_product_raises(self):
        """ep<world without dp_shard filling the rest must raise (product mismatch)."""
        results = self._spawn_expects_error(ep_size=2, dp_shard_size=1)
        for rank, err in results:
            assert err is not None, (
                f"rank {rank} did not raise; expected product mismatch"
            )
            assert "must equal" in err.lower() or "world_size" in err.lower(), err

    def test_mesh_axis_product_mismatch_raises(self):
        """world=4 with ep=2*dp=4 (product 8 != 4) raises clearly."""
        results = self._spawn_expects_error(ep_size=2, dp_shard_size=4)
        for rank, err in results:
            assert err is not None, f"rank {rank} did not raise"
            assert "must equal" in err.lower() or "world_size" in err.lower(), err


class TestExpertLoraSlicing:
    """Routed-expert LoRA must be sliced so each EP rank gets ITS OWN experts' adapter blocks, and the
    sliced shape must imply the TRUE LoRA rank ``r`` (= sliced_packed_dim // E_local), not ``r*ep_size``.

    The pure-EP bug left the adapter global-sized while the base was E_local, so
    ``_unwrap_experts_lora`` read ``rank = lora_A.shape[0] // E_local = r*ep_size`` and reshaped the
    adapter into a scrambled expert/rank layout (harmless only at step 0, where lora_B == 0). These are
    pure tensor-op checks of the slice math — no dist / FSDP / DeepEP / NVFP4 / model needed.
    """

    @pytest.mark.parametrize(
        "e_global,ep_size,r", [(8, 2, 2), (256, 8, 16), (16, 4, 3)]
    )
    def test_lora_A_slice_picks_local_experts(self, e_global, ep_size, r):
        # lora_A peft layout [E*r, in], expert-major: expert e owns rows [e*r:(e+1)*r]; tag rows = e.
        e_local, in_features = e_global // ep_size, 5
        full = torch.zeros(e_global * r, in_features)
        for e in range(e_global):
            full[e * r : (e + 1) * r] = float(e)
        for ep_rank in range(ep_size):
            start, end = ep_rank * e_local, (ep_rank + 1) * e_local
            lin = torch.nn.Linear(in_features, e_global * r, bias=False)
            lin.weight = torch.nn.Parameter(full.clone())
            assert _slice_expert_lora_param(lin, 0, e_global, start, end) == r
            sl = lin.weight.data
            # the implied rank from the SLICED shape is the true r (the bug would give r*ep_size)
            assert sl.shape[0] // e_local == r
            tags = sl.reshape(e_local, r, in_features)[:, 0, 0].long()
            assert torch.equal(tags, torch.arange(start, end))

    @pytest.mark.parametrize(
        "e_global,ep_size,r", [(8, 2, 2), (256, 8, 16), (16, 4, 3)]
    )
    def test_lora_B_slice_picks_local_experts(self, e_global, ep_size, r):
        # lora_B peft layout [out, r*E], rank-major [out, r, E]; tag column (k, e) = e.
        e_local, out_features = e_global // ep_size, 5
        full = torch.zeros(out_features, r, e_global)
        for e in range(e_global):
            full[:, :, e] = float(e)
        full = full.reshape(out_features, r * e_global)
        for ep_rank in range(ep_size):
            start, end = ep_rank * e_local, (ep_rank + 1) * e_local
            lin = torch.nn.Linear(e_global * r, out_features, bias=False)
            lin.weight = torch.nn.Parameter(full.clone())
            assert _slice_expert_lora_param(lin, 1, e_global, start, end) == r
            sl = lin.weight.data
            assert sl.shape[1] // e_local == r
            tags = sl.reshape(out_features, r, e_local)[0, 0, :].long()
            assert torch.equal(tags, torch.arange(start, end))

    @pytest.mark.parametrize(
        "e_global,ep_size,r", [(8, 2, 2), (256, 8, 16), (16, 4, 3)]
    )
    def test_forward_slice_picks_local_experts_and_true_rank(
        self, e_global, ep_size, r
    ):
        """The FORWARD-time slice (used by the ParamWrapper fastpath and the _unwrap fallback) keeps
        the adapter global and takes each rank's expert block at use time, reporting rank=r — NOT the
        r*ep_size that the un-sliced global adapter would imply (the actual pure-EP bug path)."""
        from types import SimpleNamespace

        from axolotl.integrations.kernels.libs.scattermoe_lora.experts import (
            _ep_local_expert_lora,
        )

        e_local, in_f, out_f = e_global // ep_size, 5, 7
        A = torch.zeros(e_global * r, in_f)
        for e in range(e_global):
            A[e * r : (e + 1) * r] = float(e)
        B = torch.zeros(out_f, r, e_global)
        for e in range(e_global):
            B[:, :, e] = float(e)
        B = B.reshape(out_f, r * e_global)
        for ep_rank in range(ep_size):
            offset = ep_rank * e_local
            experts = SimpleNamespace(
                num_experts=e_local,
                num_experts_global=e_global,
                local_expert_offset=offset,
            )
            a, b, n_local, rank = _ep_local_expert_lora(A, B, experts)
            assert n_local == e_local and rank == r
            assert torch.equal(
                a.reshape(e_local, r, in_f)[:, 0, 0].long(),
                torch.arange(offset, offset + e_local),
            )
            assert torch.equal(
                b.reshape(out_f, r, e_local)[0, 0, :].long(),
                torch.arange(offset, offset + e_local),
            )

    def test_forward_slice_noop_when_not_ep_sharded(self):
        from types import SimpleNamespace

        from axolotl.integrations.kernels.libs.scattermoe_lora.experts import (
            _ep_local_expert_lora,
        )

        a_in, b_in = torch.randn(8 * 2, 5), torch.randn(7, 2 * 8)
        experts = SimpleNamespace(
            num_experts=8, num_experts_global=8, local_expert_offset=0
        )
        a, b, n_local, rank = _ep_local_expert_lora(a_in, b_in, experts)
        assert n_local == 8 and rank == 2
        assert a is a_in and b is b_in  # no slice / copy when E_local == E_global

    def test_ranks_reconstruct_global_adapter(self):
        # All EP ranks' slices, concatenated on the expert axis, must rebuild the global adapter
        # (no expert dropped or duplicated).
        e_global, ep_size, r, in_features, out_features = 8, 2, 2, 5, 7
        e_local = e_global // ep_size
        A = torch.randn(e_global * r, in_features)
        B = torch.randn(out_features, r * e_global)
        a_pieces, b_pieces = [], []
        for ep_rank in range(ep_size):
            start, end = ep_rank * e_local, (ep_rank + 1) * e_local
            la = torch.nn.Linear(in_features, e_global * r, bias=False)
            la.weight = torch.nn.Parameter(A.clone())
            _slice_expert_lora_param(la, 0, e_global, start, end)
            a_pieces.append(la.weight.data)
            lb = torch.nn.Linear(e_global * r, out_features, bias=False)
            lb.weight = torch.nn.Parameter(B.clone())
            _slice_expert_lora_param(lb, 1, e_global, start, end)
            # rank-major [out, r, E_local] per rank -> stack on the expert axis
            b_pieces.append(lb.weight.data.reshape(out_features, r, e_local))
        assert torch.equal(torch.cat(a_pieces, dim=0), A)
        assert torch.equal(
            torch.cat(b_pieces, dim=2).reshape(out_features, r * e_global),
            B.reshape(out_features, r, e_global).reshape(out_features, r * e_global),
        )


class TestExpertAdapterLoadSharding:
    """The cpu_ram_efficient load reconstructs each rank's local FSDP shard of the routed-expert LoRA
    adapter from rank-0's broadcast GLOBAL (all-experts) adapter — the inverse of shard_expert_lora +
    the dp/cp FSDP sharding (``ep_adapter_load_local_shard``). The ep slice must be EXPERT-aware:
    lora_B's experts are the last axis of its ``[out, r, E]`` view (NOT contiguous in the flat ``r*E``
    dim), so a plain chunk on dim 1 would load a rank-component instead of this ep-group's experts.
    The 1-step e2e loss can't catch this (lora_B is zero-initialized — wrong zeros are still zeros); it
    only bites a non-zero / resumed adapter. Pure tensor-op checks — no dist / FSDP / model needed.
    """

    @pytest.mark.parametrize(
        "e_global,ep_size,dp_size,r", [(8, 2, 2, 2), (256, 4, 2, 16), (16, 2, 1, 3)]
    )
    def test_lora_A_load_shards_match_forward_slice(
        self, e_global, ep_size, dp_size, r
    ):
        from torch.distributed.tensor import Shard

        in_features, e_local = 5, e_global // ep_size
        # global lora_A [E_global*r, in], expert-major; expert e's row block tagged = e
        g = torch.zeros(e_global * r, in_features)
        for e in range(e_global):
            g[e * r : (e + 1) * r] = float(e)
        placements = (Shard(0),)  # FSDP shards dim 0
        for ep_coord in range(ep_size):
            shards = [
                ep_adapter_load_local_shard(
                    g, 0, e_global, ep_coord, ep_size, placements, dp_size, dp
                )
                for dp in range(dp_size)
            ]
            ep_full = torch.cat(shards, dim=0)  # gather this ep-group's dp shards
            # == the forward E_local slice shard_expert_lora produces for this ep-group
            start, end = ep_coord * e_local, (ep_coord + 1) * e_local
            la = torch.nn.Linear(in_features, e_global * r, bias=False)
            la.weight = torch.nn.Parameter(g.clone())
            _slice_expert_lora_param(la, 0, e_global, start, end)
            assert torch.equal(ep_full, la.weight.data)
            tags = ep_full.reshape(e_local, r, in_features)[:, 0, 0].long()
            assert torch.equal(tags, torch.arange(start, end))

    @pytest.mark.parametrize(
        "e_global,ep_size,dp_size,r", [(8, 2, 2, 2), (256, 4, 2, 16), (16, 2, 1, 3)]
    )
    def test_lora_B_load_shards_match_forward_slice(
        self, e_global, ep_size, dp_size, r
    ):
        from torch.distributed.tensor import Shard

        out_features, e_local = 6, e_global // ep_size
        # global lora_B [out, r*E] viewed [out, r, E]; column (k, e) tagged = e
        g = torch.zeros(out_features, r, e_global)
        for e in range(e_global):
            g[:, :, e] = float(e)
        g = g.reshape(out_features, r * e_global)
        placements = (
            Shard(0),
        )  # FSDP shards dim 0 (out); the ep slice is on the expert axis
        for ep_coord in range(ep_size):
            shards = [
                ep_adapter_load_local_shard(
                    g, 1, e_global, ep_coord, ep_size, placements, dp_size, dp
                )
                for dp in range(dp_size)
            ]
            ep_full = torch.cat(shards, dim=0)  # gather the dp (out) shards back
            start, end = ep_coord * e_local, (ep_coord + 1) * e_local
            lb = torch.nn.Linear(e_global * r, out_features, bias=False)
            lb.weight = torch.nn.Parameter(g.clone())
            _slice_expert_lora_param(lb, 1, e_global, start, end)
            assert torch.equal(ep_full, lb.weight.data)
            tags = ep_full.reshape(out_features, r, e_local)[0, 0, :].long()
            assert torch.equal(tags, torch.arange(start, end))

    def test_lora_B_plain_chunk_would_load_wrong_experts(self):
        # Documents the bug the expert-aware slice fixes: a naive chunk(ep_size, dim=1) loads a
        # rank-component (all experts of one r-slice), not this ep-group's experts.
        from torch.distributed.tensor import Shard

        e_global, ep_size, r, out = 8, 2, 2, 6
        g = torch.zeros(out, r, e_global)
        for e in range(e_global):
            g[:, :, e] = float(e)
        g = g.reshape(out, r * e_global)
        correct = ep_adapter_load_local_shard(
            g, 1, e_global, 1, ep_size, (Shard(0),), 1, 0
        )
        naive = g.chunk(ep_size, dim=1)[1]
        assert not torch.equal(correct, naive)
        e_local = e_global // ep_size
        # correct loads ep-group 1's experts {4,5,6,7}; the naive chunk would not
        got = set(correct.reshape(out, r, e_local)[0, 0, :].long().tolist())
        assert got == {4, 5, 6, 7}


class TestExpertShardAxis:
    """`expert_shard_axis` picks the non-ep mesh axis the routed experts FSDP-shard on under EP
    composition: dp_shard when present, else cp, else None (pure EP / no ep axis). Pure logic on the
    mesh dim names — no dist / mesh object needed."""

    @pytest.mark.parametrize(
        "dim_names,expected",
        [
            (("ep", "dp_shard"), "dp_shard"),  # EP × dp_shard
            (("ep", "cp"), "cp"),  # EP × cp
            (("ep", "dp_shard", "cp"), "dp_shard"),  # both -> dp_shard preferred
            (("ep",), None),  # pure EP at world_size: no secondary axis to pre-wrap on
            (("dp_shard", "cp"), None),  # no ep axis -> not an EP composition
            (("ep", "tp"), None),  # tp is not an expert-shard axis (EP×TP unsupported)
            ((), None),
            (None, None),
        ],
    )
    def test_axis_selection(self, dim_names, expected):
        assert expert_shard_axis(dim_names) == expected

    def test_dp_shard_preferred_over_cp_regardless_of_order(self):
        # order in the tuple must not change the dp_shard preference
        assert expert_shard_axis(("ep", "cp", "dp_shard")) == "dp_shard"
        assert expert_shard_axis(("ep", "dp_shard", "cp")) == "dp_shard"


class TestEpLoraSaveGating:
    """save_ep_lora_adapter must EP-gather the routed-expert adapter ONLY when it was physically
    sliced to E_local (EP×dp_shard/cp composition, where shard_expert_lora sets _ep_lora_sharded).
    Pure EP keeps the adapter global, so gathering would duplicate every expert ep_size times. This
    checks the discriminator the save relies on: shard_expert_lora flags + slices composition adapters,
    and a non-sharded wrapper carries no flag (so the save leaves it as-is)."""

    def _make_wrapper_model(self, e_global, ep_size, r, ep_rank=0):
        e_local = e_global // ep_size

        class _Experts(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.num_experts_global = e_global
                self.num_local_experts = e_local
                self.local_expert_offset = ep_rank * e_local

        class ParamWrapper(
            torch.nn.Module
        ):  # name matches _is_param_wrapper's fallback check
            def __init__(self):
                super().__init__()
                self.base_layer = _Experts()
                self.lora_A = torch.nn.ModuleDict(
                    {"default": torch.nn.Linear(5, e_global * r, bias=False)}
                )
                self.lora_B = torch.nn.ModuleDict(
                    {"default": torch.nn.Linear(e_global * r, 7, bias=False)}
                )

        class _Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.wrapper = ParamWrapper()

        return _Model()

    def test_composition_slice_sets_flag(self):
        from axolotl.integrations.expert_parallel.shard import shard_expert_lora

        e_global, ep_size, r = 8, 2, 2
        e_local = e_global // ep_size
        m = self._make_wrapper_model(e_global, ep_size, r)
        n = shard_expert_lora(m, ep_size)
        assert n == 2  # lora_A + lora_B sliced
        assert (
            m.wrapper._ep_lora_sharded is True
        )  # the gather discriminator the save reads
        # sliced to E_local: gather (ep_size copies) is what reconstructs E_global
        assert m.wrapper.lora_A["default"].weight.shape[0] == e_local * r
        assert m.wrapper.lora_B["default"].weight.shape[1] == r * e_local

    def test_pure_ep_wrapper_has_no_flag(self):
        # Pure EP never runs shard_expert_lora -> no _ep_lora_sharded -> save_ep_lora_adapter must NOT
        # EP-gather (the adapter is already global E_global).
        m = self._make_wrapper_model(8, 2, 2)
        assert getattr(m.wrapper, "_ep_lora_sharded", False) is False
        # ep_size == 1 is a no-op and never flags either
        from axolotl.integrations.expert_parallel.shard import shard_expert_lora

        assert shard_expert_lora(m, 1) == 0
        assert getattr(m.wrapper, "_ep_lora_sharded", False) is False
