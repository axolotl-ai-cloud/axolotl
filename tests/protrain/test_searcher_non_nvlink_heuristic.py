"""Unit tests for PR #17c — defensive searcher heuristic on non-NVLink rigs.

v71/v72-redux verification: bs=2 with ``n_offload > 0`` hangs deep in
autograd backward on consumer non-NVLink multi-GPU rigs; v62-style configs
(``n_persist=128, n_offload=0``) run cleanly end-to-end. Until the
underlying re-gather stream contention is fixed (PR #17b), steer the
searcher's tie-break toward ``n_offload=0`` when the rig is multi-rank
without NVLink.

These tests mock ``estimate_runtime`` so two synthetic candidates have
identical (or near-identical) predicted runtimes, then assert the
comparator's pick swings with the heuristic.
"""

from __future__ import annotations

from axolotl.integrations.protrain import search as search_pkg
from axolotl.integrations.protrain.search import exhaustive
from axolotl.integrations.protrain.search.exhaustive import search
from axolotl.integrations.protrain.types import (
    BlockId,
    ChunkId,
    ChunkLayout,
    HardwareProfile,
    OpId,
    OpRecord,
    ParamId,
    ProfilerTrace,
)

MB = 1 << 20
GB = 1 << 30


# ---------------------------------------------------------------------
# Synthetic fixtures (mirror test_searcher_kernel_aware.py)
# ---------------------------------------------------------------------


def _make_op_order(n_block: int, ops_per_block: int) -> tuple[OpRecord, ...]:
    out: list[OpRecord] = []
    op_id = 0
    for b in range(n_block):
        for k in range(ops_per_block):
            out.append(
                OpRecord(
                    op_id=OpId(op_id),
                    module_path=f"block.{b}.op.{k}",
                    qualified_name="aten::toy",
                    shape_signature=((1,),),
                    block_id=BlockId(b),
                    is_forward=True,
                )
            )
            op_id += 1
    return tuple(out)


def _make_trace(
    *,
    n_block: int = 4,
    ops_per_block: int = 3,
    activation_bytes_per_block: int = 16 * MB,
    model_state_bytes: int = 256 * MB,
    intra_delta_bytes: int = 4 * MB,
    inter_delta_bytes: int = 1 * MB,
    op_latency_s: float = 0.0001,
    world: int = 2,
) -> ProfilerTrace:
    op_order = _make_op_order(n_block, ops_per_block)
    intra_op_delta = {op.op_id: intra_delta_bytes for op in op_order}
    inter_op_delta = {op.op_id: inter_delta_bytes for op in op_order}
    activation_sizes = {BlockId(b): activation_bytes_per_block for b in range(n_block)}
    op_latencies = {op.op_id: op_latency_s for op in op_order}
    hooked_sum = sum(op_latencies.values())
    return ProfilerTrace(
        op_order=op_order,
        intra_op_delta=intra_op_delta,
        inter_op_delta=inter_op_delta,
        activation_sizes=activation_sizes,
        model_state_bytes=model_state_bytes,
        pcie_h2d_bps=12e9,
        pcie_d2h_bps=12e9,
        nccl_gather_s={} if world <= 1 else {64 * MB: 0.01},
        nccl_reduce_s={} if world <= 1 else {64 * MB: 0.012},
        arch_hash="non-nvlink-heuristic-test",
        bs=1,
        seq=128,
        sku="RTX 3090 (synthetic)",
        world=world,
        op_latencies=op_latencies,
        hooked_fwd_wall_s=hooked_sum,
        steady_fwd_wall_s=hooked_sum,
        steady_bwd_wall_s=0.0,
    )


def _make_layout(
    *, n_chunk: int = 6, s_chunk: int = 32 * MB, n_block: int = 4
) -> ChunkLayout:
    chunks = tuple((ParamId(f"param.{i}"),) for i in range(n_chunk))
    param_to_chunk = {ParamId(f"param.{i}"): ChunkId(i) for i in range(n_chunk)}
    block_to_chunks: dict[BlockId, tuple[ChunkId, ...]] = {
        BlockId(b): (ChunkId(b % n_chunk),) for b in range(n_block)
    }
    return ChunkLayout(
        S_chunk=s_chunk,
        N_chunk=n_chunk,
        chunks=chunks,
        param_to_chunk=param_to_chunk,
        block_to_chunks=block_to_chunks,
    )


def _make_hw(
    *,
    gpu_count: int = 2,
    has_nvlink: bool = False,
    gpu_memory_bytes: int = 4 * GB,
) -> HardwareProfile:
    return HardwareProfile(
        gpu_sku="RTX 3090 (synthetic)",
        gpu_memory_bytes=gpu_memory_bytes,
        gpu_count=gpu_count,
        pcie_h2d_bps=12e9,
        pcie_d2h_bps=12e9,
        has_nvlink=has_nvlink,
        zero3_shard=False,
        cpu_adam_bytes_per_sec=2e9,
        gpu_adam_bytes_per_sec=4e11,
    )


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _patch_runtime_by_n_offload(monkeypatch, base_s: float = 1.0) -> None:
    """Make ``estimate_runtime`` deterministic: identical wall regardless of n_offload.

    The cost-model output is collapsed to a single value so the comparator
    sees a perfect tie between every candidate; the heuristic's tie-break
    must then drive the pick.
    """

    def _fake(cfg, trace, layout, block_map, hw, *, chunk_bw_table=None):
        return base_s

    monkeypatch.setattr(exhaustive, "estimate_runtime", _fake)


def _patch_runtime_slight_offload_advantage(
    monkeypatch, *, base_s: float = 1.0, delta_ratio: float = 0.02
) -> None:
    """``estimate_runtime`` favours n_offload>0 by ``delta_ratio`` (e.g. 2%).

    With the heuristic ON (5% noise band), the n_offload=0 candidate must
    still win on the tie-break. With the heuristic OFF, the slightly-faster
    n_offload>0 candidate must win on pure runtime.
    """

    def _fake(cfg, trace, layout, block_map, hw, *, chunk_bw_table=None):
        if cfg.n_offload > 0:
            return base_s * (1.0 - delta_ratio)
        return base_s

    monkeypatch.setattr(exhaustive, "estimate_runtime", _fake)


# ---------------------------------------------------------------------
# Public-API surface
# ---------------------------------------------------------------------


def test_search_exposes_prefer_no_offload_kwarg():
    """``prefer_no_offload_on_non_nvlink`` is keyword-only and defaults to True."""
    import inspect

    sig = inspect.signature(search)
    param = sig.parameters["prefer_no_offload_on_non_nvlink"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is True
    nv_param = sig.parameters["non_nvlink_multi_rank"]
    assert nv_param.kind is inspect.Parameter.KEYWORD_ONLY
    assert nv_param.default is None


def test_search_pkg_reexports_search():
    """Smoke: ``search`` reachable via the package import path tests use."""
    assert search_pkg.search is search


# ---------------------------------------------------------------------
# Heuristic ON: non-NVLink multi-rank prefers n_offload=0
# ---------------------------------------------------------------------


def test_heuristic_on_non_nvlink_prefers_n_offload_zero(monkeypatch):
    """All-equal runtimes + non-NVLink multi-rank → picked cfg has n_offload=0."""
    _patch_runtime_by_n_offload(monkeypatch)
    trace = _make_trace(world=2)
    layout = _make_layout()
    hw = _make_hw(gpu_count=2, has_nvlink=False)
    result = search(
        trace,
        layout,
        capacity_bytes=2 * GB,
        hw=hw,
        prefer_no_offload_on_non_nvlink=True,
    )
    assert result.cfg.n_offload == 0, (
        "non-NVLink multi-rank heuristic must pick n_offload=0 when all "
        f"runtimes tie; got n_offload={result.cfg.n_offload}"
    )


def test_heuristic_on_within_noise_band_prefers_n_offload_zero(monkeypatch):
    """2% offload advantage < 5% noise band → heuristic still picks n_offload=0."""
    _patch_runtime_slight_offload_advantage(monkeypatch, delta_ratio=0.02)
    trace = _make_trace(world=2)
    layout = _make_layout()
    hw = _make_hw(gpu_count=2, has_nvlink=False)
    result = search(
        trace,
        layout,
        capacity_bytes=2 * GB,
        hw=hw,
        prefer_no_offload_on_non_nvlink=True,
    )
    assert result.cfg.n_offload == 0, (
        "non-NVLink heuristic must override <5% offload advantage; got "
        f"n_offload={result.cfg.n_offload}"
    )


# ---------------------------------------------------------------------
# Heuristic OFF: original tie-break preserved
# ---------------------------------------------------------------------


def test_heuristic_off_preserves_original_tiebreak(monkeypatch):
    """``prefer_no_offload_on_non_nvlink=False`` → faster n_offload>0 wins on runtime."""
    _patch_runtime_slight_offload_advantage(monkeypatch, delta_ratio=0.02)
    trace = _make_trace(world=2)
    layout = _make_layout()
    hw = _make_hw(gpu_count=2, has_nvlink=False)
    result_off = search(
        trace,
        layout,
        capacity_bytes=2 * GB,
        hw=hw,
        prefer_no_offload_on_non_nvlink=False,
    )
    # Without the heuristic, the 2% faster n_offload>0 candidate is still
    # within the 1% noise floor for the std comparator? No — 2% > 1%, so
    # std comparator strictly prefers the n_offload>0 candidate.
    assert result_off.cfg.n_offload > 0, (
        "with heuristic OFF the 2%-faster n_offload>0 candidate must win; "
        f"got n_offload={result_off.cfg.n_offload}"
    )


# ---------------------------------------------------------------------
# NVLink rigs: heuristic auto-disabled
# ---------------------------------------------------------------------


def test_nvlink_multi_rank_disables_heuristic_auto(monkeypatch):
    """``has_nvlink=True`` + multi-rank → heuristic does not fire (default ON)."""
    _patch_runtime_slight_offload_advantage(monkeypatch, delta_ratio=0.02)
    trace = _make_trace(world=2)
    layout = _make_layout()
    hw = _make_hw(gpu_count=2, has_nvlink=True)
    result = search(
        trace,
        layout,
        capacity_bytes=2 * GB,
        hw=hw,
        prefer_no_offload_on_non_nvlink=True,
    )
    # NVLink: heuristic skipped, 2% faster n_offload>0 wins on the original comparator.
    assert result.cfg.n_offload > 0, (
        "NVLink rig must auto-disable the heuristic; got "
        f"n_offload={result.cfg.n_offload}"
    )


def test_single_rank_disables_heuristic_auto(monkeypatch):
    """``gpu_count=1`` → heuristic auto-disabled (no inter-rank traffic exists)."""
    _patch_runtime_slight_offload_advantage(monkeypatch, delta_ratio=0.02)
    trace = _make_trace(world=1)
    layout = _make_layout()
    hw = _make_hw(gpu_count=1, has_nvlink=False)
    result = search(
        trace,
        layout,
        capacity_bytes=2 * GB,
        hw=hw,
        prefer_no_offload_on_non_nvlink=True,
    )
    assert result.cfg.n_offload > 0, (
        "single-rank must auto-disable the heuristic; got "
        f"n_offload={result.cfg.n_offload}"
    )


# ---------------------------------------------------------------------
# Explicit non_nvlink_multi_rank override
# ---------------------------------------------------------------------


def test_non_nvlink_multi_rank_explicit_true_forces_heuristic(monkeypatch):
    """Explicit ``non_nvlink_multi_rank=True`` overrides single-rank auto-detect."""
    _patch_runtime_slight_offload_advantage(monkeypatch, delta_ratio=0.02)
    trace = _make_trace(world=1)
    layout = _make_layout()
    hw = _make_hw(gpu_count=1, has_nvlink=False)
    result = search(
        trace,
        layout,
        capacity_bytes=2 * GB,
        hw=hw,
        prefer_no_offload_on_non_nvlink=True,
        non_nvlink_multi_rank=True,
    )
    assert result.cfg.n_offload == 0, (
        "explicit non_nvlink_multi_rank=True must engage the heuristic; "
        f"got n_offload={result.cfg.n_offload}"
    )


def test_non_nvlink_multi_rank_explicit_false_skips_heuristic(monkeypatch):
    """Explicit ``non_nvlink_multi_rank=False`` overrides multi-rank auto-detect."""
    _patch_runtime_slight_offload_advantage(monkeypatch, delta_ratio=0.02)
    trace = _make_trace(world=2)
    layout = _make_layout()
    hw = _make_hw(gpu_count=2, has_nvlink=False)
    result = search(
        trace,
        layout,
        capacity_bytes=2 * GB,
        hw=hw,
        prefer_no_offload_on_non_nvlink=True,
        non_nvlink_multi_rank=False,
    )
    assert result.cfg.n_offload > 0, (
        "explicit non_nvlink_multi_rank=False must skip the heuristic; "
        f"got n_offload={result.cfg.n_offload}"
    )


# ---------------------------------------------------------------------
# Edge case: when n_offload=0 is INFEASIBLE the heuristic still picks
# a valid feasible config (it can't conjure non-existent candidates).
# ---------------------------------------------------------------------


def test_heuristic_does_not_pick_infeasible_n_offload_zero(monkeypatch):
    """If only n_offload>0 fits under capacity, heuristic still returns a feasible cfg.

    Builds a scenario where the activation footprint forces ``n_offload>0``
    for feasibility (by mocking ``estimate_peak`` to make n_offload=0 over
    budget). The heuristic can re-rank within the noise band but cannot
    synthesise an n_offload=0 config that doesn't exist; the picked cfg
    must remain a real, feasible candidate.
    """
    _patch_runtime_by_n_offload(monkeypatch)
    trace = _make_trace(world=2)
    layout = _make_layout()
    hw = _make_hw(gpu_count=2, has_nvlink=False)

    # Inject a model_state_present override so n_offload=0 is over-budget but
    # n_offload>0 fits — model_state_present_bytes is called per-candidate in
    # search() and folds into raw_peak. Real signal: when chunks are offloaded,
    # only the persistent + buffered subset's model state lives on GPU.
    from axolotl.integrations.protrain.cost import memory as _mem

    real_model_state = _mem.model_state_present_bytes

    def _fake_model_state(cfg, layout_, trace_):
        base = real_model_state(cfg, layout_, trace_)
        if cfg.n_offload == 0:
            # Push n_offload=0 well over capacity.
            return base + 4 * GB
        return base

    monkeypatch.setattr(_mem, "model_state_present_bytes", _fake_model_state)

    result = search(
        trace,
        layout,
        capacity_bytes=2 * GB,
        hw=hw,
        prefer_no_offload_on_non_nvlink=True,
    )
    # The searcher must return SOME feasible cfg; heuristic must NOT prevent
    # picking n_offload>0 when n_offload=0 is infeasible.
    assert result.cfg is not None
    assert result.cfg.n_offload > 0, (
        "when n_offload=0 is infeasible, heuristic must fall back to a "
        f"feasible n_offload>0; got n_offload={result.cfg.n_offload}"
    )
