"""Late NCCL re-search must short-circuit when all four override knobs pin the bootstrap plan, avoiding cfg_changed RuntimeError."""

from __future__ import annotations

from typing import cast
from unittest.mock import patch

import pytest

from axolotl.integrations.protrain.profiler.cache import ProfilerCacheKey
from axolotl.integrations.protrain.types import (
    BlockId,
    BlockMode,
    BlockStrategyMap,
    ChunkLayout,
    CostConfig,
    HardwareProfile,
    OpId,
    OpRecord,
    ProfilerTrace,
    SearchResult,
    WrappedModel,
)

# ---------------------------------------------------------------------------
# Fixture builders (mirror tests/protrain/test_plugin_nccl_remeasure.py so the
# two test modules describe the helper from compatible angles).
# ---------------------------------------------------------------------------


def _make_trace(*, world: int = 1) -> ProfilerTrace:
    """Minimal ProfilerTrace stub with empty NCCL tables matching the override-skip synthesized trace."""
    op = OpRecord(
        op_id=cast(OpId, 0),
        module_path="layer0",
        qualified_name="aten::linear",
        shape_signature=((1, 4),),
        block_id=cast(BlockId, 0),
        is_forward=True,
    )
    return ProfilerTrace(
        op_order=(op,),
        intra_op_delta={cast(OpId, 0): 0},
        inter_op_delta={cast(OpId, 0): 0},
        activation_sizes={cast(BlockId, 0): 1024},
        model_state_bytes=1024,
        pcie_h2d_bps=10e9,
        pcie_d2h_bps=10e9,
        nccl_gather_s={},
        nccl_reduce_s={},
        arch_hash="deadbeef",
        bs=1,
        seq=128,
        sku="MockGPU",
        world=world,
    )


def _make_layout() -> ChunkLayout:
    return ChunkLayout(
        S_chunk=1 << 20,
        N_chunk=2,
        chunks=((),),
        param_to_chunk={},
        block_to_chunks={},
    )


def _make_hw() -> HardwareProfile:
    return HardwareProfile(
        gpu_sku="MockGPU",
        gpu_memory_bytes=24 * (1 << 30),
        gpu_count=1,
        pcie_h2d_bps=10e9,
        pcie_d2h_bps=10e9,
        has_nvlink=False,
    )


def _make_search_result() -> SearchResult:
    return SearchResult(
        cfg=CostConfig(n_persist=1, n_buffer=1, n_swap=0, n_checkpoint=0),
        block_map=cast(
            BlockStrategyMap,
            {cast(BlockId, 0): BlockMode.CKPT},
        ),
        predicted_peak_bytes=1 << 30,
        predicted_iter_s=0.1,
    )


def _make_wrapped(*, with_override_flag: bool | None = False) -> WrappedModel:
    """Build a WrappedModel-like object with the private attrs the helper needs (flag True/False/missing)."""
    import torch.nn as nn

    trace = _make_trace(world=1)
    layout = _make_layout()
    hw = _make_hw()
    cache_key = ProfilerCacheKey(
        arch_hash="deadbeef", bs=1, seq=128, sku="MockGPU", world=1
    )
    wrapped = WrappedModel(
        module=nn.Identity(),
        search_result=_make_search_result(),
        chunk_manager=None,
        scheduler=None,
        _hook_handles=[],
    )
    wrapped._trace = trace  # type: ignore[attr-defined]
    wrapped._layout = layout  # type: ignore[attr-defined]
    wrapped._capacity_bytes = 22 * (1 << 30)  # type: ignore[attr-defined]
    wrapped._hardware_profile = hw  # type: ignore[attr-defined]
    wrapped._cache_key = cache_key  # type: ignore[attr-defined]
    if with_override_flag is not None:
        wrapped._override_skip_trace = with_override_flag  # type: ignore[attr-defined]
    return wrapped


def _patch_dist(*, initialized: bool, world_size: int = 4):
    """Patch ``torch.distributed`` to look like a live multi-rank PG."""
    import torch.distributed as dist

    return [
        patch.object(dist, "is_available", return_value=True),
        patch.object(dist, "is_initialized", return_value=initialized),
        patch.object(dist, "get_world_size", return_value=world_size),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_late_search_skipped_when_overrides_set():
    """With _override_skip_trace=True the helper short-circuits before measure_nccl or search.search runs."""
    pytest.importorskip("torch")

    from axolotl.integrations.protrain import plugin as plugin_mod

    wrapped = _make_wrapped(with_override_flag=True)
    orig_search_result = wrapped.search_result
    orig_trace = wrapped._trace  # type: ignore[attr-defined]

    measure_calls: list[int] = []
    search_calls: list[ProfilerTrace] = []

    def fake_measure(world_size: int):
        measure_calls.append(world_size)
        return {1 << 20: 0.001}, {1 << 20: 0.001}

    def fake_search(trace, layout, capacity_bytes, hw, cpu_capacity_bytes=None):
        search_calls.append(trace)
        return _make_search_result()

    patches = _patch_dist(initialized=True, world_size=4) + [
        patch(
            "axolotl.integrations.protrain.profiler.measure_nccl",
            side_effect=fake_measure,
        ),
        patch(
            "axolotl.integrations.protrain.search.search",
            side_effect=fake_search,
        ),
    ]
    for p in patches:
        p.start()
    try:
        updated, changed = plugin_mod._remeasure_nccl_and_research(wrapped)
    finally:
        for p in patches:
            p.stop()

    # Helper returned the no-op signal.
    assert (updated, changed) == (False, False)

    # Crucially: neither measurement nor search ran.
    assert measure_calls == [], (
        f"measure_nccl was called {measure_calls} times on the override-skip "
        "path; the gate should short-circuit before the measurement."
    )
    assert search_calls == [], (
        f"search.search was called {len(search_calls)} times on the override-"
        "skip path; the gate should short-circuit before the re-run."
    )

    # Trace and search_result untouched (still the bootstrap synthesis).
    assert wrapped.search_result is orig_search_result
    assert wrapped._trace is orig_trace  # type: ignore[attr-defined]
    assert wrapped._trace.nccl_gather_s == {}  # type: ignore[attr-defined]
    assert wrapped._trace.nccl_reduce_s == {}  # type: ignore[attr-defined]
    # post_nccl_search_result must NOT have been stashed (no late search ran).
    assert not hasattr(wrapped, "post_nccl_search_result")
    assert not hasattr(wrapped, "post_nccl_trace")


def test_late_search_runs_when_overrides_not_set(tmp_path, monkeypatch):
    """Control: _override_skip_trace=False makes measure_nccl and search.search both fire."""
    pytest.importorskip("torch")

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    from axolotl.integrations.protrain import plugin as plugin_mod

    wrapped = _make_wrapped(with_override_flag=False)

    fake_gather = {1 << 20: 0.0023}
    fake_reduce = {1 << 20: 0.0019}

    measure_calls: list[int] = []
    search_calls: list[ProfilerTrace] = []

    def fake_measure(world_size: int):
        measure_calls.append(world_size)
        return fake_gather, fake_reduce

    def fake_search(trace, layout, capacity_bytes, hw, cpu_capacity_bytes=None):
        search_calls.append(trace)
        # Return the SAME cfg so cfg_changed=False (no fail-fast raise).
        return _make_search_result()

    patches = _patch_dist(initialized=True, world_size=4) + [
        patch(
            "axolotl.integrations.protrain.profiler.measure_nccl",
            side_effect=fake_measure,
        ),
        patch(
            "axolotl.integrations.protrain.search.search",
            side_effect=fake_search,
        ),
    ]
    for p in patches:
        p.start()
    try:
        updated, changed = plugin_mod._remeasure_nccl_and_research(wrapped)
    finally:
        for p in patches:
            p.stop()

    # Both fired exactly once.
    assert measure_calls == [4], (
        f"measure_nccl call list {measure_calls} mismatched expected [4] on "
        "the non-override searcher path"
    )
    assert len(search_calls) == 1, (
        f"search.search ran {len(search_calls)} times; expected 1 on the "
        "non-override searcher path"
    )

    # Trace got the new tables; search_result swapped (same cfg, refreshed).
    assert (updated, changed) == (True, False)
    assert wrapped._trace.nccl_gather_s == fake_gather  # type: ignore[attr-defined]
    assert wrapped._trace.nccl_reduce_s == fake_reduce  # type: ignore[attr-defined]


def test_late_search_skipped_when_attr_missing_does_not_skip(tmp_path, monkeypatch):
    """Missing _override_skip_trace must not short-circuit; gate is positive opt-in."""
    pytest.importorskip("torch")

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    from axolotl.integrations.protrain import plugin as plugin_mod

    wrapped = _make_wrapped(with_override_flag=None)
    assert not hasattr(wrapped, "_override_skip_trace"), (
        "test setup invariant: this case must NOT have the attribute"
    )

    measure_calls: list[int] = []
    search_calls: list[ProfilerTrace] = []

    def fake_measure(world_size: int):
        measure_calls.append(world_size)
        return {1 << 20: 0.001}, {1 << 20: 0.001}

    def fake_search(trace, layout, capacity_bytes, hw, cpu_capacity_bytes=None):
        search_calls.append(trace)
        return _make_search_result()

    patches = _patch_dist(initialized=True, world_size=4) + [
        patch(
            "axolotl.integrations.protrain.profiler.measure_nccl",
            side_effect=fake_measure,
        ),
        patch(
            "axolotl.integrations.protrain.search.search",
            side_effect=fake_search,
        ),
    ]
    for p in patches:
        p.start()
    try:
        updated, changed = plugin_mod._remeasure_nccl_and_research(wrapped)
    finally:
        for p in patches:
            p.stop()

    # Without the flag, the helper ran the full path (single multi-rank
    # measurement, single search).
    assert measure_calls == [4]
    assert len(search_calls) == 1
    assert (updated, changed) == (True, False)
