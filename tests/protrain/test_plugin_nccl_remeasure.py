"""Tests for ``plugin._remeasure_nccl_and_research`` lifecycle wiring.

The helper bridges the gap between ``post_model_load`` (where the profiler
ran without a live process group, so NCCL tables are empty) and
``post_trainer_create`` (where Accelerate has finished bringing up dist).
Real NCCL collectives require a multi-rank rendezvous, so these tests
exercise the *wiring* — when the helper fires, what it splices into the
trace, and whether it logs / updates the SearchResult on a config change
— with ``torch.distributed`` and ``measure_nccl`` mocked. Measurement
correctness itself is covered by ``scripts/protrain/measure_nccl.py``
under torchrun.
"""

from __future__ import annotations

import dataclasses
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
# Test fixtures
# ---------------------------------------------------------------------------


def _make_trace(*, world: int = 1, with_nccl: bool = False) -> ProfilerTrace:
    """Minimal ProfilerTrace stub. Fields are typed; values are unrealistic."""
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
        nccl_gather_s={1 << 20: 0.001} if with_nccl else {},
        nccl_reduce_s={1 << 20: 0.001} if with_nccl else {},
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
        chunks=((),),  # contents irrelevant for the helper
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


def _make_search_result(
    *, n_persist: int = 1, n_buffer: int = 1, predicted_iter_s: float = 0.10
) -> SearchResult:
    return SearchResult(
        cfg=CostConfig(
            n_persist=n_persist, n_buffer=n_buffer, n_swap=0, n_checkpoint=0
        ),
        block_map=cast(
            BlockStrategyMap,
            {cast(BlockId, 0): BlockMode.CKPT},
        ),
        predicted_peak_bytes=1 << 30,
        predicted_iter_s=predicted_iter_s,
    )


def _make_wrapped(*, with_nccl: bool = False) -> WrappedModel:
    """Build a WrappedModel-like object with the private attrs the helper needs."""
    import torch.nn as nn

    trace = _make_trace(world=1, with_nccl=with_nccl)
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
    return wrapped


def _patch_dist(*, initialized: bool, world_size: int = 2):
    """Patch ``torch.distributed`` to look like a live process group."""
    import torch.distributed as dist

    return [
        patch.object(dist, "is_available", return_value=True),
        patch.object(dist, "is_initialized", return_value=initialized),
        patch.object(dist, "get_world_size", return_value=world_size),
    ]


# ---------------------------------------------------------------------------
# Helper behavior
# ---------------------------------------------------------------------------


def test_remeasure_noop_when_dist_not_initialized():
    """Single-process / pre-init: helper must report no-op without touching anything."""
    pytest.importorskip("torch")

    from axolotl.integrations.protrain.plugin import _remeasure_nccl_and_research

    wrapped = _make_wrapped()
    patches = _patch_dist(initialized=False, world_size=1)
    for p in patches:
        p.start()
    try:
        updated, changed = _remeasure_nccl_and_research(wrapped)
    finally:
        for p in patches:
            p.stop()

    assert updated is False
    assert changed is False
    # Trace untouched.
    assert wrapped._trace.nccl_gather_s == {}  # type: ignore[attr-defined]
    assert wrapped._trace.world == 1  # type: ignore[attr-defined]


def test_remeasure_noop_on_world_size_one():
    """world_size==1 means no NCCL traffic — helper short-circuits."""
    pytest.importorskip("torch")

    from axolotl.integrations.protrain.plugin import _remeasure_nccl_and_research

    wrapped = _make_wrapped()
    patches = _patch_dist(initialized=True, world_size=1)
    for p in patches:
        p.start()
    try:
        updated, changed = _remeasure_nccl_and_research(wrapped)
    finally:
        for p in patches:
            p.stop()

    assert (updated, changed) == (False, False)


def test_remeasure_noop_when_trace_already_has_nccl_for_this_world():
    """Idempotent: a trace already populated for the live world is left alone."""
    pytest.importorskip("torch")

    from axolotl.integrations.protrain.plugin import _remeasure_nccl_and_research

    wrapped = _make_wrapped(with_nccl=True)
    # Pre-populate trace with world=2 + non-empty tables (cache hit case).
    wrapped._trace = dataclasses.replace(wrapped._trace, world=2)  # type: ignore[attr-defined]
    patches = _patch_dist(initialized=True, world_size=2)
    for p in patches:
        p.start()
    try:
        updated, changed = _remeasure_nccl_and_research(wrapped)
    finally:
        for p in patches:
            p.stop()

    assert (updated, changed) == (False, False)


def test_remeasure_splices_nccl_and_keeps_search_result_when_unchanged(
    tmp_path, monkeypatch
):
    """Happy path with same cfg: trace gets new tables, search re-runs, no WARN config change."""
    pytest.importorskip("torch")

    # Redirect cache writes so we don't pollute the real ~/.cache.
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    from axolotl.integrations.protrain import plugin as plugin_mod

    wrapped = _make_wrapped()
    fake_gather = {1 << 20: 0.0023, 64 << 20: 0.0117}
    fake_reduce = {1 << 20: 0.0019, 64 << 20: 0.0094}

    orig_result = wrapped.search_result
    new_result = _make_search_result(predicted_iter_s=0.12)  # same cfg, new ETA
    assert new_result.cfg == orig_result.cfg, "test setup invariant"

    measure_calls: list[int] = []

    def fake_measure(world_size: int):
        measure_calls.append(world_size)
        return fake_gather, fake_reduce

    search_calls: list[ProfilerTrace] = []

    def fake_search(trace, layout, capacity_bytes, hw, cpu_capacity_bytes=None):
        search_calls.append(trace)
        return new_result

    patches = _patch_dist(initialized=True, world_size=2) + [
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

    assert updated is True
    assert changed is False  # cfg + block_map matched
    assert measure_calls == [2]
    assert len(search_calls) == 1, "search() should be re-run exactly once"

    # Trace got the new tables and the new world size.
    new_trace = wrapped._trace  # type: ignore[attr-defined]
    assert new_trace.nccl_gather_s == fake_gather
    assert new_trace.nccl_reduce_s == fake_reduce
    assert new_trace.world == 2

    # search_result swapped to the new (cfg-equal) result so its
    # predicted_iter_s reflects the updated NCCL cost.
    assert wrapped.search_result is new_result
    assert wrapped.search_result.predicted_iter_s == pytest.approx(0.12)

    # Trace was persisted under the world=2 cache key (not the original
    # world=1 key, which we leave alone).
    new_key = ProfilerCacheKey(
        arch_hash="deadbeef", bs=1, seq=128, sku="MockGPU", world=2
    )
    expected_path = tmp_path / "protrain" / "profiler" / f"{new_key.fingerprint()}.json"
    assert expected_path.exists(), (
        f"updated trace not persisted at expected path {expected_path}"
    )


def test_remeasure_raises_when_cfg_changes(tmp_path, monkeypatch):
    """Different cfg post-NCCL: helper raises RuntimeError so training halts.

    Per CodeRabbit PR #19: continuing under the bootstrap plan when the
    accurate (post-NCCL) search picks a different cfg is a silent
    correctness drift — the chunk_manager / scheduler / hooks / optimizer
    state slots are already wired for the bootstrap plan and cannot be
    rebuilt mid-flight, so we must fail fast and direct the user to fix
    the early-dist-init path. The post-NCCL plan is still stashed on
    ``wrapped.post_nccl_search_result`` BEFORE the raise so callers can
    introspect both plans from the exception's caller.
    """
    pytest.importorskip("torch")

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    from axolotl.integrations.protrain import plugin as plugin_mod

    wrapped = _make_wrapped()
    sentinel_chunk_manager = object()
    wrapped.chunk_manager = sentinel_chunk_manager

    orig_cfg = wrapped.search_result.cfg
    orig_search_result = wrapped.search_result  # capture for later identity check
    different_result = _make_search_result(
        n_persist=orig_cfg.n_persist + 1, predicted_iter_s=0.08
    )
    assert different_result.cfg != orig_cfg

    patches = _patch_dist(initialized=True, world_size=4) + [
        patch(
            "axolotl.integrations.protrain.profiler.measure_nccl",
            return_value=({1 << 20: 0.001}, {1 << 20: 0.001}),
        ),
        patch(
            "axolotl.integrations.protrain.search.search",
            return_value=different_result,
        ),
    ]
    for p in patches:
        p.start()
    try:
        with pytest.raises(RuntimeError) as exc_info:
            plugin_mod._remeasure_nccl_and_research(wrapped)
    finally:
        for p in patches:
            p.stop()

    # The error message must mention both cfgs so callers / tests can
    # introspect, and point at the early-dist-init fix.
    msg = str(exc_info.value)
    assert "late NCCL re-search" in msg
    assert "bootstrap cfg" in msg
    assert "post-NCCL cfg" in msg
    assert "ddp_backend" in msg or "launcher" in msg
    # Live runtime state untouched — search_result/_trace continue to reflect
    # the installed (bootstrap) plan because chunk_manager/scheduler/hooks/
    # optimizer slots cannot be rebuilt mid-flight.
    assert wrapped.search_result is orig_search_result
    assert wrapped.search_result is not different_result
    # The post-NCCL plan is published on telemetry-only fields BEFORE the
    # raise, so post-mortem inspection sees both plans.
    assert getattr(wrapped, "post_nccl_search_result", None) is different_result
    # chunk_manager preserved — the spec is explicit that we must not
    # rebuild it post-research (optimizer state slots are wired into the
    # trainer already).
    assert wrapped.chunk_manager is sentinel_chunk_manager


def test_remeasure_swallows_measure_failure_and_leaves_state_intact(monkeypatch):
    """If measure_nccl raises, trace + search_result remain untouched and we report no-op."""
    pytest.importorskip("torch")

    from axolotl.integrations.protrain import plugin as plugin_mod

    wrapped = _make_wrapped()
    orig_trace = wrapped._trace  # type: ignore[attr-defined]
    orig_result = wrapped.search_result

    patches = _patch_dist(initialized=True, world_size=2) + [
        patch(
            "axolotl.integrations.protrain.profiler.measure_nccl",
            side_effect=RuntimeError("boom"),
        ),
        # search() must NOT be called when measurement fails.
        patch(
            "axolotl.integrations.protrain.search.search",
            side_effect=AssertionError("search should not run on measure failure"),
        ),
    ]
    for p in patches:
        p.start()
    try:
        updated, changed = plugin_mod._remeasure_nccl_and_research(wrapped)
    finally:
        for p in patches:
            p.stop()

    assert (updated, changed) == (False, False)
    assert wrapped._trace is orig_trace  # type: ignore[attr-defined]
    assert wrapped.search_result is orig_result


def test_remeasure_skips_when_wrapped_missing_stashed_state(caplog):
    """A WrappedModel that pre-dates the stash (or was hand-built) gets a WARN, no crash."""
    pytest.importorskip("torch")
    import logging

    import torch.nn as nn

    from axolotl.integrations.protrain.plugin import _remeasure_nccl_and_research

    bare = WrappedModel(
        module=nn.Identity(),
        search_result=_make_search_result(),
        chunk_manager=None,
        scheduler=None,
        _hook_handles=[],
    )
    # Deliberately do NOT set _trace / _layout / _hardware_profile / _capacity_bytes.

    patches = _patch_dist(initialized=True, world_size=2)
    for p in patches:
        p.start()
    # ``axolotl.logging_config.configure_logging()`` (run at axolotl.cli
    # import time, which CI hits) sets ``propagate=False`` on the
    # ``axolotl`` logger. Pytest's ``caplog`` installs its handler at the
    # root, so non-propagating records never reach it and the assertion
    # below sees an empty ``caplog.records``. Force propagation for the
    # duration of the test (and restore on exit) so caplog deterministically
    # sees the production WARN.
    ax_logger = logging.getLogger("axolotl")
    prev_propagate = ax_logger.propagate
    ax_logger.propagate = True
    try:
        with caplog.at_level(logging.WARNING):
            updated, changed = _remeasure_nccl_and_research(bare)
    finally:
        ax_logger.propagate = prev_propagate
        for p in patches:
            p.stop()

    assert (updated, changed) == (False, False)
    assert any("missing one of" in rec.message for rec in caplog.records), (
        "expected a WARN explaining which fields were missing"
    )
