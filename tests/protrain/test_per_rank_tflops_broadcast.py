"""Tests for the PR #20 per-rank searcher-critical HW broadcast.

The broadcast (``_broadcast_searcher_critical_hw`` in
``axolotl.integrations.protrain.api.model_wrapper``) forces every rank to use rank-0's
``gpu_compute_tflops`` and ``gpu_memory_bytes`` before the cost-model searcher runs.
Without it, per-rank ``measure_compute_rate`` outliers AND per-rank GPU-memory-size
deltas (3090 = 24576 MiB vs 3090 Ti = 24564 MiB) on mixed-SKU rigs flow through
``cost.runtime._sku_compute_scale`` (compute path) and the searcher's capacity cutoff
(memory path), hit the 1%-noise-band tie-breaker in ``search.exhaustive``, and ranks
pick different CostConfigs. Different picks -> different block_maps -> different chunks
sharded -> NCCL all_gather collectives fire on different chunk_ids per rank -> deadlock
at the first multi-rank collective.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from axolotl.integrations.protrain.types import HardwareProfile

_DEFAULT_MEM = 24 * (1 << 30)


def _hw(tflops: float, memory_bytes: int = _DEFAULT_MEM) -> HardwareProfile:
    return HardwareProfile(
        gpu_sku="MockGPU",
        gpu_memory_bytes=memory_bytes,
        gpu_count=1,
        pcie_h2d_bps=10e9,
        pcie_d2h_bps=10e9,
        has_nvlink=False,
        gpu_compute_tflops=tflops,
    )


def _patch_dist(
    *,
    initialized: bool,
    world_size: int,
    rank: int,
    rank0_tflops: float,
    rank0_memory: int = _DEFAULT_MEM,
):
    """Patch torch.distributed for in-process tests of the broadcast helper."""
    import torch.distributed as dist

    def _fake_broadcast(object_list, src=0):  # noqa: ARG001 — mimic dist API
        # The helper packs [tflops, memory_bytes].
        if len(object_list) >= 1:
            object_list[0] = rank0_tflops
        if len(object_list) >= 2:
            object_list[1] = rank0_memory

    return [
        patch.object(dist, "is_available", return_value=True),
        patch.object(dist, "is_initialized", return_value=initialized),
        patch.object(dist, "get_world_size", return_value=world_size),
        patch.object(dist, "get_rank", return_value=rank),
        patch.object(dist, "broadcast_object_list", side_effect=_fake_broadcast),
    ]


# ---------------------------------------------------------------------------
# Behavioral tests against the live helper.
# ---------------------------------------------------------------------------


def test_broadcast_overrides_nonzero_rank_when_value_differs(caplog):
    """Rank 1 measures 67.5 TFLOPS (outlier); rank-0 broadcast pushes 33.4 — rank 1 adopts it."""
    pytest.importorskip("torch")
    from axolotl.integrations.protrain.api import model_wrapper as mw

    hw_in = _hw(67.5)
    patches = _patch_dist(initialized=True, world_size=4, rank=1, rank0_tflops=33.4)
    for p in patches:
        p.start()
    try:
        with caplog.at_level("WARNING"):
            hw_out = mw._broadcast_searcher_critical_hw(hw_in)
    finally:
        for p in patches:
            p.stop()

    assert hw_out.gpu_compute_tflops == pytest.approx(33.4)
    assert any(
        "overriding local gpu_compute_tflops=67.50 with rank-0's 33.40"
        in rec.getMessage()
        for rec in caplog.records
    )


def test_broadcast_noop_on_single_rank(caplog):
    """world_size=1: no broadcast attempted, value unchanged, no WARN emitted."""
    pytest.importorskip("torch")
    from axolotl.integrations.protrain.api import model_wrapper as mw

    hw_in = _hw(40.0)
    patches = _patch_dist(initialized=True, world_size=1, rank=0, rank0_tflops=40.0)
    for p in patches:
        p.start()
    try:
        with caplog.at_level("WARNING"):
            hw_out = mw._broadcast_searcher_critical_hw(hw_in)
    finally:
        for p in patches:
            p.stop()

    assert hw_out is hw_in
    assert hw_out.gpu_compute_tflops == pytest.approx(40.0)
    assert not any("gpu_compute_tflops" in rec.getMessage() for rec in caplog.records)


def test_broadcast_noop_when_dist_not_initialized():
    """Pre-init: short-circuit, no value change, no exception."""
    pytest.importorskip("torch")
    from axolotl.integrations.protrain.api import model_wrapper as mw

    hw_in = _hw(33.4)
    patches = _patch_dist(initialized=False, world_size=4, rank=0, rank0_tflops=99.9)
    for p in patches:
        p.start()
    try:
        hw_out = mw._broadcast_searcher_critical_hw(hw_in)
    finally:
        for p in patches:
            p.stop()

    assert hw_out is hw_in
    assert hw_out.gpu_compute_tflops == pytest.approx(33.4)


def test_broadcast_silent_when_all_ranks_agree(caplog):
    """All ranks measure the same value: broadcast is a no-op, no override WARN."""
    pytest.importorskip("torch")
    from axolotl.integrations.protrain.api import model_wrapper as mw

    hw_in = _hw(33.4)
    patches = _patch_dist(initialized=True, world_size=4, rank=2, rank0_tflops=33.4)
    for p in patches:
        p.start()
    try:
        with caplog.at_level("WARNING"):
            hw_out = mw._broadcast_searcher_critical_hw(hw_in)
    finally:
        for p in patches:
            p.stop()

    assert hw_out.gpu_compute_tflops == pytest.approx(33.4)
    assert not any(
        "overriding local gpu_compute_tflops" in rec.getMessage()
        for rec in caplog.records
    )


def test_broadcast_silent_on_rank_zero(caplog):
    """Rank 0 IS the source of truth — no WARN fires even on outlier values."""
    pytest.importorskip("torch")
    from axolotl.integrations.protrain.api import model_wrapper as mw

    hw_in = _hw(67.5)
    patches = _patch_dist(initialized=True, world_size=4, rank=0, rank0_tflops=67.5)
    for p in patches:
        p.start()
    try:
        with caplog.at_level("WARNING"):
            hw_out = mw._broadcast_searcher_critical_hw(hw_in)
    finally:
        for p in patches:
            p.stop()

    assert hw_out.gpu_compute_tflops == pytest.approx(67.5)
    assert not any(
        "overriding local gpu_compute_tflops" in rec.getMessage()
        for rec in caplog.records
    )


def test_broadcast_skips_replace_on_zero_value():
    """If rank-0 broadcasts 0.0 (measurement failure), the local value is preserved."""
    pytest.importorskip("torch")
    from axolotl.integrations.protrain.api import model_wrapper as mw

    hw_in = _hw(40.0)
    patches = _patch_dist(initialized=True, world_size=4, rank=3, rank0_tflops=0.0)
    for p in patches:
        p.start()
    try:
        hw_out = mw._broadcast_searcher_critical_hw(hw_in)
    finally:
        for p in patches:
            p.stop()

    # rank-0 reported 0.0 -> the _bcast_tflops > 0.0 guard keeps the local value.
    assert hw_out.gpu_compute_tflops == pytest.approx(40.0)


def test_broadcast_swallows_broadcast_failure(caplog):
    """A raised RuntimeError from broadcast_object_list must not crash the caller."""
    pytest.importorskip("torch")
    from axolotl.integrations.protrain.api import model_wrapper as mw

    hw_in = _hw(40.0)

    import torch.distributed as dist

    def _raise(*args, **kwargs):  # noqa: ARG001
        raise RuntimeError("simulated NCCL collective failure")

    patches = [
        patch.object(dist, "is_available", return_value=True),
        patch.object(dist, "is_initialized", return_value=True),
        patch.object(dist, "get_world_size", return_value=4),
        patch.object(dist, "get_rank", return_value=1),
        patch.object(dist, "broadcast_object_list", side_effect=_raise),
    ]
    for p in patches:
        p.start()
    try:
        with caplog.at_level("WARNING"):
            hw_out = mw._broadcast_searcher_critical_hw(hw_in)
    finally:
        for p in patches:
            p.stop()

    assert hw_out.gpu_compute_tflops == pytest.approx(40.0)
    assert any(
        "searcher-critical HW broadcast failed" in rec.getMessage()
        for rec in caplog.records
    )


def test_broadcast_overrides_gpu_memory_bytes_on_size_delta(caplog):
    """3090 (24576 MiB) + 3090 Ti (24564 MiB) rig: non-zero ranks adopt rank-0's
    gpu_memory_bytes so the searcher capacity-cutoff is identical across ranks."""
    pytest.importorskip("torch")
    from axolotl.integrations.protrain.api import model_wrapper as mw

    _3090_MEM = 24576 * (1 << 20)
    _3090TI_MEM = 24564 * (1 << 20)

    # Rank 1 is a 3090 (24576 MiB); rank 0 is a 3090 Ti (24564 MiB).
    hw_in = _hw(40.0, memory_bytes=_3090_MEM)
    patches = _patch_dist(
        initialized=True,
        world_size=4,
        rank=1,
        rank0_tflops=40.0,
        rank0_memory=_3090TI_MEM,
    )
    for p in patches:
        p.start()
    try:
        with caplog.at_level("WARNING"):
            hw_out = mw._broadcast_searcher_critical_hw(hw_in)
    finally:
        for p in patches:
            p.stop()

    assert hw_out.gpu_memory_bytes == _3090TI_MEM
    assert any(
        "overriding local gpu_memory_bytes" in rec.getMessage()
        for rec in caplog.records
    )


def test_broadcast_capacity_overrides_when_cpu_capacity_differs(caplog):
    """psutil-derived cpu_capacity_bytes can differ by 50+ MiB between ranks;
    the searcher capacity-broadcast forces every rank to use rank-0's value."""
    pytest.importorskip("torch")
    from axolotl.integrations.protrain.api import model_wrapper as mw

    rank0_cap = 20 * (1 << 30)
    rank0_cpu_cap = 90 * (1 << 30)  # ~90 GiB available per rank
    local_cpu_cap = 89 * (1 << 30)  # rank-N saw 1 GiB less

    import torch.distributed as dist

    def _fake_broadcast(object_list, src=0):  # noqa: ARG001
        object_list[0] = rank0_cap
        object_list[1] = rank0_cpu_cap

    patches = [
        patch.object(dist, "is_available", return_value=True),
        patch.object(dist, "is_initialized", return_value=True),
        patch.object(dist, "get_world_size", return_value=4),
        patch.object(dist, "get_rank", return_value=2),
        patch.object(dist, "broadcast_object_list", side_effect=_fake_broadcast),
    ]
    for p in patches:
        p.start()
    try:
        with caplog.at_level("WARNING"):
            cap_out, cpu_cap_out = mw._broadcast_searcher_capacity(
                rank0_cap, local_cpu_cap
            )
    finally:
        for p in patches:
            p.stop()

    assert cap_out == rank0_cap
    assert cpu_cap_out == rank0_cpu_cap
    assert any(
        "overriding local cpu_capacity_bytes" in rec.getMessage()
        for rec in caplog.records
    )


def test_broadcast_capacity_noop_on_single_rank():
    """world_size=1: capacity broadcast is a no-op, values unchanged."""
    pytest.importorskip("torch")
    import torch.distributed as dist

    from axolotl.integrations.protrain.api import model_wrapper as mw

    patches = [
        patch.object(dist, "is_available", return_value=True),
        patch.object(dist, "is_initialized", return_value=True),
        patch.object(dist, "get_world_size", return_value=1),
        patch.object(dist, "get_rank", return_value=0),
    ]
    for p in patches:
        p.start()
    try:
        cap_out, cpu_cap_out = mw._broadcast_searcher_capacity(123, 456)
    finally:
        for p in patches:
            p.stop()

    assert cap_out == 123
    assert cpu_cap_out == 456


def test_broadcast_capacity_handles_none_cpu_capacity():
    """When cpu_capacity_bytes is None (psutil missing), broadcast preserves None."""
    pytest.importorskip("torch")
    import torch.distributed as dist

    from axolotl.integrations.protrain.api import model_wrapper as mw

    def _fake_broadcast(object_list, src=0):  # noqa: ARG001
        # Rank 0 also has None -> -1 sentinel.
        object_list[0] = 10 * (1 << 30)
        object_list[1] = -1

    patches = [
        patch.object(dist, "is_available", return_value=True),
        patch.object(dist, "is_initialized", return_value=True),
        patch.object(dist, "get_world_size", return_value=4),
        patch.object(dist, "get_rank", return_value=1),
        patch.object(dist, "broadcast_object_list", side_effect=_fake_broadcast),
    ]
    for p in patches:
        p.start()
    try:
        cap_out, cpu_cap_out = mw._broadcast_searcher_capacity(10 * (1 << 30), None)
    finally:
        for p in patches:
            p.stop()

    assert cap_out == 10 * (1 << 30)
    assert cpu_cap_out is None


def test_cache_key_sku_broadcast_makes_keys_identical():
    """Mixed-SKU rigs must build identical ProfilerCacheKey.sku so every rank loads the
    same cached ProfilerTrace. The broadcast happens inline at the cache_key construction
    site; this test exercises the underlying logic via a small in-process reproducer."""
    pytest.importorskip("torch")

    import torch.distributed as dist

    from axolotl.integrations.protrain.profiler.cache import ProfilerCacheKey

    local_sku = "NVIDIA GeForce RTX 3090"
    rank0_sku = "NVIDIA GeForce RTX 3090 Ti"

    def _fake_broadcast(object_list, src=0):  # noqa: ARG001
        object_list[0] = rank0_sku

    patches = [
        patch.object(dist, "is_available", return_value=True),
        patch.object(dist, "is_initialized", return_value=True),
        patch.object(dist, "get_world_size", return_value=4),
        patch.object(dist, "get_rank", return_value=2),
        patch.object(dist, "broadcast_object_list", side_effect=_fake_broadcast),
    ]
    for p in patches:
        p.start()
    try:
        # Replicate the broadcast block: rank-2 starts with local SKU, receives rank-0's.
        sku_holder = [local_sku]
        dist.broadcast_object_list(sku_holder, src=0)
        bcast_sku = str(sku_holder[0])
        rank0_key = ProfilerCacheKey(
            arch_hash="deadbeef",
            bs=2,
            seq=256,
            sku=rank0_sku,
            world=4,
        )
        rankn_key = ProfilerCacheKey(
            arch_hash="deadbeef",
            bs=2,
            seq=256,
            sku=bcast_sku,
            world=4,
        )
        assert rank0_key.fingerprint() == rankn_key.fingerprint()
    finally:
        for p in patches:
            p.stop()


def test_broadcast_silent_on_memory_match():
    """When gpu_memory_bytes already matches across ranks, the memory WARN does not fire."""
    pytest.importorskip("torch")
    import logging

    from axolotl.integrations.protrain.api import model_wrapper as mw

    hw_in = _hw(33.4)
    patches = _patch_dist(
        initialized=True,
        world_size=4,
        rank=2,
        rank0_tflops=33.4,
        rank0_memory=_DEFAULT_MEM,
    )

    captured: list[str] = []
    handler = logging.Handler()
    handler.emit = lambda record: captured.append(record.getMessage())
    logger = logging.getLogger("axolotl.integrations.protrain.api.model_wrapper")
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)

    try:
        for p in patches:
            p.start()
        try:
            hw_out = mw._broadcast_searcher_critical_hw(hw_in)
        finally:
            for p in patches:
                p.stop()
    finally:
        logger.removeHandler(handler)

    assert hw_out.gpu_memory_bytes == _DEFAULT_MEM
    assert not any("overriding local gpu_memory_bytes" in m for m in captured)
