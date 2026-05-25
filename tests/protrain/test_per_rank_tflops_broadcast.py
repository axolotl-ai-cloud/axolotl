"""Tests for the PR #20 per-rank gpu_compute_tflops broadcast.

The broadcast (``_broadcast_gpu_compute_tflops`` in
``axolotl.integrations.protrain.api.model_wrapper``) forces every rank to use rank-0's
``gpu_compute_tflops`` before the cost-model searcher runs. Without it, per-rank
``measure_compute_rate`` outliers on mixed-SKU rigs (3090 + 3090 Ti, plus thermal/clock
variance) flow through ``cost.runtime._sku_compute_scale``, hit the 1%-noise-band
tie-breaker in ``search.exhaustive``, and ranks pick different CostConfigs. Different
picks -> different block_maps -> different chunks sharded -> NCCL all_gather
collectives fire on different chunk_ids per rank -> deadlock at the first multi-rank
collective.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from axolotl.integrations.protrain.types import HardwareProfile


def _hw(tflops: float) -> HardwareProfile:
    return HardwareProfile(
        gpu_sku="MockGPU",
        gpu_memory_bytes=24 * (1 << 30),
        gpu_count=1,
        pcie_h2d_bps=10e9,
        pcie_d2h_bps=10e9,
        has_nvlink=False,
        gpu_compute_tflops=tflops,
    )


def _patch_dist(
    *, initialized: bool, world_size: int, rank: int, rank0_value: float
):
    """Patch torch.distributed for in-process tests of the broadcast helper."""
    import torch.distributed as dist

    def _fake_broadcast(object_list, src=0):  # noqa: ARG001 — mimic dist API
        object_list[0] = rank0_value

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
    patches = _patch_dist(
        initialized=True, world_size=4, rank=1, rank0_value=33.4
    )
    for p in patches:
        p.start()
    try:
        with caplog.at_level("WARNING"):
            hw_out = mw._broadcast_gpu_compute_tflops(hw_in)
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
    patches = _patch_dist(
        initialized=True, world_size=1, rank=0, rank0_value=40.0
    )
    for p in patches:
        p.start()
    try:
        with caplog.at_level("WARNING"):
            hw_out = mw._broadcast_gpu_compute_tflops(hw_in)
    finally:
        for p in patches:
            p.stop()

    assert hw_out is hw_in
    assert hw_out.gpu_compute_tflops == pytest.approx(40.0)
    assert not any(
        "gpu_compute_tflops" in rec.getMessage() for rec in caplog.records
    )


def test_broadcast_noop_when_dist_not_initialized():
    """Pre-init: short-circuit, no value change, no exception."""
    pytest.importorskip("torch")
    from axolotl.integrations.protrain.api import model_wrapper as mw

    hw_in = _hw(33.4)
    patches = _patch_dist(
        initialized=False, world_size=4, rank=0, rank0_value=99.9
    )
    for p in patches:
        p.start()
    try:
        hw_out = mw._broadcast_gpu_compute_tflops(hw_in)
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
    patches = _patch_dist(
        initialized=True, world_size=4, rank=2, rank0_value=33.4
    )
    for p in patches:
        p.start()
    try:
        with caplog.at_level("WARNING"):
            hw_out = mw._broadcast_gpu_compute_tflops(hw_in)
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
    patches = _patch_dist(
        initialized=True, world_size=4, rank=0, rank0_value=67.5
    )
    for p in patches:
        p.start()
    try:
        with caplog.at_level("WARNING"):
            hw_out = mw._broadcast_gpu_compute_tflops(hw_in)
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
    patches = _patch_dist(
        initialized=True, world_size=4, rank=3, rank0_value=0.0
    )
    for p in patches:
        p.start()
    try:
        hw_out = mw._broadcast_gpu_compute_tflops(hw_in)
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
            hw_out = mw._broadcast_gpu_compute_tflops(hw_in)
    finally:
        for p in patches:
            p.stop()

    assert hw_out.gpu_compute_tflops == pytest.approx(40.0)
    assert any(
        "gpu_compute_tflops broadcast failed" in rec.getMessage()
        for rec in caplog.records
    )
