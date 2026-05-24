"""Coverage for the cluster-wide ``max_memory_allocated`` collective.

Backs the §14.iii fix from the multimodal-CPT / ProTrain proposal: the
``memory/max_active (GiB)`` metric logged by ``AxolotlTrainer`` previously
reported only rank 0's peak. Under round-robin partition or any layout
where one rank owns a larger share (e.g. the rank holding the lm_head
before within-shard fallback), the worst-case peak is invisible from the
training log -- exactly the value that matters for fitting the GPU budget.

These tests exercise :func:`axolotl.utils.bench._gather_per_rank_peak_bytes`
which performs the ``all_reduce(MAX)`` over a per-rank int64 tensor.
"""

from __future__ import annotations

import os

import pytest


def test_single_rank_returns_local_peak(monkeypatch) -> None:
    """No dist init -> return the local value unchanged in both slots."""
    import torch
    import torch.distributed as dist

    from axolotl.utils.bench import _gather_per_rank_peak_bytes

    # Force the "dist inactive" branch even if a previous test left a PG behind.
    monkeypatch.setattr(dist, "is_available", lambda: False)

    cluster_max, per_rank = _gather_per_rank_peak_bytes(local_peak_bytes=4096)
    assert cluster_max == 4096
    assert per_rank == [4096]

    # Zero peak (no CUDA / no allocation) -> still returns the local value.
    cluster_max, per_rank = _gather_per_rank_peak_bytes(local_peak_bytes=0)
    assert cluster_max == 0
    assert per_rank == [0]

    # Auto-detect path: when CUDA isn't available the helper falls back to 0
    # rather than crashing.
    if not torch.cuda.is_available():
        cluster_max, per_rank = _gather_per_rank_peak_bytes()
        assert cluster_max == 0
        assert per_rank == [0]


def _worker_multi_rank_gloo_max(
    rank: int, world_size: int, tmpdir: str, planted: list[int]
) -> None:
    """Spawned worker: each rank plants ``planted[rank]`` as its local peak.

    The cluster-wide max must equal ``max(planted)`` regardless of which
    rank carries the largest value, and the per-rank list must reproduce
    ``planted`` in rank order.
    """
    import torch
    import torch.distributed as dist

    from axolotl.utils.bench import _gather_per_rank_peak_bytes

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29555")
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{tmpdir}/rendezvous-peak",
        rank=rank,
        world_size=world_size,
    )

    try:
        local = int(planted[rank])
        cluster_max, per_rank = _gather_per_rank_peak_bytes(local_peak_bytes=local)
        expected_max = int(max(planted))
        assert cluster_max == expected_max, (
            f"rank {rank}: cluster_max={cluster_max} expected={expected_max}"
        )
        assert len(per_rank) == world_size, (
            f"rank {rank}: per_rank len={len(per_rank)} expected={world_size}"
        )
        for i, value in enumerate(planted):
            assert per_rank[i] == int(value), (
                f"rank {rank}: per_rank[{i}]={per_rank[i]} expected={int(value)}"
            )
        torch.distributed.barrier()
    finally:
        dist.destroy_process_group()


def test_multi_rank_gloo_max(tmp_path) -> None:
    """2-rank gloo all_reduce(MAX) returns the larger plant on every rank.

    Plants rank 0 -> 1 GiB, rank 1 -> 4 GiB (rank 1 is the peak rank). The
    cluster-wide max must be 4 GiB on both ranks; rank-0-only collection
    would have reported 1 GiB and missed the real peak.
    """
    pytest.importorskip("torch")
    import torch

    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable")

    import torch.multiprocessing as mp

    world_size = 2
    planted = [1 * (1 << 30), 4 * (1 << 30)]  # 1 GiB, 4 GiB
    mp.spawn(
        _worker_multi_rank_gloo_max,
        args=(world_size, str(tmp_path), planted),
        nprocs=world_size,
        join=True,
    )


def _worker_multi_rank_gloo_max_rank0(rank: int, world_size: int, tmpdir: str) -> None:
    """Worker: plant the peak on rank 0 to confirm direction-independence.

    Mirrors the round-robin partition case where the peak rank IS rank 0,
    so the gather must still produce ``(local, [local, smaller])`` rather
    than collapsing to rank 1's smaller value.
    """
    import torch.distributed as dist

    from axolotl.utils.bench import _gather_per_rank_peak_bytes

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29556")
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{tmpdir}/rendezvous-peak-rank0",
        rank=rank,
        world_size=world_size,
    )

    try:
        # Rank 0 carries the peak this time.
        planted = [8 * (1 << 30), 2 * (1 << 30)]  # 8 GiB on rank 0, 2 GiB on rank 1
        local = int(planted[rank])
        cluster_max, per_rank = _gather_per_rank_peak_bytes(local_peak_bytes=local)
        assert cluster_max == int(planted[0])
        assert per_rank == [int(planted[0]), int(planted[1])]
    finally:
        dist.destroy_process_group()


def test_multi_rank_gloo_max_rank0_is_peak(tmp_path) -> None:
    """Sanity: when rank 0 IS the peak rank, the helper still works."""
    pytest.importorskip("torch")
    import torch

    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable")

    import torch.multiprocessing as mp

    mp.spawn(
        _worker_multi_rank_gloo_max_rank0,
        args=(2, str(tmp_path)),
        nprocs=2,
        join=True,
    )
