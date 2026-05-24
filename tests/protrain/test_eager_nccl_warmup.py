"""Tests for ``plugin._eager_nccl_warmup`` wiring.

The helper pays ``ncclCommInitRank`` cost up-front by firing one no-op of
each NCCL collective the per-chunk path uses. Real NCCL collectives need a
multi-rank rendezvous, so these tests exercise the *wiring* — which
collectives fire, on which device, and how failures degrade — with
``torch.distributed`` mocked. Measurement / NCCL correctness lives in the
multi-rank smoke tests under torchrun.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _patch_dist(*, initialized: bool, world_size: int = 2, rank: int = 0):
    """Patch ``torch.distributed`` to look like a live process group."""
    import torch.distributed as dist

    def _noop_barrier(*args, **kwargs):  # noqa: ARG001 — match dist API
        return None

    return [
        patch.object(dist, "is_available", return_value=True),
        patch.object(dist, "is_initialized", return_value=initialized),
        patch.object(dist, "get_world_size", return_value=world_size),
        patch.object(dist, "get_rank", return_value=rank),
        patch.object(dist, "barrier", side_effect=_noop_barrier),
    ]


def _make_chunk_manager(*, zero3_shard: bool, n_chunk: int = 4):
    """Minimal duck-typed chunk_manager stand-in for the warmup helper."""
    layout = SimpleNamespace(N_chunk=n_chunk)
    return SimpleNamespace(zero3_shard=zero3_shard, layout=layout)


def test_warmup_noop_when_dist_not_initialized():
    """Single-process / pre-init: helper must short-circuit without touching dist."""
    pytest.importorskip("torch")

    import torch

    from axolotl.integrations.protrain.plugin import _eager_nccl_warmup

    chunk_manager = _make_chunk_manager(zero3_shard=False)
    patches = _patch_dist(initialized=False, world_size=1)
    calls = {"all_reduce": 0, "reduce_scatter_tensor": 0, "all_gather_into_tensor": 0}

    def _track(name):
        def _fn(*_args, **_kwargs):
            calls[name] += 1

        return _fn

    import torch.distributed as dist

    extra = [
        patch.object(dist, "all_reduce", side_effect=_track("all_reduce")),
        patch.object(
            dist, "reduce_scatter_tensor", side_effect=_track("reduce_scatter_tensor")
        ),
        patch.object(
            dist,
            "all_gather_into_tensor",
            side_effect=_track("all_gather_into_tensor"),
        ),
    ]
    for p in patches + extra:
        p.start()
    try:
        _eager_nccl_warmup(chunk_manager, torch.device("cpu"))
    finally:
        for p in patches + extra:
            p.stop()

    assert calls == {
        "all_reduce": 0,
        "reduce_scatter_tensor": 0,
        "all_gather_into_tensor": 0,
    }


def test_warmup_noop_on_world_size_one():
    """world_size==1: no NCCL traffic so helper short-circuits."""
    pytest.importorskip("torch")

    import torch
    import torch.distributed as dist

    from axolotl.integrations.protrain.plugin import _eager_nccl_warmup

    chunk_manager = _make_chunk_manager(zero3_shard=False)
    calls = {"all_reduce": 0, "reduce_scatter_tensor": 0, "all_gather_into_tensor": 0}

    def _track(name):
        def _fn(*_args, **_kwargs):
            calls[name] += 1

        return _fn

    patches = _patch_dist(initialized=True, world_size=1) + [
        patch.object(dist, "all_reduce", side_effect=_track("all_reduce")),
        patch.object(
            dist, "reduce_scatter_tensor", side_effect=_track("reduce_scatter_tensor")
        ),
        patch.object(
            dist,
            "all_gather_into_tensor",
            side_effect=_track("all_gather_into_tensor"),
        ),
    ]
    for p in patches:
        p.start()
    try:
        _eager_nccl_warmup(chunk_manager, torch.device("cpu"))
    finally:
        for p in patches:
            p.stop()

    assert calls == {
        "all_reduce": 0,
        "reduce_scatter_tensor": 0,
        "all_gather_into_tensor": 0,
    }


def test_warmup_fires_replicated_collectives_when_zero3_shard_false():
    """Mode A/B (replicated): warm all_reduce + reduce_scatter only.

    all_gather_into_tensor is only used by the sharded _gather_sharded path, so
    skip the warmup for it when zero3_shard=False.
    """
    pytest.importorskip("torch")

    import torch
    import torch.distributed as dist

    from axolotl.integrations.protrain.plugin import _eager_nccl_warmup

    chunk_manager = _make_chunk_manager(zero3_shard=False)
    calls = {"all_reduce": 0, "reduce_scatter_tensor": 0, "all_gather_into_tensor": 0}

    def _track(name):
        def _fn(*_args, **_kwargs):
            calls[name] += 1

        return _fn

    patches = _patch_dist(initialized=True, world_size=2) + [
        patch.object(dist, "all_reduce", side_effect=_track("all_reduce")),
        patch.object(
            dist, "reduce_scatter_tensor", side_effect=_track("reduce_scatter_tensor")
        ),
        patch.object(
            dist,
            "all_gather_into_tensor",
            side_effect=_track("all_gather_into_tensor"),
        ),
    ]
    for p in patches:
        p.start()
    try:
        _eager_nccl_warmup(chunk_manager, torch.device("cpu"))
    finally:
        for p in patches:
            p.stop()

    assert calls["all_reduce"] == 1
    assert calls["reduce_scatter_tensor"] == 1
    assert calls["all_gather_into_tensor"] == 0


def test_warmup_fires_all_gather_when_zero3_shard_true():
    """Mode C (sharded): all three collectives must fire."""
    pytest.importorskip("torch")

    import torch
    import torch.distributed as dist

    from axolotl.integrations.protrain.plugin import _eager_nccl_warmup

    chunk_manager = _make_chunk_manager(zero3_shard=True)
    calls = {"all_reduce": 0, "reduce_scatter_tensor": 0, "all_gather_into_tensor": 0}

    def _track(name):
        def _fn(*_args, **_kwargs):
            calls[name] += 1

        return _fn

    patches = _patch_dist(initialized=True, world_size=4) + [
        patch.object(dist, "all_reduce", side_effect=_track("all_reduce")),
        patch.object(
            dist, "reduce_scatter_tensor", side_effect=_track("reduce_scatter_tensor")
        ),
        patch.object(
            dist,
            "all_gather_into_tensor",
            side_effect=_track("all_gather_into_tensor"),
        ),
    ]
    for p in patches:
        p.start()
    try:
        _eager_nccl_warmup(chunk_manager, torch.device("cpu"))
    finally:
        for p in patches:
            p.stop()

    assert calls["all_reduce"] == 1
    assert calls["reduce_scatter_tensor"] == 1
    assert calls["all_gather_into_tensor"] == 1


def test_warmup_swallows_collective_exception(caplog):
    """A failing all_reduce must NOT propagate — better slow first iter than blocked training."""
    pytest.importorskip("torch")
    import logging

    import torch
    import torch.distributed as dist

    from axolotl.integrations.protrain.plugin import _eager_nccl_warmup

    chunk_manager = _make_chunk_manager(zero3_shard=False)

    def _raise(*_args, **_kwargs):
        raise RuntimeError("simulated NCCL init failure")

    patches = _patch_dist(initialized=True, world_size=2) + [
        patch.object(dist, "all_reduce", side_effect=_raise),
        patch.object(dist, "reduce_scatter_tensor", side_effect=_raise),
        patch.object(dist, "all_gather_into_tensor", side_effect=_raise),
    ]
    for p in patches:
        p.start()
    try:
        with caplog.at_level(logging.WARNING):
            _eager_nccl_warmup(chunk_manager, torch.device("cpu"))
    finally:
        for p in patches:
            p.stop()

    assert any(
        "eager NCCL warm-up failed" in record.message for record in caplog.records
    )
