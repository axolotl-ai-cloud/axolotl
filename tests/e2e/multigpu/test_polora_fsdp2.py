# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0
"""PoLoRA gather-compute-scatter correctness under a 2-rank device mesh.

Upstream polora is single-device only: it builds ``r x r`` Gram matrices and global
spectral norms from whole factors. Under FSDP2 the factors arrive as DTensors sharded
on dim 0, which for ``A`` is the LoRA rank itself, so the factory gathers them, runs the
dense update redundantly on every rank, and scatters the local shard back.

These tests verify that the sharded path reproduces the single-device update exactly,
for both the rank-sharded ``A`` and the output-sharded ``B``, and that ranks stay in
lockstep (the update must be bit-identical across ranks, since each writes its own
shard of a value every rank computed independently).

Run with::

    torchrun --nproc-per-node=2 -m pytest tests/e2e/multigpu/test_polora_fsdp2.py

On a 1-GPU executor the tests skip with a clear reason.
"""

import os

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

polora = pytest.importorskip("polora", reason="polora is an optional dependency")

from axolotl.utils.optimizers.polora import (  # noqa: E402
    _gather_into,
    _gathered_pairs,
    _polora_cls,
    _scatter_from,
)

_TORCHRUN_LOCAL_RANK = os.environ.get("LOCAL_RANK")
_TORCHRUN_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    pytest.mark.skipif(
        torch.cuda.device_count() < 2, reason="Need >=2 GPUs for FSDP2 multi-rank tests"
    ),
    pytest.mark.skipif(
        _TORCHRUN_LOCAL_RANK is None or _TORCHRUN_WORLD_SIZE < 2,
        reason="Launch via `torchrun --nproc-per-node=2 -m pytest <file>`",
    ),
]

R, D_IN, D_OUT = 16, 64, 32


@pytest.fixture(scope="module")
def mesh():
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    dm = init_device_mesh(
        "cuda", (dist.get_world_size(),), mesh_dim_names=("dp_shard",)
    )
    yield dm
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _factors(n_pairs=2):
    """Identical-across-ranks full factors and grads, seeded so every rank agrees."""
    torch.manual_seed(0)
    pairs, grads = [], []
    for _ in range(n_pairs):
        A = torch.randn(R, D_IN, device="cuda")
        B = torch.randn(D_OUT, R, device="cuda")
        pairs.append((A, B))
        grads.append((torch.randn_like(A), torch.randn_like(B)))
    return pairs, grads


def _single_device_reference(pairs, grads, steps, lr):
    dense = [
        (torch.nn.Parameter(A.clone()), torch.nn.Parameter(B.clone())) for A, B in pairs
    ]
    opt = _polora_cls()(pairs=dense, lr=lr)
    for _ in range(steps):
        for (A, B), (gA, gB) in zip(dense, grads, strict=True):
            A.grad = gA.clone()
            B.grad = gB.clone()
        opt.step()
    return dense


def _sharded_run(mesh, pairs, grads, steps, lr):
    """Same update, with A rank-sharded and B output-sharded across the mesh."""
    real = [
        (
            torch.nn.Parameter(distribute_tensor(A.clone(), mesh, [Shard(0)])),
            torch.nn.Parameter(distribute_tensor(B.clone(), mesh, [Shard(0)])),
        )
        for A, B in pairs
    ]
    stand_ins = _gathered_pairs(real)
    opt = _polora_cls()(pairs=stand_ins, lr=lr)
    opt._sharded_pairs = real  # pylint: disable=protected-access

    for _ in range(steps):
        for (A, B), (gA, gB) in zip(real, grads, strict=True):
            A.grad = distribute_tensor(gA.clone(), mesh, [Shard(0)])
            B.grad = distribute_tensor(gB.clone(), mesh, [Shard(0)])
        opt.step()
    return real


@pytest.mark.parametrize("steps", [1, 3])
def test_sharded_matches_single_device(mesh, steps):
    """The gathered update must reproduce the dense one bit-for-bit after scatter."""
    pairs, grads = _factors()
    expected = _single_device_reference(pairs, grads, steps, lr=1e-2)
    got = _sharded_run(mesh, pairs, grads, steps, lr=1e-2)

    for idx, ((eA, eB), (gA, gB)) in enumerate(zip(expected, got, strict=True)):
        torch.testing.assert_close(
            gA.full_tensor(), eA.data, msg=f"pair {idx} factor A diverged"
        )
        torch.testing.assert_close(
            gB.full_tensor(), eB.data, msg=f"pair {idx} factor B diverged"
        )


def test_ranks_stay_in_lockstep(mesh):
    """Each rank writes its own shard of an update every rank computed independently,
    so any cross-rank drift silently corrupts the weights rather than erroring."""
    pairs, grads = _factors()
    real = _sharded_run(mesh, pairs, grads, steps=3, lr=1e-2)

    for idx, (A, B) in enumerate(real):
        for name, param in (("A", A), ("B", B)):
            full = param.full_tensor()
            broadcast = full.clone()
            dist.broadcast(broadcast, src=0)
            torch.testing.assert_close(
                full, broadcast, msg=f"pair {idx} factor {name} differs across ranks"
            )


def test_gather_scatter_round_trip_is_lossless(mesh):
    """Scatter-then-gather must be the identity, independent of the optimizer math."""
    pairs, _ = _factors(n_pairs=1)
    real = [
        (
            torch.nn.Parameter(distribute_tensor(A.clone(), mesh, [Shard(0)])),
            torch.nn.Parameter(distribute_tensor(B.clone(), mesh, [Shard(0)])),
        )
        for A, B in pairs
    ]
    stand_ins = _gathered_pairs(real)
    for (A, B), (gA, gB) in zip(real, [(pairs[0][0], pairs[0][1])], strict=True):
        A.grad = distribute_tensor(torch.randn_like(gA), mesh, [Shard(0)])
        B.grad = distribute_tensor(torch.randn_like(gB), mesh, [Shard(0)])

    _gather_into(real, stand_ins)
    torch.testing.assert_close(stand_ins[0][0].data, pairs[0][0])
    torch.testing.assert_close(stand_ins[0][1].data, pairs[0][1])

    stand_ins[0][0].data.mul_(2.0)
    _scatter_from(stand_ins, real)
    torch.testing.assert_close(real[0][0].full_tensor(), pairs[0][0] * 2.0)
