# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0
"""DistSinkGD sharded spectral-norm correctness under a 2-rank device mesh.

Verifies that the sharded power iteration (Gram-matrix matvec + one vector all-reduce per
step, matrix never gathered) reproduces the single-device full-matrix spectral rescale, for
both row- and column-sharded weights, and that the persisted power-iteration vector
round-trips through the optimizer state dict.

Run with::

    torchrun --nproc-per-node=2 -m pytest tests/e2e/multigpu/test_sinkgd_fsdp2.py

On a 1-GPU executor the tests skip with a clear reason.
"""

import os

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from axolotl.utils.optimizers.sinkgd import DistSinkGD, DistSinkGDMD, sr_sinkhorn

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


def _full_reference(w_full, grad_full, shard_dim, target_mode, sn_iters, lr_alpha):
    """Full-matrix equivalent of the sharded step: full SR-Sinkhorn -> Gram power iteration
    (seed-0 init, matching DistSinkGD._dist_specnorm_vec) -> operator-norm rescale -> apply."""
    m, n = w_full.shape
    u_mat = sr_sinkhorn(grad_full.float(), 5, 1e-8)
    vec_len = n if shard_dim == -2 else m
    g = torch.Generator(device=w_full.device).manual_seed(0)
    vec = torch.randn(vec_len, generator=g, device=w_full.device, dtype=torch.float32)
    vec = vec / vec.norm().clamp_min(1e-12)
    nrm = None
    for _ in range(sn_iters):
        if shard_dim == -2:
            up = u_mat.t() @ (u_mat @ vec)
        else:
            up = u_mat @ (u_mat.t() @ vec)
        nrm = up.norm().clamp_min(1e-8)
        vec = up / nrm
    sigma = nrm.sqrt()
    target = 1.0 if target_mode == "unit" else (m / n) ** 0.5
    return w_full.float() - lr_alpha * (u_mat * (target / sigma))


@pytest.mark.parametrize("shard_dim,dim", [(-2, 0), (-1, 1)])
@pytest.mark.parametrize("target_mode", ["unit", "muon"])
def test_sharded_spectral_matches_full(mesh, shard_dim, dim, target_mode):
    torch.manual_seed(0)
    m, n = 128, 96
    w_full = torch.randn(m, n, device="cuda", dtype=torch.float32)
    grad_full = torch.randn(m, n, device="cuda", dtype=torch.float32)
    dist.broadcast(w_full, 0)
    dist.broadcast(grad_full, 0)

    w = distribute_tensor(w_full.clone(), mesh, [Shard(dim)])
    w = torch.nn.Parameter(w)
    w.grad = distribute_tensor(grad_full.clone(), mesh, [Shard(dim)])

    sn_iters = 3
    opt = DistSinkGD(
        [{"params": [w], "use_sinkgd": True, "weight_decay": 0.0}],
        lr=1.0,
        sinkgd_lr_scale=1.0,
        sinkgd_spectral_norm=True,
        sinkgd_spectral_norm_iters=sn_iters,
        sinkgd_spectral_target=target_mode,
        process_group=mesh["dp_shard"].get_group(),
    )
    opt.step()

    ref = _full_reference(
        w_full, grad_full, shard_dim, target_mode, sn_iters, lr_alpha=1.0
    )
    got = w.detach().full_tensor()
    torch.testing.assert_close(got, ref, rtol=2e-3, atol=2e-3)

    # the persisted power-iteration vector lives on the unsharded axis and round-trips
    u = opt.state[w]["specnorm_u"]
    assert u.shape[-1] == (n if shard_dim == -2 else m)
    sd = opt.state_dict()
    opt2 = DistSinkGD(
        [{"params": [w], "use_sinkgd": True, "weight_decay": 0.0}],
        lr=1.0,
        sinkgd_lr_scale=1.0,
        sinkgd_spectral_norm=True,
        sinkgd_spectral_norm_iters=sn_iters,
        sinkgd_spectral_target=target_mode,
        process_group=mesh["dp_shard"].get_group(),
    )
    opt2.load_state_dict(sd)
    torch.testing.assert_close(opt2.state[w]["specnorm_u"], u)


@pytest.mark.parametrize("dim", [0, 1])
def test_dist_md_sphere_matches_full_and_stays_on_sphere(mesh, dim):
    """DistSinkGDMD (A+B) on a sharded weight keeps the GLOBAL weight on its enable-time
    Frobenius sphere across steps; sphere radius + power-iter vector round-trip through the
    state dict. (Fused-vs-compiled MD equivalence is covered by
    test_dist_fused_matches_compiled.)"""
    torch.manual_seed(0)
    m, n = 128, 96
    w_full = torch.randn(m, n, device="cuda", dtype=torch.float32)
    dist.broadcast(w_full, 0)
    tn0 = w_full.norm().item()

    w = torch.nn.Parameter(distribute_tensor(w_full.clone(), mesh, [Shard(dim)]))
    opt = DistSinkGDMD(
        [{"params": [w], "use_sinkgd": True, "weight_decay": 0.0}],
        lr=1.0,
        sinkgd_lr_scale=0.1,
        sinkgd_spectral_norm_iters=3,
        process_group=mesh["dp_shard"].get_group(),
    )
    for _ in range(4):
        w.grad = distribute_tensor(
            torch.randn(m, n, device="cuda", dtype=torch.float32), mesh, [Shard(dim)]
        )
        opt.step()

    # global weight stays on the enable-time Frobenius sphere
    assert w.detach().full_tensor().norm().item() == pytest.approx(tn0, rel=1e-3)
    # sphere radius + power-iteration vector are persisted and round-trip
    assert "md_target_norm" in opt.state[w] and "specnorm_u" in opt.state[w]
    sd = opt.state_dict()
    opt2 = DistSinkGDMD(
        [{"params": [w], "use_sinkgd": True, "weight_decay": 0.0}],
        lr=1.0,
        sinkgd_lr_scale=0.1,
        sinkgd_spectral_norm_iters=3,
        process_group=mesh["dp_shard"].get_group(),
    )
    opt2.load_state_dict(sd)
    torch.testing.assert_close(
        opt2.state[w]["md_target_norm"], opt.state[w]["md_target_norm"]
    )


@pytest.mark.parametrize("mode", ["base", "spec", "md"])
@pytest.mark.parametrize("wide", [False, True])
def test_dist_fused_matches_compiled(mesh, mode, wide):
    """sinkgd_fused_kernel=True on rows-sharded DTensors matches the compiled dist path for
    all modes, in both the tall and (row-starved) wide kernel regimes."""
    torch.manual_seed(0)
    m, n = (128, 4096) if wide else (256, 96)
    w_full = torch.randn(m, n, device="cuda", dtype=torch.float32)
    g_full = torch.randn(m, n, device="cuda", dtype=torch.float32)
    dist.broadcast(w_full, 0)
    dist.broadcast(g_full, 0)

    results = []
    for fused in (False, True):
        w = torch.nn.Parameter(distribute_tensor(w_full.clone(), mesh, [Shard(0)]))
        # sn_iters=3: the compiled dist path estimates sigma via Gram power iteration,
        # the fused path via the normalized two-matvec form — identical at convergence,
        # transiently different from a cold start, so compare near convergence.
        kw = dict(
            lr=1e-2,
            sinkgd_lr_scale=0.5,
            sinkgd_fused_kernel=fused,
            sinkgd_spectral_norm_iters=3,
            process_group=mesh["dp_shard"].get_group(),
        )
        if mode == "spec":
            kw.update(sinkgd_spectral_norm=True, sinkgd_spectral_target="muon")
        cls = DistSinkGDMD if mode == "md" else DistSinkGD
        opt = cls([{"params": [w], "use_sinkgd": True, "weight_decay": 0.0}], **kw)
        for _ in range(3):
            w.grad = distribute_tensor(g_full.clone(), mesh, [Shard(0)])
            opt.step()
        results.append(w.detach().full_tensor())
    rel = (results[0] - results[1]).abs().max() / results[0].abs().max()
    assert rel.item() < 3e-2, f"{mode} wide={wide}: rel={rel.item():.2e}"
