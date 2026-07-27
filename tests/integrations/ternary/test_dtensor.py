"""FSDP2/DTensor paths: bake, monitoring snapshots and the fused-kernel guard.

Run single-process (gloo, world size 1) so the DTensor code paths — `full_tensor`,
`distribute_tensor`, `to_local` and the write-through bake — are exercised in CI
without a multi-GPU box. The failure modes they cover are shape/identity bugs, not
shard-count bugs: rebinding `.data` is a silent no-op on a DTensor at any world size.
"""

import os

import pytest
import torch
import torch.distributed as dist
from torch import nn

from axolotl.integrations.ternary.modules import (
    TernaryLinear,
    as_dtensor,
    as_local,
)

pytest.importorskip("torch.distributed.tensor")

IN_FEATURES = 64
OUT_FEATURES = 32


@pytest.fixture(name="mesh", scope="module")
def fixture_mesh():
    """A single-rank gloo device mesh; the process group is torn down after."""
    from torch.distributed.device_mesh import init_device_mesh

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29733")
    created = not dist.is_initialized()
    if created:
        dist.init_process_group("gloo", rank=0, world_size=1)
    try:
        yield init_device_mesh("cpu", (1,))
    finally:
        if created and dist.is_initialized():
            dist.destroy_process_group()


def _sharded_module(mesh, **kwargs) -> TernaryLinear:
    from torch.distributed.tensor import Shard, distribute_tensor

    torch.manual_seed(0)
    module = TernaryLinear.from_linear(
        nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False), **kwargs
    )
    module.weight = nn.Parameter(
        distribute_tensor(module.weight.detach(), mesh, [Shard(0)])
    )
    return module


def test_as_dtensor_sees_through_a_parameter(mesh):
    module = _sharded_module(mesh)

    assert as_dtensor(module.weight.detach()) is not None
    assert as_dtensor(module.weight) is not None
    assert as_dtensor(torch.zeros(4)) is None
    assert as_local(module.weight.detach()).__class__ is torch.Tensor


def test_bake_writes_through_the_sharded_parameter(mesh):
    """Rebinding `.data` is a silent no-op on a DTensor; `copy_` is not."""
    module = _sharded_module(mesh)
    parameter = module.weight
    before = as_local(module.weight.detach()).clone()

    module._post_training(None, "q_proj")

    after = as_local(module.weight.detach())
    assert module.weight is parameter
    assert not torch.equal(after, before)
    assert set(torch.unique(after.abs()).tolist()) <= {
        0.0,
        float(after.abs().max()),
    }
    assert module.baked and module.is_baked()


def test_bake_matches_the_unsharded_result(mesh):
    """The per-tensor scale is global, so sharding must not change the baked values."""
    plain = TernaryLinear.from_linear(nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False))
    torch.manual_seed(0)
    sharded = _sharded_module(mesh)
    with torch.no_grad():
        plain.weight.copy_(as_local(sharded.weight.detach()))

    sharded._post_training(None, "q_proj")
    plain._post_training(None, "q_proj")

    assert torch.equal(as_local(sharded.weight.detach()), plain.weight.detach())


def test_code_snapshot_covers_the_local_shard(mesh):
    module = _sharded_module(mesh)

    packed = module.code_snapshot()

    assert packed.__class__ is torch.Tensor
    assert module.code_count() == as_local(module.weight.detach()).numel()
    assert packed.numel() == (module.code_count() + 3) // 4


def test_a_learnable_scale_is_shardable(mesh):
    """`fully_shard` rejects 0-dim parameters outright."""
    from torch.distributed.tensor import Shard, distribute_tensor

    module = _sharded_module(mesh, weight_scale="learnable")

    assert module.scale.shape == (1,)
    distributed = distribute_tensor(module.scale.detach(), mesh, [Shard(0)])
    assert distributed.to_local().numel() == 1


def test_the_fused_kernels_are_skipped_for_sharded_tensors(mesh):
    module = _sharded_module(mesh, fused=True)

    assert module._ops(module.weight) is None
    assert module._ops(module.weight.detach()) is None


def test_require_cuda_rejects_a_dtensor(mesh):
    from axolotl.integrations.ternary.kernels.triton import _common

    if not _common.HAVE_TRITON or not torch.cuda.is_available():
        pytest.skip("the fused kernels need triton and a CUDA device")

    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    cuda_mesh = init_device_mesh("cuda", (1,))
    sharded = distribute_tensor(torch.zeros(8, 8, device="cuda"), cuda_mesh, [Shard(0)])

    with pytest.raises(RuntimeError, match="need plain tensors"):
        _common.require_cuda(sharded)
