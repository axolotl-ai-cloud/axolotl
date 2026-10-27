"""Four-rank CPU regression checks for SinkGD matrix reduction groups."""

from datetime import timedelta
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor
from torch.distributed.tensor.placement_types import _StridedShard

from axolotl.utils.optimizers.sinkgd import (
    DistSinkGD,
    DistSinkGDMD,
    _matrix_shard_dim,
    _matrix_shard_group,
)


def check_matrix_shards(rank, rendezvous):
    """Check group membership and multi-step parity against a one-axis shard."""
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=60),
    )
    try:
        mesh = init_device_mesh("cpu", (2, 1, 2), mesh_dim_names=("dp", "cp", "tp"))
        flat = init_device_mesh("cpu", (4,), mesh_dim_names=("flat",))
        torch.manual_seed(42)
        full = torch.randn(4, 16, 12)
        for placements, axis in (
            ((Replicate(), Replicate(), Replicate()), None),
            ((Shard(0), Replicate(), Replicate()), None),
            ((Shard(0), Replicate(), Shard(1)), 2),
            ((Shard(2), Replicate(), Replicate()), 0),
            ((_StridedShard(1, split_factor=2), Replicate(), Replicate()), 0),
        ):
            param = distribute_tensor(full, mesh, placements)
            group = _matrix_shard_group(param)
            assert group is (None if axis is None else mesh.get_group(axis))

        for placements, dim in (
            ((Shard(1), Replicate(), Shard(1)), 1),
            ((_StridedShard(1, split_factor=2), Replicate(), Shard(1)), 1),
            ((Shard(2), Replicate(), Shard(2)), 2),
        ):
            for cls, spectral in (
                (DistSinkGD, False),
                (DistSinkGD, True),
                (DistSinkGDMD, True),
            ):
                params = [
                    torch.nn.Parameter(
                        distribute_tensor(full.clone(), mesh, placements)
                    ),
                    torch.nn.Parameter(
                        distribute_tensor(full.clone(), flat, [Shard(dim)])
                    ),
                ]
                assert _matrix_shard_dim(params[0]) == dim - full.ndim
                group = _matrix_shard_group(params[0])
                assert dist.get_process_group_ranks(group) == [0, 1, 2, 3]
                assert _matrix_shard_group(params[0]) is group
                # Exercise the collectives without compiling identical kernels in each worker.
                with patch("torch.compile", side_effect=lambda fn, **kwargs: fn):
                    opts = [
                        cls(
                            [{"params": [param], "use_sinkgd": True}],
                            lr=0.1,
                            sinkgd_lr_scale=1.0,
                            sinkgd_spectral_norm=spectral,
                            sinkgd_spectral_norm_iters=3,
                        )
                        for param in params
                    ]
                for _ in range(2):
                    grad = torch.randn_like(full)
                    params[0].grad = distribute_tensor(grad.clone(), mesh, placements)
                    params[1].grad = distribute_tensor(grad.clone(), flat, [Shard(dim)])
                    for opt in opts:
                        opt.step()
                    torch.testing.assert_close(
                        params[0].full_tensor(),
                        params[1].full_tensor(),
                        atol=2e-6,
                        rtol=2e-6,
                    )
    finally:
        dist.destroy_process_group()
