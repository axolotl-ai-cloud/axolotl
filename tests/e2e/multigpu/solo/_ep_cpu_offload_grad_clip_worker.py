"""Four-rank Gloo regression for EP CPU-offloaded gradient clipping."""

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard

from axolotl.monkeypatch.accelerate.parallelism_config import _ep_aware_clip_grad_norm
from axolotl.utils.gradient_clipping import (
    clip_grad_norm_ep_local_shards_,
    ep_local_parameter_ids,
)


def _dtensor(local, mesh, placements, shape):
    return DTensor.from_local(
        local, mesh, placements, run_check=False, shape=torch.Size(shape), stride=(1,)
    )


def main():
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    mesh = DeviceMesh("cpu", torch.arange(4).reshape(2, 2), mesh_dim_names=("ep", "dp"))
    ep, dp = mesh.get_coordinate()
    dp_mesh = mesh["dp"]
    all_mesh = mesh._flatten("all")

    expert_values = ((3.0, 4.0), (12.0, 0.0))
    expert = torch.nn.Parameter(_dtensor(torch.zeros(1), dp_mesh, (Shard(0),), (2,)))
    expert.grad = _dtensor(
        torch.tensor([expert_values[ep][dp]]), dp_mesh, (Shard(0),), (2,)
    )
    nonexpert = torch.nn.Parameter(
        _dtensor(torch.zeros(1), all_mesh, (Shard(0),), (4,))
    )
    nonexpert.grad = _dtensor(
        torch.tensor([5.0 if rank == 0 else 0.0]), all_mesh, (Shard(0),), (4,)
    )
    replica = torch.nn.Parameter(
        _dtensor(torch.zeros(1), all_mesh, (Replicate(),), (1,))
    )
    replica.grad = _dtensor(torch.tensor([7.0]), all_mesh, (Replicate(),), (1,))
    plain_replica = torch.nn.Parameter(torch.zeros(1))
    plain_replica.grad = torch.tensor([6.0])
    plain_ep_local = torch.nn.Parameter(torch.zeros(1))
    plain_ep_local.grad = torch.tensor([1.0 if ep == 0 else 3.0])
    parameters = [expert, nonexpert, replica, plain_replica, plain_ep_local]

    expected = torch.tensor(289.0).sqrt()
    norm = clip_grad_norm_ep_local_shards_(
        parameters,
        expected.item() / 3,
        ep_local_parameters={id(expert), id(plain_ep_local)},
        global_mesh=mesh,
    )
    torch.testing.assert_close(norm, expected)
    coefficient = (expected / 3 / (expected + 1e-6)).item()
    for parameter in parameters:
        local = (
            parameter.grad.to_local()
            if isinstance(parameter.grad, DTensor)
            else parameter.grad
        )
        before = (
            expert_values[ep][dp]
            if parameter is expert
            else 5.0
            if parameter is nonexpert and rank == 0
            else 0.0
            if parameter is nonexpert
            else 7.0
            if parameter is replica
            else 6.0
            if parameter is plain_replica
            else 1.0
            if ep == 0
            else 3.0
        )
        torch.testing.assert_close(local, torch.tensor([before * coefficient]))

    expert.grad = _dtensor(
        torch.tensor([expert_values[ep][dp]]), dp_mesh, (Shard(0),), (2,)
    )
    nonexpert.grad = _dtensor(
        torch.tensor([5.0 if rank == 0 else 0.0]), all_mesh, (Shard(0),), (4,)
    )
    replica.grad = _dtensor(torch.tensor([7.0]), all_mesh, (Replicate(),), (1,))
    plain_replica.grad = torch.tensor([6.0])
    gpu_path_norm = _ep_aware_clip_grad_norm(
        [expert, nonexpert, replica, plain_replica],
        torch.tensor(279.0).sqrt().item() / 3,
        ep_local_parameters={id(expert)},
        global_mesh=mesh,
    )
    torch.testing.assert_close(gpu_path_norm, torch.tensor(279.0).sqrt())

    sparse = torch.nn.Parameter(torch.zeros(1))
    sparse.grad = torch.tensor([2.0]) if rank != 3 else None
    empty_norm = clip_grad_norm_ep_local_shards_(
        [] if rank == 3 else [sparse], 1.0, ep_local_parameters=set(), global_mesh=mesh
    )
    torch.testing.assert_close(empty_norm, torch.tensor(2.0))

    nested = torch.nn.Module()
    nested._ep_lora_sharded = True
    nested.adapter = torch.nn.Linear(1, 1, bias=False)
    container = torch.nn.Module()
    container.wrapper = nested
    assert id(nested.adapter.weight) in ep_local_parameter_ids(container)

    subgroup = torch.nn.Parameter(_dtensor(torch.zeros(1), dp_mesh, (Shard(0),), (2,)))
    subgroup.grad = _dtensor(
        torch.tensor([3.0 if dp == 0 else 4.0]), dp_mesh, (Shard(0),), (2,)
    )
    subgroup_norm = clip_grad_norm_ep_local_shards_(
        [subgroup], 10.0, ep_local_parameters=set(), global_mesh=mesh
    )
    torch.testing.assert_close(subgroup_norm, torch.tensor(5.0))
    subgroup.grad = _dtensor(
        torch.tensor([3.0 if dp == 0 else 4.0]), dp_mesh, (Shard(0),), (2,)
    )
    torch.testing.assert_close(
        _ep_aware_clip_grad_norm([subgroup], 10.0, global_mesh=mesh),
        torch.tensor(5.0),
    )
    dist.barrier()
    if rank == 0:
        print("EP_CPU_OFFLOAD_GRAD_CLIP_PASS", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
