"""Two-rank NCCL EP clipping with CPU-resident DTensor shard gradients."""

import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard

from axolotl.utils.gradient_clipping import clip_grad_norm_ep_local_shards_


def _parameter(local, mesh, placement, shape):
    def distributed(value):
        return DTensor.from_local(
            value,
            mesh,
            (placement,),
            run_check=False,
            shape=torch.Size(shape),
            stride=(1,),
        ).to("cpu")

    parameter = torch.nn.Parameter(distributed(torch.zeros_like(local)))
    parameter.grad = distributed(local)
    assert parameter.grad.to_local().device.type == "cpu"
    return parameter


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    mesh = DeviceMesh(
        "cuda", torch.arange(2).reshape(2, 1), mesh_dim_names=("ep", "dp")
    )
    full_mesh = mesh._flatten("all")
    expert_mesh = mesh["dp"]
    for norm_type in (2.0, float("inf")):
        shared = _parameter(
            torch.tensor([3.0 if rank == 0 else 4.0]), full_mesh, Shard(0), (2,)
        )
        expert = _parameter(
            torch.tensor([5.0 if rank == 0 else 12.0]), expert_mesh, Shard(0), (1,)
        )
        replica = _parameter(torch.tensor([7.0]), full_mesh, Replicate(), (1,))
        parameters = [shared, expert, replica]
        before = [parameter.grad.to_local().clone() for parameter in parameters]
        expected = (
            torch.tensor(243.0).sqrt() if norm_type == 2.0 else torch.tensor(12.0)
        )
        norm = clip_grad_norm_ep_local_shards_(
            parameters,
            expected.item() / 2,
            norm_type=norm_type,
            ep_local_parameters={id(expert)},
            global_mesh=mesh,
        )
        assert norm.device.type == "cpu"
        torch.testing.assert_close(norm, expected)
        coefficient = expected / 2 / (expected + 1e-6)
        for parameter, original in zip(parameters, before, strict=True):
            assert parameter.grad.to_local().device.type == "cpu"
            torch.testing.assert_close(
                parameter.grad.to_local(), original * coefficient
            )

    local_expert = _parameter(torch.tensor([2.0]), expert_mesh, Shard(0), (1,))
    norm = clip_grad_norm_ep_local_shards_(
        [local_expert] if rank == 0 else [],
        1.0,
        ep_local_parameters={id(local_expert)},
        global_mesh=mesh,
    )
    torch.testing.assert_close(norm, torch.tensor(2.0))
    if rank == 0:
        torch.testing.assert_close(
            local_expert.grad.to_local(), torch.tensor([2.0 / (2.0 + 1e-6)])
        )
    empty = clip_grad_norm_ep_local_shards_(
        [], 1.0, ep_local_parameters=set(), global_mesh=mesh
    )
    torch.testing.assert_close(empty, torch.tensor(0.0))
    dist.barrier()
    if rank == 0:
        print("EP_CPU_OFFLOAD_NCCL_CLIP_PASS", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
