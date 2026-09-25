"""Two-rank NCCL regression for CPU-offloaded DTensor gradient clipping."""

import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard

from axolotl.utils.gradient_clipping import (
    clip_grad_norm_local_shards_,
    get_grad_norm_local_shards_,
)


def _cpu_local_dtensor(local, mesh, shape):
    return DTensor.from_local(
        local.cuda(),
        mesh,
        [Shard(0)],
        run_check=False,
        shape=torch.Size(shape),
        stride=(1,),
    ).to("cpu")


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    mesh = DeviceMesh("cuda", torch.arange(dist.get_world_size()))

    shard = _cpu_local_dtensor(
        torch.tensor([3.0, 4.0]) if rank == 0 else torch.tensor([12.0]),
        mesh,
        (3,),
    )
    empty_shard = _cpu_local_dtensor(
        torch.tensor([8.0]) if rank == 0 else torch.empty(0), mesh, (1,)
    )
    replica = DTensor.from_local(
        torch.tensor([5.0], device="cuda"), mesh, [Replicate()], run_check=False
    )
    replica = replica.to("cpu")
    plain = torch.tensor([6.0])
    parameters = [
        SimpleNamespace(grad=shard),
        SimpleNamespace(grad=empty_shard),
        SimpleNamespace(grad=replica),
        SimpleNamespace(grad=plain),
    ]

    reference = torch.tensor((3**2 + 4**2 + 12**2 + 8**2 + 5**2 + 6**2) ** 0.5)
    before_telemetry = [
        parameter.grad.to_local().clone()
        if isinstance(parameter.grad, DTensor)
        else parameter.grad.clone()
        for parameter in parameters
    ]
    telemetry = get_grad_norm_local_shards_(parameters)
    torch.testing.assert_close(telemetry.cpu(), reference)
    for parameter, before in zip(parameters, before_telemetry, strict=True):
        actual = (
            parameter.grad.to_local()
            if isinstance(parameter.grad, DTensor)
            else parameter.grad
        )
        torch.testing.assert_close(actual, before)
    norm = clip_grad_norm_local_shards_(parameters, reference.item() / 2)
    torch.testing.assert_close(norm.cpu(), reference)
    for parameter in parameters:
        assert (
            parameter.grad.to_local().device.type
            if isinstance(parameter.grad, DTensor)
            else parameter.grad.device.type
        ) == "cpu"
    expected_shard = torch.tensor([1.5, 2.0]) if rank == 0 else torch.tensor([6.0])
    expected_empty = torch.tensor([4.0]) if rank == 0 else torch.empty(0)
    torch.testing.assert_close(shard.to_local(), expected_shard)
    torch.testing.assert_close(empty_shard.to_local(), expected_empty)
    torch.testing.assert_close(replica.to_local(), torch.tensor([2.5]))
    torch.testing.assert_close(plain, torch.tensor([3.0]))

    for norm_type in (2.0, float("inf")):

        def parameter(local, gradient):
            value = _cpu_local_dtensor(torch.zeros_like(local), mesh, (2,))
            result = torch.nn.Parameter(value)
            if gradient is not None:
                result.grad = _cpu_local_dtensor(gradient, mesh, (2,))
            return result

        before = parameter(
            torch.tensor([0.0]), torch.tensor([3.0 if rank == 0 else 4.0])
        )
        missing = parameter(
            torch.tensor([0.0]), None if rank == 0 else torch.tensor([5.0])
        )
        after = parameter(
            torch.tensor([0.0]), torch.tensor([6.0]) if rank == 0 else None
        )
        parameters = [before, missing, after]
        pointers = [
            parameter.to_local().untyped_storage().data_ptr()
            for parameter in parameters
        ]
        expected = torch.tensor(86.0).sqrt() if norm_type == 2.0 else torch.tensor(6.0)
        norm = clip_grad_norm_local_shards_(parameters, expected.item() / 2, norm_type)
        torch.testing.assert_close(norm.cpu(), expected)
        coefficient = expected / 2 / (expected + 1e-6)
        expected_before = torch.tensor([3.0 if rank == 0 else 4.0]) * coefficient
        torch.testing.assert_close(before.grad.to_local(), expected_before)
        if rank == 0:
            assert missing.grad is None
            torch.testing.assert_close(
                after.grad.to_local(), torch.tensor([6.0]) * coefficient
            )
        else:
            torch.testing.assert_close(
                missing.grad.to_local(), torch.tensor([5.0]) * coefficient
            )
            assert after.grad is None
        assert pointers == [
            parameter.to_local().untyped_storage().data_ptr()
            for parameter in parameters
        ]

    inf_shard = _cpu_local_dtensor(
        torch.tensor([8.0]) if rank == 0 else torch.empty(0), mesh, (1,)
    )
    inf_plain = torch.tensor([6.0])
    inf_norm = clip_grad_norm_local_shards_(
        [SimpleNamespace(grad=inf_shard), SimpleNamespace(grad=inf_plain)],
        4.0,
        float("inf"),
    )
    torch.testing.assert_close(inf_norm.cpu(), torch.tensor(8.0))
    torch.testing.assert_close(
        inf_shard.to_local(), torch.tensor([4.0]) if rank == 0 else torch.empty(0)
    )
    torch.testing.assert_close(inf_plain, torch.tensor([3.0]))

    dist.barrier()
    if rank == 0:
        print("FSDP2_CPU_OFFLOAD_GRAD_CLIP_PASS", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
