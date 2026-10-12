"""Two-rank NCCL check of per-tensor ratio clipping on FSDP2, with and without CPU offload."""

import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard
from torch.distributed.tensor import DTensor, distribute_tensor

from axolotl.core.trainers.mixins.grad_norm_guard import (
    GRAD_NORM_EMA_KEY,
    GradNormGuardMixin,
)

RATIO, BETA = 1.5, 0.8


class _Base:
    def _get_grad_norm(self, model, grad_norm=None):
        return grad_norm


class _Trainer(GradNormGuardMixin, _Base):
    def __init__(self, optimizer):
        self.args = SimpleNamespace(
            step_outlier_grad_norm_zscore=None,
            step_outlier_loss_zscore=None,
            grad_clip_norm_ratio=RATIO,
            grad_clip_norm_ratio_beta=BETA,
        )
        self.state = SimpleNamespace(global_step=0)
        self.optimizer = optimizer


def _run(mesh, offload: bool):
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 5), torch.nn.Linear(5, 1)).cuda()
    kwargs = {"offload_policy": CPUOffloadPolicy()} if offload else {}
    for layer in model:
        fully_shard(layer, mesh=mesh, **kwargs)
    fully_shard(model, mesh=mesh, **kwargs)
    named = dict(model.named_parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = _Trainer(optimizer)
    ema: dict[str, torch.Tensor] = {}
    for step in range(5):
        generator = torch.Generator().manual_seed(step)
        grads = {
            name: torch.randn(param.shape, generator=generator)
            * (40.0 if step == 3 and name == "0.weight" else 1.0)
            for name, param in named.items()
        }
        for name, param in named.items():
            grad = distribute_tensor(
                grads[name].cuda(), mesh, param.placements, src_data_rank=None
            )
            # CPU offload keeps the local shards (and their gradients) on the CPU
            param.grad = grad.to("cpu") if offload else grad
            assert param.grad.to_local().device.type == ("cpu" if offload else "cuda")
        trainer._get_grad_norm(model, torch.tensor(1.0))
        for name, param in named.items():
            norm = grads[name].double().norm()
            average = ema.get(name, norm)
            coef = torch.clamp(RATIO * average / (norm + 1e-6), max=1.0)
            ema[name] = average + (norm * coef - average) * (1 - BETA)
            # gather on the GPU: an NCCL-only group cannot gather offloaded CPU shards
            gathered = DTensor.from_local(
                param.grad.to_local().cuda(),
                mesh,
                param.grad.placements,
                shape=param.grad.shape,
                stride=param.grad.stride(),
            ).full_tensor()
            torch.testing.assert_close(
                gathered.cpu().double(), grads[name].double() * coef
            )
            if offload:
                assert param.grad.to_local().device.type == "cpu"
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    for name, param in named.items():
        torch.testing.assert_close(
            optimizer.state[param][GRAD_NORM_EMA_KEY].cpu().double(), ema[name]
        )


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    mesh = init_device_mesh("cuda", (dist.get_world_size(),))
    for offload in (False, True):
        _run(mesh, offload)
    dist.barrier()
    if dist.get_rank() == 0:
        print("GRAD_NORM_RATIO_FSDP2_NCCL_PASS", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
