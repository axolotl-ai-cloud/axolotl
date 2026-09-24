"""CPU/CUDA DDP and FSDP2 precision versus a serial global-batch reference."""

import copy
import os
from contextlib import nullcontext

import torch
import torch.distributed as dist
from test_lora_fp32_gradients import projection
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from axolotl.utils.lora_precision import upcast_lora_parameters


def main():
    device_type = os.environ.get("TEST_DEVICE", "cpu")
    if device_type == "cuda":
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl" if device_type == "cuda" else "gloo")
    device = torch.device(device_type)
    torch.manual_seed(3)
    model = nn.Sequential(projection(device))
    upcast_lora_parameters(model)
    reference = copy.deepcopy(model)
    fsdp = os.environ.get("TEST_FSDP2") == "1"
    if fsdp:
        from types import SimpleNamespace

        from accelerate import FullyShardedDataParallelPlugin
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import MixedPrecisionPolicy

        from axolotl.monkeypatch.accelerate.fsdp2 import fsdp2_prepare_model

        mesh = init_device_mesh(
            device_type, (dist.get_world_size(),), mesh_dim_names=("dp_shard",)
        )
        plugin = FullyShardedDataParallelPlugin(
            fsdp_version=2,
            cpu_ram_efficient_loading=False,
            mixed_precision_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16),
        )
        accelerator = SimpleNamespace(
            device=device,
            is_main_process=dist.get_rank() == 0,
            state=SimpleNamespace(
                fsdp_plugin=plugin,
                device_mesh=mesh,
                parallelism_config=SimpleNamespace(fsdp_dim_names=("dp_shard",)),
            ),
        )
        model._axolotl_lora_fp32_gradients = True
        fsdp2_prepare_model(accelerator, model)
        ddp = model
    else:
        ddp = DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()] if device_type == "cuda" else None,
        )
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    ref_opt = torch.optim.AdamW(reference.parameters(), lr=0.01)
    world = dist.get_world_size()
    for update in range(2):
        opt.zero_grad(set_to_none=True)
        ref_opt.zero_grad(set_to_none=True)
        for micro in range(3):
            generator = torch.Generator().manual_seed(100 * update + micro)
            x = torch.randn(world, 2, 32, generator=generator, dtype=torch.bfloat16).to(
                device
            )
            if fsdp:
                model.set_requires_gradient_sync(micro == 2)
            with ddp.no_sync() if micro < 2 and not fsdp else nullcontext():
                ddp(x[dist.get_rank()]).float().square().mean().div(3).backward()
            reference(x.flatten(0, 1)).float().square().mean().div(3).backward()
        for p, q in zip(model.parameters(), reference.parameters(), strict=True):
            if p.requires_grad:
                assert p.grad.dtype == torch.float32
                actual_grad = p.grad.full_tensor() if fsdp else p.grad
                torch.testing.assert_close(actual_grad, q.grad, rtol=1e-5, atol=1e-7)
        opt.step()
        ref_opt.step()
        for p, q in zip(model.parameters(), reference.parameters(), strict=True):
            actual = p.full_tensor() if fsdp else p
            torch.testing.assert_close(actual, q, rtol=1e-5, atol=1e-7)
    backend = "FSDP2" if fsdp else "DDP"
    print(
        f"LORA_FP32_{backend}_OK device={device_type} rank={dist.get_rank()} "
        f"world_size={world} updates=2",
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
