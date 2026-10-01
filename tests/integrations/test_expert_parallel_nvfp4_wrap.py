"""EP with no shard axis FSDP-wraps pre-quantized (NVFP4) experts on the per-rank mesh too:
the frozen NVFP4 weight round-trips through FSDP2's single-rank unshard and the trainable
expert params in the same unit reduce like any other expert (CPU, gloo)."""

import os
import queue as queue_mod
import socket
import time
import traceback

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytest.importorskip(
    "torchao.prototype.mx_formats.nvfp4_tensor", reason="torchao required"
)

WORLD = 2
E, N, K = 2, 8, 32


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_spawned(target, world_size=WORLD, timeout=240, device="cpu"):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(target=target, args=(rank, world_size, port, q, device))
        for rank in range(world_size)
    ]
    for p in procs:
        p.start()
    results = {}
    deadline = time.monotonic() + timeout
    while len(results) < world_size and time.monotonic() < deadline:
        try:
            rank, res = q.get(timeout=5)
            results[rank] = res
        except queue_mod.Empty:
            if all(not p.is_alive() for p in procs):
                break
    for p in procs:
        p.join(timeout=20)
        if p.is_alive():
            p.kill()
    errors = {r: res for r, res in results.items() if isinstance(res, str)}
    assert not errors, "\n".join(f"rank {r}:\n{e}" for r, e in errors.items())
    assert len(results) == world_size, results
    return results


class _Experts(torch.nn.Module):
    def forward(self, x):
        w = self.gate_up_proj.dequantize(torch.float32)[0]
        return (x @ w) @ self.down_proj[0]


def _nvfp4_worker(rank, world_size, port, q, device="cpu"):
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        if device == "cuda":
            torch.cuda.set_device(rank)
        dist.init_process_group(
            "nccl" if device == "cuda" else "gloo", rank=rank, world_size=world_size
        )
        torch.set_default_device(device)
        from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

        from axolotl.integrations.expert_parallel.plugin import (
            ExpertParallelPlugin,
            per_rank_expert_mesh,
        )
        from axolotl.integrations.kernels.libs.scattermoe_lora.nvfp4_fsdp import (
            patch_nvfp4_fsdp,
        )

        patch_nvfp4_fsdp()
        torch.manual_seed(rank)
        qdata = torch.randint(0, 255, (E, N, K // 2), dtype=torch.uint8)
        scale = (torch.rand(E, N, K // 16) * 0.5 + 0.5).to(torch.float8_e4m3fn)
        nv = NVFP4Tensor(qdata, scale, 16, torch.bfloat16, torch.tensor(0.37))
        w_ref = nv.dequantize(torch.float32).clone()

        experts = _Experts()
        experts.gate_up_proj = torch.nn.Parameter(nv, requires_grad=False)
        experts.down_proj = torch.nn.Parameter(torch.full((E, K, 4), 0.5))
        experts.num_experts = experts.num_local_experts = E
        experts.num_experts_global = E * world_size
        root = torch.nn.Module()
        root.experts = experts
        mesh = per_rank_expert_mesh(None, device)
        ExpertParallelPlugin.fully_shard_experts(
            root, mesh, {"reshard_after_forward": True}
        )

        x = torch.full((3, N), float(rank + 1))
        y = experts(x)
        y.sum().backward()
        y2 = experts(x)  # second unshard reuses the packed buffer in place

        down_ref = torch.full((E, K, 4), 0.5, requires_grad=True)
        ((x @ w_ref[0]) @ down_ref[0]).sum().backward()

        param = experts.gate_up_proj
        q.put(
            (
                rank,
                {
                    "fwd": (y - (x @ w_ref[0]) @ down_ref[0]).abs().max().item(),
                    "fwd_again": (y2 - y).abs().max().item(),
                    "param_cls": type(param).__name__,
                    "local_cls": type(param._local_tensor).__name__,
                    "local_ok": torch.equal(
                        param._local_tensor.dequantize(torch.float32), w_ref
                    ),
                    "d_down": (
                        experts.down_proj.grad.full_tensor()
                        - down_ref.grad / world_size
                    )
                    .abs()
                    .max()
                    .item(),
                },
            )
        )
    except Exception:  # pylint: disable=broad-except
        q.put((rank, traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _check(results, tol=1e-5):
    for rank in range(WORLD):
        res = results[rank]
        assert isinstance(res, dict), res
        assert res["param_cls"] == "DTensor", res
        assert res["local_cls"] == "NVFP4Tensor", res
        assert res["local_ok"], res
        for metric in ("fwd", "fwd_again", "d_down"):
            assert res[metric] <= tol, f"rank {rank} {metric}: {res}"


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < WORLD,
    reason="needs two CUDA devices",
)
def test_pure_ep_wraps_nvfp4_experts_per_rank_cuda():
    _check(_run_spawned(_nvfp4_worker, device="cuda"), tol=1e-3)


def test_pure_ep_wraps_nvfp4_experts_per_rank():
    _check(_run_spawned(_nvfp4_worker))
