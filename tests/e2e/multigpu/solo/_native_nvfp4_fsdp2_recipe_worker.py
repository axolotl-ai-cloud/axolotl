"""Actual DTensor scatter coverage for native NVFP4 FSDP2 reconstruction."""

import datetime
import faulthandler
import os
import sys
import traceback

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard
from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

from axolotl.integrations.kernels.libs.scattermoe_lora.nvfp4_fsdp import (
    patch_nvfp4_fsdp,
)
from axolotl.monkeypatch.accelerate.fsdp2 import _broadcast_nvfp4_param


def _tensor(shape, dtype, start=0):
    return (
        torch.arange(
            start,
            start + torch.tensor(shape).prod().item(),
            device="cuda",
            dtype=torch.float32,
        )
        .reshape(shape)
        .to(dtype)
    )


def _full(per_expert):
    qdata = _tensor((2, 16, 8), torch.uint8)
    scale = (_tensor((2, 16, 1), torch.float32, 1) / 32).to(torch.float8_e4m3fn)
    pts = (
        _tensor((2, 1, 1), torch.float32, 3)
        if per_expert
        else torch.tensor(0.25, device="cuda")
    )
    act_pts = (
        _tensor((2, 1, 1), torch.float32, 7)
        if per_expert
        else torch.tensor(0.5, device="cuda")
    )
    kwargs = dict(
        act_per_tensor_scale=act_pts,
        is_swizzled_scales=False,
        use_triton_kernel=False,
        act_quant_kwargs={"recipe": "gate"},
    )
    return NVFP4Tensor(qdata, scale, 16, torch.float32, per_tensor_scale=pts, **kwargs)


def _meta_local(per_expert):
    qdata = torch.empty((1, 16, 8), device="meta", dtype=torch.uint8)
    scale = torch.empty((1, 16, 1), device="meta", dtype=torch.float8_e4m3fn)
    pts = (
        torch.empty((1, 1, 1), device="meta")
        if per_expert
        else torch.empty((), device="meta")
    )
    act_pts = (
        torch.empty((1, 1, 1), device="meta")
        if per_expert
        else torch.empty((), device="meta")
    )
    return NVFP4Tensor(
        qdata,
        scale,
        16,
        torch.float32,
        per_tensor_scale=pts,
        act_per_tensor_scale=act_pts,
        is_swizzled_scales=False,
        use_triton_kernel=False,
        act_quant_kwargs={"recipe": "gate"},
    )


def _check(per_expert):
    rank = dist.get_rank()
    print(f"rank={rank} check={per_expert} begin", flush=True)
    mesh = init_device_mesh("cuda", (2,))
    meta = DTensor.from_local(
        _meta_local(per_expert), mesh, (Shard(0),), run_check=False
    )
    local_meta = meta._local_tensor
    assert local_meta.orig_dtype == torch.float32
    assert local_meta.act_quant_kwargs == {"recipe": "gate"}
    assert local_meta.per_tensor_scale.shape == (() if not per_expert else (1, 1, 1))
    assert local_meta.act_per_tensor_scale.shape == (
        () if not per_expert else (1, 1, 1)
    )
    full = _full(per_expert) if rank == 0 else None
    print(f"rank={rank} check={per_expert} before broadcast", flush=True)
    result = _broadcast_nvfp4_param(
        meta, full, rank == 0, "cuda", NVFP4Tensor
    )._local_tensor
    print(f"rank={rank} check={per_expert} after broadcast", flush=True)
    expected = _full(per_expert)
    assert torch.equal(result.qdata, expected.qdata[rank : rank + 1])
    assert torch.equal(result.scale, expected.scale[rank : rank + 1])
    if per_expert:
        assert torch.equal(
            result.per_tensor_scale, expected.per_tensor_scale[rank : rank + 1]
        )
        assert torch.equal(
            result.act_per_tensor_scale, expected.act_per_tensor_scale[rank : rank + 1]
        )
    else:
        assert torch.equal(result.per_tensor_scale, expected.per_tensor_scale)
        assert torch.equal(result.act_per_tensor_scale, expected.act_per_tensor_scale)
    assert result.orig_dtype == expected.orig_dtype
    assert result.act_quant_kwargs == expected.act_quant_kwargs
    print(f"rank={rank} check={per_expert} done", flush=True)


if __name__ == "__main__":
    faulthandler.dump_traceback_later(60, repeat=True)
    print(f"rank={os.environ['RANK']} before set_device", flush=True)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    print(f"rank={os.environ['RANK']} before patch", flush=True)
    patch_nvfp4_fsdp()
    print(f"rank={os.environ['RANK']} before init", flush=True)
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=90))
    print(f"rank={dist.get_rank()} after init", flush=True)
    try:
        _check(False)
        _check(True)
        dist.barrier()
        if dist.get_rank() == 0:
            print("NATIVE_NVFP4_FSDP2_RECIPE_PASS", flush=True)
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        os._exit(1)
    else:
        dist.destroy_process_group()
