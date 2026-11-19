# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Micro-benchmark: native Blackwell hardware-cvt to_nvfp4 vs torchao (run with CUDA_DEVICE_ORDER=PCI_BUS_ID)."""

from __future__ import annotations

import time

import torch
from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

import axolotl.integrations.kernels.libs.scattermoe_lora.nvfp4_native_quant as nq


def _blackwell_device() -> int:
    for i in range(torch.cuda.device_count()):
        if torch.cuda.get_device_properties(i).major in (10, 12):
            return i
    raise SystemExit("no Blackwell (cc major 10/12) GPU found")


def _bench(fn, iters: int = 50, warmup: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e3  # ms


def main() -> None:
    dev_idx = _blackwell_device()
    torch.cuda.set_device(dev_idx)
    dev = f"cuda:{dev_idx}"
    name = torch.cuda.get_device_properties(dev_idx).name
    shape = (131072, 2048)
    x = torch.randn(*shape, device=dev, dtype=torch.bfloat16).contiguous()

    print(f"device: {name} (cuda:{dev_idx}); shape {shape} bf16")
    for swz in (True, False):
        nq.uninstall_native_nvfp4()
        ao = _bench(
            lambda s=swz: NVFP4Tensor.to_nvfp4(x, block_size=16, is_swizzled_scales=s)
        )
        if (
            not nq.install_native_nvfp4()
            or not nq.is_blackwell_native_nvfp4_available()
        ):
            raise SystemExit(
                "native NVFP4 path unavailable; benchmark would measure torchao fallback"
            )
        nat = _bench(
            lambda s=swz: NVFP4Tensor.to_nvfp4(x, block_size=16, is_swizzled_scales=s)
        )
        nq.uninstall_native_nvfp4()
        tag = "swizzled (PTQ)" if swz else "non-swizzled (MoE)"
        print(
            f"  {tag:20s}  torchao {ao:7.3f} ms | native {nat:7.3f} ms | {ao / nat:5.2f}x"
        )


if __name__ == "__main__":
    main()
