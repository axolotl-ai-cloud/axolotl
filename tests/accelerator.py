"""Run GPU tests on any supported GPU backend (CUDA or XPU), not only CUDA."""

import torch


def gpu_device_type() -> str | None:
    if torch.cuda.is_available():
        return "cuda"
    if torch.xpu.is_available():
        return "xpu"
    return None


# "cuda" on CPU-only hosts keeps module-level device constants valid; those tests skip.
DEVICE_TYPE = gpu_device_type() or "cuda"


def gpu_available() -> bool:
    return gpu_device_type() is not None


def device_module():
    return getattr(torch, DEVICE_TYPE)
