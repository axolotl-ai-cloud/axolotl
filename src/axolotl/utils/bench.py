"""Benchmarking and measurement utilities"""

import pynvml
import torch


def gpu_memory_usage(device=0):
    return torch.cuda.memory_allocated(device) / 1024.0**3


def gpu_memory_usage_all(device=0):
    usage = torch.cuda.memory_allocated(device) / 1024.0**3
    reserved = torch.cuda.memory_reserved(device) / 1024.0**3
    smi = gpu_memory_usage_smi(device)
    return usage, reserved - usage, max(0, smi - reserved)


def gpu_memory_usage_smi(device=0):
    if isinstance(device, torch.device):
        device = device.index
    if isinstance(device, str) and device.startswith("cuda:"):
        device = int(device[5:])

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.used / 1024.0**3


def log_gpu_memory_usage(log, msg, device):
    usage, cache, misc = gpu_memory_usage_all(device)
    extras = []
    if cache > 0:
        extras.append(f"+{cache:.03f}GB cache")
    if misc > 0:
        extras.append(f"+{misc:.03f}GB misc")
    log.info(
        f"GPU memory usage {msg}: {usage:.03f}GB ({', '.join(extras)})", stacklevel=2
    )
    return usage, cache, misc
