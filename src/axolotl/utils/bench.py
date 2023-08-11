"""Benchmarking and measurement utilities"""

import pynvml
import torch


def gpu_memory_usage(device):
    if isinstance(device, torch.device):
        device = device.index
    if isinstance(device, str) and device.startswith("cuda:"):
        device = int(device[5:])

    # NB torch.cuda.memory_usage returns zero so we use lower level api
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.used / 1024.0**3


def log_gpu_memory_usage(log, msg, device):
    log.info(
        f"GPU memory usage {msg}: {gpu_memory_usage(device):.03f} GB", stacklevel=2
    )
