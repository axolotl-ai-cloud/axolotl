"""Benchmarking and measurement utilities"""
import functools

import pynvml
import torch
from pynvml.nvml import NVMLError


def check_cuda_device(default_value):
    """
    wraps a function and returns the default value instead of running the
    wrapped function if cuda isn't available or the device is auto
    :param default_value:
    :return:
    """

    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            device = kwargs.get("device", args[0] if args else None)

            if (
                not torch.cuda.is_available()
                or device == "auto"
                or torch.device(device).type == "cpu"
            ):
                return default_value

            return func(*args, **kwargs)

        return wrapper

    return deco


@check_cuda_device(0.0)
def gpu_memory_usage(device=0):
    return torch.cuda.memory_allocated(device) / 1024.0**3


@check_cuda_device((0.0, 0.0, 0.0))
def gpu_memory_usage_all(device=0):
    usage = torch.cuda.memory_allocated(device) / 1024.0**3
    reserved = torch.cuda.memory_reserved(device) / 1024.0**3
    smi = gpu_memory_usage_smi(device)
    return usage, reserved - usage, max(0, smi - reserved)


@check_cuda_device(0.0)
def gpu_memory_usage_smi(device=0):
    if isinstance(device, torch.device):
        device = device.index
    if isinstance(device, str) and device.startswith("cuda:"):
        device = int(device[5:])
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / 1024.0**3
    except NVMLError:
        return 0.0


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
