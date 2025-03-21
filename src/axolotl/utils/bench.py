"""Benchmarking and measurement utilities"""

import functools

import torch
from transformers.utils.import_utils import is_torch_npu_available

from axolotl.utils.distributed import get_device_type

try:
    from pynvml import (
        NVMLError,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo,
        nvmlInit,
    )
except ImportError:
    NVMLError = None
    nvmlDeviceGetHandleByIndex = None
    nvmlDeviceGetMemoryInfo = None
    nvmlInit = None


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
                device is None
                or not torch.cuda.is_available()
                or device == "auto"
                or torch.device(device).type == "cpu"
                or torch.device(device).type == "meta"
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


def mps_memory_usage_all():
    usage = torch.mps.current_allocated_memory() / 1024.0**3
    reserved = torch.mps.driver_allocated_memory() / 1024.0**3
    return usage, reserved - usage, 0


def npu_memory_usage_all(device=0):
    usage = torch.npu.memory_allocated(device) / 1024.0**3
    reserved = torch.npu.memory_reserved(device) / 1024.0**3
    return usage, reserved - usage, 0


@check_cuda_device(0.0)
def gpu_memory_usage_smi(device=0):
    if isinstance(device, torch.device):
        device = device.index
    if isinstance(device, str) and device.startswith("cuda:"):
        device = int(device[5:])
    if not nvmlInit:
        return 0.0
    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(device)
        info = nvmlDeviceGetMemoryInfo(handle)
        return info.used / 1024.0**3
    except NVMLError:
        return 0.0


def log_gpu_memory_usage(log, msg, device):
    cur_device = get_device_type()
    if torch.backends.mps.is_available():
        usage, cache, misc = mps_memory_usage_all()
    elif "npu" in str(cur_device) and is_torch_npu_available():
        usage, cache, misc = npu_memory_usage_all(device)
    else:
        usage, cache, misc = gpu_memory_usage_all(device)
    extras = []
    if cache > 0:
        extras.append(f"+{cache:.03f}GB cache")
    if misc > 0:
        extras.append(f"+{misc:.03f}GB misc")
    log.info(
        f"{str(cur_device)} memory usage {msg}: {usage:.03f}GB ({', '.join(extras)})",
        stacklevel=2,
    )
    return usage, cache, misc
