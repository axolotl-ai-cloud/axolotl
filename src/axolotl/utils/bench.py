"""Benchmarking and measurement utilities"""

import functools
import logging

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
    active = torch.cuda.memory_stats().get("active_bytes.all.peak", 0) / 1024.0**3
    allocated = torch.cuda.max_memory_allocated(device) / 1024.0**3
    reserved = torch.cuda.max_memory_reserved(device) / 1024.0**3
    torch.cuda.reset_peak_memory_stats(device)
    return active, allocated, reserved


def mps_memory_usage_all():
    active = torch.mps.current_allocated_memory() / 1024.0**3
    allocated = torch.mps.driver_allocated_memory() / 1024.0**3
    return active, allocated, 0


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


def get_gpu_memory_usage(device: int | torch.device = 0):
    cur_device_type = str(get_device_type())
    if torch.backends.mps.is_available():
        usage, cache, misc = mps_memory_usage_all()
    elif "npu" in cur_device_type and is_torch_npu_available():
        usage, cache, misc = npu_memory_usage_all(device)
    elif "cuda" in cur_device_type and torch.cuda.is_available():
        usage, cache, misc = gpu_memory_usage_all(device)
    else:
        return 0.0, 0.0, 0.0

    return usage, cache, misc


def log_gpu_memory_usage(
    log: logging.Logger | logging.LoggerAdapter,
    msg: str = "",
    device: int | torch.device = 0,
):
    try:
        active, allocated, reserved = get_gpu_memory_usage(device)
    except ValueError:
        # likely CPU, ignore
        return
    cur_device_type = str(get_device_type())
    extras = []
    if allocated > 0:
        extras.append(f"+{allocated:.03f}GB allocated")
    if reserved > 0:
        extras.append(f"+{reserved:.03f}GB reserved")
    msg = f"{cur_device_type} memory active:" if not msg else msg
    log.debug(
        f"{msg} {active:.03f}GB ({', '.join(extras)})",
        stacklevel=2,
    )
