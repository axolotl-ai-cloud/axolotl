"""
utils to get GPU info for the current environment
"""

import os
import subprocess  # nosec B404
from importlib.metadata import version

from accelerate.utils.environment import (
    check_cuda_p2p_ib_support as accelerate_check_cuda_p2p_ib_support,
)
from accelerate.utils.environment import (
    get_gpu_info,
)
from packaging.version import Version, parse


def check_cuda_p2p_ib_support():
    if not accelerate_check_cuda_p2p_ib_support():
        return False
    if not check_runpod_p2p_support():
        return False
    unsupported_devices = {"RTX 6000 Ada", "L40S"}
    try:
        device_names, device_count = get_gpu_info()
        if 1 < device_count < 8:
            if any(
                unsupported_device in device_name
                for device_name in device_names
                for unsupported_device in unsupported_devices
            ):
                return False
    except Exception:  # nosec B110
        pass
    return True


def check_runpod_p2p_support() -> bool:
    if "RUNPOD_GPU_COUNT" not in os.environ:
        return True
    gpu_count = int(os.environ.get("RUNPOD_GPU_COUNT", "1"))
    if gpu_count >= 2:
        # run `nnvidia-smi topo -p2p n | grep GPU0 | tail -n1` as subprocess and check if "OK" string is present
        output = (
            subprocess.check_output(["nvidia-smi", "topo", "-p2p", "n"])  # nosec B603 B607
            .decode("utf-8")
            .strip()
        )
        output_lines = output.split("\n")
        # filter lines that contain "GPU0"
        output_lines = [line for line in output_lines if "GPU0" in line]
        # check if "OK" string is present in the last line
        if "OK" in output_lines[-1]:
            return True
        return False
    return True


def get_package_version(package: str) -> Version:
    version_str = version(package)
    return parse(version_str)


def is_package_version_ge(package: str, version_: str) -> bool:
    package_version = get_package_version(package)
    return package_version >= parse(version_)
