"""
utils to get GPU info for the current environment
"""

import os
import subprocess  # nosec B404
from importlib.metadata import version

from accelerate.utils.environment import (
    check_cuda_p2p_ib_support as accelerate_check_cuda_p2p_ib_support,
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
    try:
        gpu_count = int(os.environ.get("RUNPOD_GPU_COUNT", "1"))
    except ValueError:
        return True
    if gpu_count >= 2:
        # run `nvidia-smi topo -p2p n` and inspect the GPU0 row
        try:
            result = subprocess.run(  # nosec B603 B607
                ["nvidia-smi", "topo", "-p2p", "n"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            return True  # fail-open if detection fails
        output_lines = result.stdout.strip().split("\n")
        # filter rows that start with "GPU0" (avoid header row)
        gpu0_rows = [line for line in output_lines if line.lstrip().startswith("GPU0")]
        if not gpu0_rows:
            return True
        # consider P2P supported if any OK is present in the GPU0 row
        return "OK" in gpu0_rows[-1]
    return True


def get_package_version(package: str) -> Version:
    version_str = version(package)
    return parse(version_str)


def is_package_version_ge(package: str, version_: str) -> bool:
    package_version = get_package_version(package)
    return package_version >= parse(version_)
