"""
utils to get GPU info for the current environment
"""

from accelerate.utils.environment import (
    check_cuda_p2p_ib_support as accelerate_check_cuda_p2p_ib_support,
)
from accelerate.utils.environment import (
    get_gpu_info,
)


def check_cuda_p2p_ib_support():
    if not accelerate_check_cuda_p2p_ib_support():
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
    except Exception:  # pylint: disable=broad-except # nosec
        pass
    return True
