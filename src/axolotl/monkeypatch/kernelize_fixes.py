"""Repair for transformers' kernelization under ``use_kernels=True``.

transformers hands ``kernels.kernelize`` the kernels library's own ``Device``,
which kernels >= 0.18 reads an ``index`` off; it gets the device type instead.
"""

import functools

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

_ORIG_KERNELIZE = None


def _kernelize_with_device_type(kernelize):
    @functools.wraps(kernelize)
    def wrapper(model, *args, device=None, **kwargs):
        if not (device is None or isinstance(device, str) or hasattr(device, "index")):
            device = device.type
        return kernelize(model, *args, device=device, **kwargs)

    return wrapper


def patch_kernelize_fixes() -> bool:
    global _ORIG_KERNELIZE
    if _ORIG_KERNELIZE is not None:
        return True
    from transformers.integrations import hub_kernels

    # only bound when transformers accepts the installed kernels release
    kernelize = getattr(hub_kernels, "_kernels_kernelize", None)
    if kernelize is None:
        return False
    hub_kernels._kernels_kernelize = _kernelize_with_device_type(kernelize)
    _ORIG_KERNELIZE = kernelize
    LOG.info("kernelize_fixes: patched transformers' kernels.kernelize call")
    return True


def unpatch_kernelize_fixes() -> None:
    global _ORIG_KERNELIZE
    if _ORIG_KERNELIZE is None:
        return
    from transformers.integrations import hub_kernels

    hub_kernels._kernels_kernelize = _ORIG_KERNELIZE
    _ORIG_KERNELIZE = None
