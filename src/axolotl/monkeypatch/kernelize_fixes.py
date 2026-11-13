"""Repair for transformers' kernelization under ``use_kernels=True``.

transformers hands ``kernels.kernelize`` the kernels library's own ``Device``,
which kernels >= 0.18 reads an ``index`` off; it gets the device type instead.
"""

import functools
from copy import copy

import torch

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

_ORIG_KERNELIZE = None
_ORIG_KERNEL_MAPPING = None


def _with_compatible_kernel_versions(mapping):
    from kernels import LayerRepository

    if isinstance(mapping, dict):
        return {
            key: _with_compatible_kernel_versions(value)
            for key, value in mapping.items()
        }
    if isinstance(mapping, LayerRepository) and mapping._revision is None:
        upgrades = {
            ("kernels-community/mamba-ssm", 3): 4,
            ("kernels-community/fla", 1): 2,
        }
        version = upgrades.get((mapping._repo_id, mapping._version))
        if version is not None:
            mapping = copy(mapping)
            mapping._version = version
    return mapping


def _kernel_mapping_with_compatible_versions(get_mapping):
    @functools.wraps(get_mapping)
    def wrapper():
        return _with_compatible_kernel_versions(get_mapping())

    return wrapper


def _kernelize_with_device_type(kernelize):
    @functools.wraps(kernelize)
    def wrapper(model, *args, device=None, **kwargs):
        if not (device is None or isinstance(device, str) or hasattr(device, "index")):
            # kernels takes "meta" as a torch.device but rejects it as a string
            device = torch.device("meta") if device.type == "meta" else device.type
        return kernelize(model, *args, device=device, **kwargs)

    return wrapper


def patch_kernelize_fixes() -> bool:
    global _ORIG_KERNELIZE, _ORIG_KERNEL_MAPPING
    if _ORIG_KERNELIZE is not None:
        return True
    from transformers.integrations import hub_kernels

    # only bound when transformers accepts the installed kernels release
    kernelize = getattr(hub_kernels, "_kernels_kernelize", None)
    if kernelize is None:
        return False
    _ORIG_KERNEL_MAPPING = hub_kernels.get_kernel_mapping_transformers
    hub_kernels.get_kernel_mapping_transformers = (
        _kernel_mapping_with_compatible_versions(_ORIG_KERNEL_MAPPING)
    )
    hub_kernels._kernels_kernelize = _kernelize_with_device_type(kernelize)
    _ORIG_KERNELIZE = kernelize
    LOG.info("kernelize_fixes: patched transformers' kernels.kernelize call")
    return True


def unpatch_kernelize_fixes() -> None:
    global _ORIG_KERNELIZE, _ORIG_KERNEL_MAPPING
    if _ORIG_KERNELIZE is None:
        return
    from transformers.integrations import hub_kernels

    hub_kernels._kernels_kernelize = _ORIG_KERNELIZE
    hub_kernels.get_kernel_mapping_transformers = _ORIG_KERNEL_MAPPING
    _ORIG_KERNELIZE = None
    _ORIG_KERNEL_MAPPING = None
