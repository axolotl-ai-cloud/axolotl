"""
utils to get GPU info for the current environment
"""

import os
from importlib.metadata import version

import torch
from accelerate.utils.environment import (
    check_cuda_p2p_ib_support as accelerate_check_cuda_p2p_ib_support,
)
from packaging.version import Version, parse

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def check_cuda_p2p_ib_support():
    if not accelerate_check_cuda_p2p_ib_support():
        return False
    if not check_cuda_p2p_support():
        return False
    return True


def check_cuda_p2p_support() -> bool:
    """Return True iff every local-GPU pair supports P2P; rank-symmetric and fail-closed on introspection failure."""
    # fail-closed: unintrospectable pairs must be treated as unsafe so all ranks agree on NCCL_P2P_DISABLE
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        LOG.warning(
            "check_cuda_p2p_support: invalid WORLD_SIZE=%r; disabling P2P "
            "(fail-closed posture).",
            os.environ.get("WORLD_SIZE"),
        )
        return False

    if world_size <= 1:
        return True

    try:
        n = torch.cuda.device_count()
    except Exception as exc:  # pragma: no cover - defensive  # noqa: BLE001
        LOG.warning(
            "check_cuda_p2p_support: device_count failed (%s); disabling P2P "
            "(fail-closed posture).",
            exc,
        )
        return False
    if n <= 1:
        return True

    for i in range(n):
        for j in range(i + 1, n):
            try:
                if not torch.cuda.can_device_access_peer(i, j):
                    return False
            except Exception as exc:  # noqa: BLE001 — broad catch keeps fail-closed even if C++ binding raises a non-AssertionError
                LOG.warning(
                    "check_cuda_p2p_support: can_device_access_peer(%s, %s) "
                    "raised %s (%s); disabling P2P (fail-closed posture).",
                    i,
                    j,
                    type(exc).__name__,
                    exc,
                )
                return False
    return True


def get_package_version(package: str) -> Version:
    version_str = version(package)
    return parse(version_str)


def is_package_version_ge(package: str, version_: str) -> bool:
    package_version = get_package_version(package)
    return package_version >= parse(version_)
