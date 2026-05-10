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
    """Return whether ALL local-GPU pairs support peer-to-peer access.

    Iterates the full local-peer matrix and returns False if any unordered
    pair lacks P2P. The result is rank-symmetric — every rank computes the
    same answer regardless of its ``LOCAL_RANK``. This matters on
    heterogeneous-NVLink topologies (e.g. some pairs have NVLink, others
    don't): the prior implementation probed only one ``(local_rank,
    other_rank)`` pair where ``other_rank`` collapsed to 0 or 1, which
    returned different answers per rank and produced an asymmetric
    ``NCCL_P2P_DISABLE`` setting across ranks → SIGSEGV in the first
    NCCL collective. See ProTrain Phase 2 audit follow-up
    (multigpu_segfault_diagnosis.md).
    """
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        return True

    if world_size <= 1:
        return True

    try:
        n = torch.cuda.device_count()
    except Exception as exc:  # pragma: no cover - defensive
        LOG.warning(
            "check_cuda_p2p_support: device_count failed (%s); assuming p2p ok",
            exc,
        )
        return True
    if n <= 1:
        return True

    for i in range(n):
        for j in range(i + 1, n):
            try:
                if not torch.cuda.can_device_access_peer(i, j):
                    return False
            except AssertionError as exc:
                # Indexing problem; bail safe to True so we don't force-disable
                # P2P on a config we can't introspect.
                LOG.warning(exc)
                return True
    return True


def get_package_version(package: str) -> Version:
    version_str = version(package)
    return parse(version_str)


def is_package_version_ge(package: str, version_: str) -> bool:
    package_version = get_package_version(package)
    return package_version >= parse(version_)
