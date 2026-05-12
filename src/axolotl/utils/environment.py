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
    # D9 (fail-closed posture): when the introspection that would let us
    # *prove* every local-peer pair supports P2P fails or is ambiguous,
    # return ``False`` (i.e. disable P2P) instead of optimistically
    # returning ``True``. The previous fail-open posture trusted the
    # absence of evidence as evidence of safety; for an NCCL P2P
    # configuration knob the safer degradation is to disable P2P
    # symmetrically across ranks. The unsupported-NVLink case (the
    # original bug this helper was written for) is then handled
    # uniformly with the "introspection unreliable" case: NCCL_P2P_DISABLE
    # gets set, every rank agrees, and NCCL falls back to a slower but
    # functional path rather than SIGSEGV'ing on the first collective.
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
            except Exception as exc:  # noqa: BLE001 — fail-closed posture, see below
                # F-#7 (Major) widens the catch from ``AssertionError``
                # to ``Exception``. PyTorch 2.6's
                # ``torch.cuda.can_device_access_peer`` validates
                # device indices with ``AssertionError("Invalid device
                # id")`` but ALSO delegates to the C++ binding
                # ``_cuda_canDeviceAccessPeer`` which can surface
                # exceptions from the CUDA runtime (e.g.
                # ``RuntimeError`` wrapping ``cudaErrorInvalidDevice``
                # or peer-access-machinery errors) that wouldn't
                # match ``AssertionError``. An unhandled exception
                # from the C++ layer would propagate out of this
                # helper and break the fail-closed contract: ranks
                # would disagree about ``NCCL_P2P_DISABLE``, which is
                # exactly the SIGSEGV class commit ``91e0912e`` set
                # out to prevent.
                #
                # Indexing / introspection problem on this (i, j) pair —
                # the rank-symmetric guarantee we need (every rank
                # agrees on whether P2P is available) requires that we
                # treat an unintrospectable pair as "P2P not safe"
                # rather than "assume safe". Disable P2P; NCCL falls
                # back to a non-P2P path uniformly across ranks.
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
