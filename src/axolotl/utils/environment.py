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
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    except ValueError:
        return True

    if world_size > 1:
        node_world_size = int(os.environ.get("NODE_WORLD_SIZE", "8"))
        local_other_rank = (local_rank // node_world_size) * node_world_size
        local_other_rank += 1 if (local_rank % node_world_size) == 0 else 0
        try:
            can_p2p = torch.cuda.can_device_access_peer(local_rank, local_other_rank)
        except AssertionError as exc:
            # some sort of logic error in indexing processes, assume p2p is fine for now
            LOG.warning(exc)
            return True
        return can_p2p

    return True


def get_package_version(package: str) -> Version:
    version_str = version(package)
    return parse(version_str)


def is_package_version_ge(package: str, version_: str) -> bool:
    package_version = get_package_version(package)
    return package_version >= parse(version_)
