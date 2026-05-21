"""Shared HardwareProfile construction (device probe + world-size resolution + PCIe seed)."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from axolotl.integrations.protrain.types import HardwareProfile
from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    pass

LOG = get_logger(__name__)


# Pre-profile seed for HardwareProfile; profiler overwrites with measured value.
DEFAULT_PCIE_BPS = 13e9


def resolve_world_size_from_env() -> int:
    """Return WORLD_SIZE env (default 1)."""
    raw = os.environ.get("WORLD_SIZE")
    if raw is None:
        return 1
    try:
        world_size = int(raw)
    except ValueError as exc:
        # Malformed WORLD_SIZE + set RANK/LOCAL_RANK → raise; collapse to 1 only without rank vars.
        if (
            os.environ.get("RANK") is not None
            or os.environ.get("LOCAL_RANK") is not None
        ):
            LOG.error(
                "ProTrain: WORLD_SIZE=%r is not an integer but RANK / LOCAL_RANK "
                "is set; refusing to silently collapse a distributed run to 1.",
                raw,
            )
            raise RuntimeError(
                f"WORLD_SIZE={raw!r} is not a valid integer; cannot resolve "
                "world size while RANK / LOCAL_RANK is set."
            ) from exc
        return 1
    # Same RANK-aware policy for non-positive WORLD_SIZE.
    if world_size < 1:
        if (
            os.environ.get("RANK") is not None
            or os.environ.get("LOCAL_RANK") is not None
        ):
            LOG.error(
                "ProTrain: WORLD_SIZE=%r is not >= 1 but RANK / LOCAL_RANK "
                "is set; refusing to silently collapse a distributed run to 1.",
                raw,
            )
            raise RuntimeError(
                f"WORLD_SIZE={raw!r} must be >= 1 when RANK / LOCAL_RANK is set."
            )
        return 1
    return world_size


def _resolve_world_size() -> int:
    """Prefer initialised PG, then env (gated on RANK/LOCAL_RANK), then 1."""
    try:
        import torch.distributed as _dist

        if _dist.is_available() and _dist.is_initialized():
            return max(1, int(_dist.get_world_size()))
        # Gate env-WORLD_SIZE on RANK/LOCAL_RANK presence to catch misconfigured exports.
        if (
            os.environ.get("RANK") is not None
            and os.environ.get("LOCAL_RANK") is not None
        ):
            return resolve_world_size_from_env()
        return 1
    except ImportError:
        return 1


def _resolve_device_index() -> int:
    """Pick CUDA ordinal from LOCAL_RANK; falls back to current_device on parse/range errors."""
    import torch

    raw_local_rank = os.environ.get("LOCAL_RANK", "0")
    try:
        local_rank = int(raw_local_rank)
    except ValueError:
        LOG.warning(
            "ProTrain: invalid LOCAL_RANK=%r; falling back to current CUDA device.",
            raw_local_rank,
        )
        return torch.cuda.current_device()

    visible = int(torch.cuda.device_count())
    if visible <= 0:
        raise RuntimeError("ProTrain requires at least one visible CUDA device.")
    if not (0 <= local_rank < visible):
        LOG.warning(
            "ProTrain: LOCAL_RANK=%d out of visible CUDA range [0, %d); "
            "falling back to current CUDA device.",
            local_rank,
            visible,
        )
        return torch.cuda.current_device()
    return local_rank


def build_hardware_profile(
    *,
    world_size_override: int | None = None,
    zero3_shard: bool = False,
    device_index: int | None = None,
) -> HardwareProfile:
    """Construct HardwareProfile from torch.cuda state; raises if CUDA unavailable."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "ProTrain requires a CUDA device; torch.cuda.is_available() is False."
        )

    if device_index is None:
        device_index = _resolve_device_index()
    else:
        if not isinstance(device_index, int) or isinstance(device_index, bool):
            raise RuntimeError(
                f"device_index must be an int, got {type(device_index).__name__}."
            )
        device_count = int(torch.cuda.device_count())
        if device_count <= 0:
            raise RuntimeError("ProTrain requires at least one visible CUDA device.")
        if device_index < 0 or device_index >= device_count:
            raise RuntimeError(
                f"device_index={device_index} is out of range; expected an "
                f"int in [0, {device_count - 1}] (torch.cuda.device_count()="
                f"{device_count})."
            )

    props = torch.cuda.get_device_properties(device_index)
    gpu_memory_bytes = int(props.total_memory)
    gpu_sku = torch.cuda.get_device_name(device_index)

    if world_size_override is not None:
        # bool check first; isinstance(True, int) would otherwise sneak past.
        if isinstance(world_size_override, bool):
            raise RuntimeError(
                "world_size_override must be a positive int, got "
                f"{world_size_override!r}."
            )
        try:
            ws_int = int(world_size_override)
        except (ValueError, TypeError) as exc:
            raise RuntimeError(
                "world_size_override must be a positive int, got "
                f"{world_size_override!r}."
            ) from exc
        if ws_int < 1:
            raise RuntimeError(
                "world_size_override must be a positive int, got "
                f"{world_size_override!r}."
            )
        world_size = ws_int
    else:
        world_size = _resolve_world_size()

    # Reject non-bool zero3_shard explicitly to avoid truthy-coerce of bogus values.
    if not isinstance(zero3_shard, bool):
        raise TypeError(
            f"zero3_shard must be a bool, got {type(zero3_shard).__name__}: "
            f"{zero3_shard!r}"
        )

    return HardwareProfile(
        gpu_sku=gpu_sku,
        gpu_memory_bytes=gpu_memory_bytes,
        gpu_count=world_size,
        pcie_h2d_bps=DEFAULT_PCIE_BPS,
        pcie_d2h_bps=DEFAULT_PCIE_BPS,
        has_nvlink=False,
        zero3_shard=zero3_shard,
    )


__all__ = [
    "DEFAULT_PCIE_BPS",
    "build_hardware_profile",
    "resolve_world_size_from_env",
]
