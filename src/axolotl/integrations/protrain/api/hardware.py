"""Shared :class:`HardwareProfile` construction (§3.2, §7).

Lifts the device-probe + world-size-resolution logic out of
:mod:`axolotl.integrations.protrain.plugin` so both the plugin path
(``post_model_load`` -> ``protrain_model_wrapper``) and the direct API
helper (:func:`axolotl.integrations.protrain.api.auto_wrap`) share a
single canonical recipe.

Three knobs the helper exposes (all optional):

* ``world_size_override`` — force a specific value when the caller knows
  better than the live PG / env (mostly tests). When ``None``, the
  helper prefers an initialised ``torch.distributed`` PG, then the
  ``WORLD_SIZE`` env (only when ``RANK`` and ``LOCAL_RANK`` are also
  set — mirrors :func:`_early_init_dist_for_nccl`'s launcher-env sanity
  check), then 1.
* ``zero3_shard`` — direct override for the ``HardwareProfile.zero3_shard``
  bool. The plugin-side wrapper computes this from cfg flags + the
  resolved world size; the direct API caller almost never wants
  sharding, so default is False.
* ``device_index`` — explicit CUDA ordinal. When ``None``, the helper
  honours ``LOCAL_RANK`` (with a defensive fallback to
  ``torch.cuda.current_device()`` on parse / range errors) so per-rank
  heterogeneous-memory rigs report the correct ``capacity_bytes`` /
  SKU instead of always reading device 0.

PCIe bandwidth is seeded with :data:`DEFAULT_PCIE_BPS`. The profiler's
microbenchmark overwrites this once a cache miss runs; the seed only
feeds the cost model's effective-bandwidth prior between import time
and the first profile.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from axolotl.integrations.protrain.types import HardwareProfile
from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    pass

LOG = get_logger(__name__)


# Default PCIe H2D bandwidth assumed for HardwareProfile construction when
# no measured value is available. 13 GB/s matches a typical PCIe Gen4 x16
# 3090 rig; the profiler's microbench will overwrite this once the cache
# key misses and a full profile runs — this constant only seeds the
# constructor for the cost model's effective-bandwidth prior.
DEFAULT_PCIE_BPS = 13e9


def resolve_world_size_from_env() -> int:
    """Return ``WORLD_SIZE`` from the env, defaulting to 1.

    Both torchrun and Accelerate's launchers populate ``WORLD_SIZE`` /
    ``RANK`` / ``LOCAL_RANK`` / ``MASTER_ADDR`` / ``MASTER_PORT`` before
    the user script starts. The plugin path treats the env as the source
    of truth before the trainer (and thus Accelerate) has had a chance
    to call :func:`torch.distributed.init_process_group`.
    """
    raw = os.environ.get("WORLD_SIZE")
    if raw is None:
        return 1
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


def _resolve_world_size() -> int:
    """Prefer an initialised PG, then env, then 1.

    Mirrors the resolution path documented on
    :func:`axolotl.integrations.protrain.plugin._build_hardware_profile`:
    the visible CUDA device count is NOT a substitute for the
    distributed rank count, so on a single-process run on a multi-GPU
    host this would otherwise inflate ``world_size`` from 1 to N and
    skew the profiler cache key, the per-rank CPU-capacity budget, and
    the cost-model sharding divisor.
    """
    try:
        import torch.distributed as _dist

        if _dist.is_available() and _dist.is_initialized():
            return max(1, int(_dist.get_world_size()))
        # Sanity-check the launcher provided enough env to rendezvous: a
        # bare ``WORLD_SIZE > 1`` without ``RANK`` / ``LOCAL_RANK``
        # typically indicates a misconfigured manual export rather than a
        # real torchrun-managed process.
        if (
            os.environ.get("RANK") is not None
            and os.environ.get("LOCAL_RANK") is not None
        ):
            return resolve_world_size_from_env()
        return 1
    except ImportError:
        return 1


def _resolve_device_index() -> int:
    """Pick the CUDA ordinal this rank should report from.

    Honours ``LOCAL_RANK`` so heterogeneous-memory multi-GPU rigs report
    the correct ``capacity_bytes`` / SKU per rank instead of always
    reading device 0. Falls back to ``torch.cuda.current_device()`` on
    parse / range errors.
    """
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
        raise RuntimeError(
            "ProTrain requires at least one visible CUDA device."
        )
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
    """Construct a :class:`HardwareProfile` from live ``torch.cuda`` state.

    Shared between :func:`axolotl.integrations.protrain.plugin._build_hardware_profile`
    (the plugin path) and :func:`axolotl.integrations.protrain.api.auto_wrap`
    (the direct API path) so both paths agree on what fields a
    HardwareProfile carries by default.

    Parameters
    ----------
    world_size_override : int | None
        Skip world-size auto-resolution and use this value verbatim. When
        ``None``, prefer an initialised ``torch.distributed`` PG, then
        the ``WORLD_SIZE`` env (gated on ``RANK`` / ``LOCAL_RANK`` being
        set), then 1.
    zero3_shard : bool
        Pass-through to :class:`HardwareProfile.zero3_shard`. Direct API
        callers default to False (no sharding); the plugin path computes
        this from cfg flags and passes it explicitly.
    device_index : int | None
        Explicit CUDA ordinal to probe. When ``None``, honours
        ``LOCAL_RANK``.

    Raises
    ------
    RuntimeError
        If ``torch.cuda.is_available()`` is False, or no visible CUDA
        device exists.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "ProTrain requires a CUDA device; torch.cuda.is_available() is False."
        )

    if device_index is None:
        device_index = _resolve_device_index()
    elif int(torch.cuda.device_count()) <= 0:
        raise RuntimeError("ProTrain requires at least one visible CUDA device.")

    props = torch.cuda.get_device_properties(device_index)
    gpu_memory_bytes = int(props.total_memory)
    gpu_sku = torch.cuda.get_device_name(device_index)

    if world_size_override is not None:
        world_size = max(1, int(world_size_override))
    else:
        world_size = _resolve_world_size()

    return HardwareProfile(
        gpu_sku=gpu_sku,
        gpu_memory_bytes=gpu_memory_bytes,
        gpu_count=world_size,
        pcie_h2d_bps=DEFAULT_PCIE_BPS,
        pcie_d2h_bps=DEFAULT_PCIE_BPS,
        has_nvlink=False,
        zero3_shard=bool(zero3_shard),
    )


__all__ = [
    "DEFAULT_PCIE_BPS",
    "build_hardware_profile",
    "resolve_world_size_from_env",
]
