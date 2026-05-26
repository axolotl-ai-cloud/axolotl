"""Per-rank chunk manager driving the persistent / non-persistent split."""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any, cast

from axolotl.integrations.protrain.types import (
    ChunkId,
    ChunkLayout,
    ParamId,
)
from axolotl.utils.logging import get_logger


def _slow_gather_threshold_s() -> float:
    """PROTRAIN_DEBUG_SLOW_GATHER_S (default 5.0) — gathers above this WARN-log per-chunk wall time."""
    raw = os.environ.get("PROTRAIN_DEBUG_SLOW_GATHER_S", "5.0")
    try:
        v = float(raw)
    except ValueError:
        return 5.0
    return max(0.0, v)


_SLOW_GATHER_THRESHOLD_S: float = _slow_gather_threshold_s()


def _slow_offload_regather_threshold_s() -> float:
    """PROTRAIN_DEBUG_SLOW_OFFLOAD_REGATHER_S (default 5.0) — per-chunk OFFLOAD re-gather WARN threshold.

    Separate from SLOW_GATHER so the OFFLOAD backward H2D + all_gather path can be
    localized independently of the forward initial gather. v71/v72-redux confirmed
    the bs=2 + n_offload>0 hang is invisible to SLOW_GATHER/SLOW_ADAM; this watchdog
    times each backward re-gather to pinpoint stream-contention on non-NVLink topology.
    """
    raw = os.environ.get("PROTRAIN_DEBUG_SLOW_OFFLOAD_REGATHER_S", "5.0")
    try:
        v = float(raw)
    except ValueError:
        return 5.0
    return max(0.0, v)


_SLOW_OFFLOAD_REGATHER_S: float = _slow_offload_regather_threshold_s()


def _slow_sharded_gather_threshold_s() -> float:
    """PROTRAIN_DEBUG_SLOW_SHARDED_GATHER_S (default 5.0) — per-call sharded all_gather WARN threshold.

    Times each ``_gather_sharded`` invocation (forward or backward) so that
    Mode C ``zero3_shard=True`` stalls in the NCCL ``all_gather_into_tensor``
    path are observable independently of SLOW_GATHER / SLOW_OFFLOAD_REGATHER.
    The forward-gather path for sharded chunks is the v73 Mode C bs=2 hang
    site (block=10 forward); this watchdog will pinpoint a repeat if it occurs.
    """
    raw = os.environ.get("PROTRAIN_DEBUG_SLOW_SHARDED_GATHER_S", "5.0")
    try:
        v = float(raw)
    except ValueError:
        return 5.0
    return max(0.0, v)


_SLOW_SHARDED_GATHER_S: float = _slow_sharded_gather_threshold_s()


def _gather_internals_trace_enabled() -> bool:
    """PROTRAIN_DEBUG_GATHER_INTERNALS={1,true,yes,on} enables 4-phase sub-op timing inside gather().

    Times each sub-op within ``_gather_impl_body`` / ``_gather_sharded`` /
    ``_rebind_params_to_buffer``: ``bind_metadata`` (CPU-side rebind, pre-collective),
    ``h2d_copy`` (per-region shard H2D copy), ``all_gather_issue`` (CPU-side issue
    point of ``dist.all_gather_into_tensor``, NOT completion), ``split_rebind``
    (post-collective param.data rebind). Mode C bs=2 forward hangs between
    prefetch_stream.wait_stream return and gather() return on ranks 0+3; this
    split distinguishes CPU bookkeeping vs allocator vs NCCL issue. Zero overhead
    when off. Per-line ``[rank=N] chunk_id=K phase=<...> sub_op=<...> took X.Xs``.
    """
    raw = os.environ.get("PROTRAIN_DEBUG_GATHER_INTERNALS", "")
    return raw.strip().lower() in ("1", "true", "yes", "on")


_GATHER_INTERNALS_TRACE: bool = _gather_internals_trace_enabled()


if TYPE_CHECKING:
    import torch
    from torch import nn

    from axolotl.integrations.protrain.chunk.buffer_pool import BufferPool
    from axolotl.integrations.protrain.chunk.optim import (
        CpuFusedAdamAdapter,
        GpuFusedAdamAdapter,
    )
    from axolotl.integrations.protrain.chunk.pinned_alloc import PinnedHostMemory

LOG = get_logger(__name__)

_DDP_IGNORE_ATTR = "_ddp_params_and_buffers_to_ignore"
_DDP_ORIGINAL_IGNORE_ATTR = "_protrain_ddp_original_ignore"
_DDP_IGNORE_OWNERS_ATTR = "_protrain_ddp_ignore_owners"
_DDP_IGNORE_OWNER_CHUNK = "chunk"


def _as_ignore_list(value: Any) -> list[str]:
    if value is None:
        return []
    try:
        return [str(v) for v in value]
    except TypeError:
        return [str(value)]


def _ddp_ignore_owners(model: Any) -> dict[str, set[str]]:
    owners = getattr(model, _DDP_IGNORE_OWNERS_ATTR, None)
    if not isinstance(owners, dict):
        owners = {}
    normalized: dict[str, set[str]] = {}
    for owner, names in owners.items():
        normalized[str(owner)] = set(_as_ignore_list(names))
    return normalized


def register_protrain_ddp_ignore_names(
    model: Any,
    owner: str,
    names: set[str],
) -> tuple[int, int | None]:
    """Add one ProTrain owner's DDP-ignore names without owning caller names."""
    names = {str(name) for name in names if str(name)}
    if not names:
        return (0, None)

    existing_raw = getattr(model, _DDP_IGNORE_ATTR, None)
    existing = _as_ignore_list(existing_raw)
    if not hasattr(model, _DDP_ORIGINAL_IGNORE_ATTR):
        setattr(
            model,
            _DDP_ORIGINAL_IGNORE_ATTR,
            None if existing_raw is None else list(existing),
        )

    original = getattr(model, _DDP_ORIGINAL_IGNORE_ATTR)
    original_set = set(_as_ignore_list(original))
    owners = _ddp_ignore_owners(model)
    previous = set(owners.get(owner, set()))
    owners[owner] = set(names)

    stale_owned = previous - names - original_set
    live: list[str] = []
    seen: set[str] = set()
    for name in existing:
        if name in stale_owned:
            continue
        if name in seen:
            continue
        live.append(name)
        seen.add(name)

    for name in sorted(names):
        if name in seen:
            continue
        live.append(name)
        seen.add(name)

    setattr(model, _DDP_IGNORE_ATTR, live)
    setattr(model, _DDP_IGNORE_OWNERS_ATTR, owners)
    original_len = None if original is None else len(original_set)
    return (len(names), original_len)


def restore_protrain_ddp_ignore_names(model: Any, owner: str) -> tuple[int, int]:
    """Remove one ProTrain owner's DDP-ignore names while preserving caller edits."""
    owners = _ddp_ignore_owners(model)
    owned = set(owners.pop(owner, set()))
    if not owned:
        return (0, len(_as_ignore_list(getattr(model, _DDP_IGNORE_ATTR, None))))

    original = getattr(model, _DDP_ORIGINAL_IGNORE_ATTR, None)
    original_set = set(_as_ignore_list(original))
    other_owned: set[str] = set()
    for names in owners.values():
        other_owned.update(names)

    remove = owned - original_set - other_owned
    existing = _as_ignore_list(getattr(model, _DDP_IGNORE_ATTR, None))
    live: list[str] = []
    seen: set[str] = set()
    for name in existing:
        if name in remove:
            continue
        if name in seen:
            continue
        live.append(name)
        seen.add(name)

    if live:
        setattr(model, _DDP_IGNORE_ATTR, live)
    elif hasattr(model, _DDP_IGNORE_ATTR):
        try:
            delattr(model, _DDP_IGNORE_ATTR)
        except AttributeError:
            pass

    if owners:
        setattr(model, _DDP_IGNORE_OWNERS_ATTR, owners)
    else:
        if hasattr(model, _DDP_IGNORE_OWNERS_ATTR):
            try:
                delattr(model, _DDP_IGNORE_OWNERS_ATTR)
            except AttributeError:
                pass
        if hasattr(model, _DDP_ORIGINAL_IGNORE_ATTR):
            try:
                delattr(model, _DDP_ORIGINAL_IGNORE_ATTR)
            except AttributeError:
                pass

    return (len(remove), len(live))


def restore_all_protrain_ddp_ignore_names(model: Any) -> tuple[int, int]:
    """Remove every ProTrain-owned DDP-ignore name from ``model``."""
    owners = _ddp_ignore_owners(model)
    total_removed = 0
    remaining = len(_as_ignore_list(getattr(model, _DDP_IGNORE_ATTR, None)))
    for owner in list(owners):
        removed, remaining = restore_protrain_ddp_ignore_names(model, owner)
        total_removed += removed
    return (total_removed, remaining)


class _CpuParamSlot:
    """Per-parameter bookkeeping for a non-persistent chunk."""

    __slots__ = (
        "param_id",
        "cpu_data",
        "cpu_grad",
        "shape",
        "dtype",
        "byte_offset",
        "numel",
        "element_size",
    )

    def __init__(
        self,
        param_id: ParamId,
        cpu_data: "torch.Tensor | None",
        cpu_grad: "torch.Tensor | None",
        shape: "torch.Size",
        dtype: "torch.dtype",
        byte_offset: int,
        numel: int,
        element_size: int,
    ) -> None:
        self.param_id = param_id
        self.cpu_data = cpu_data
        self.cpu_grad = cpu_grad
        self.shape = shape
        self.dtype = dtype
        self.byte_offset = byte_offset
        self.numel = numel
        self.element_size = element_size


class _DtypeRegion:
    """One contiguous same-dtype byte region inside a sharded chunk."""

    __slots__ = (
        "chunk_offset",
        "region_bytes",
        "region_bytes_padded",
        "shard_bytes",
        "dtype",
        "element_size",
        "cpu_shard_bytes",
        "cpu_shard_grad_bytes",
        "shard_param",
        "is_trainable",
    )

    def __init__(
        self,
        chunk_offset: int,
        region_bytes: int,
        region_bytes_padded: int,
        shard_bytes: int,
        dtype: "torch.dtype",
        element_size: int,
        cpu_shard_bytes: "torch.Tensor",
        cpu_shard_grad_bytes: "torch.Tensor | None",
        shard_param: "torch.Tensor",
        is_trainable: bool,
    ) -> None:
        self.chunk_offset = chunk_offset
        self.region_bytes = region_bytes
        self.region_bytes_padded = region_bytes_padded
        self.shard_bytes = shard_bytes
        self.dtype = dtype
        self.element_size = element_size
        self.cpu_shard_bytes = cpu_shard_bytes
        self.cpu_shard_grad_bytes = cpu_shard_grad_bytes
        self.shard_param = shard_param
        self.is_trainable = is_trainable


class _ChunkShardState:
    """Per-chunk ZeRO-3 shard bookkeeping (populated when ``zero3_shard=True``)."""

    __slots__ = (
        "regions",
        "chunk_bytes",
        "shard_bytes",
    )

    def __init__(
        self,
        regions: "list[_DtypeRegion]",
        chunk_bytes: int,
        shard_bytes: int,
    ) -> None:
        self.regions = regions
        self.chunk_bytes = chunk_bytes
        self.shard_bytes = shard_bytes

    @property
    def is_sharded(self) -> bool:
        """Whether this chunk is genuinely in the sharded path."""
        return bool(self.regions)


class BackwardHandle:
    """RAII refcount handle for a chunk pinned across a backward window."""

    __slots__ = ("_chunk_id", "_manager", "_released")

    def __init__(self, chunk_id: ChunkId, manager: "ChunkManager") -> None:
        """Bind the handle to ``chunk_id`` on ``manager``; refcount already bumped."""
        self._chunk_id = chunk_id
        self._manager = manager
        self._released = False

    def release(self) -> None:
        """Drop this handle, decrementing the manager's refcount; idempotent."""
        if self._released:
            return
        self._released = True
        # Hold a hard ref to manager so __del__ during shutdown still works.
        try:
            self._manager._release_backward_handle(self._chunk_id)  # noqa: SLF001
        except Exception as exc:  # noqa: BLE001 — best-effort during shutdown
            LOG.debug(
                "BackwardHandle.release: drain failed for chunk %d: %s",
                int(self._chunk_id),
                exc,
            )

    def __del__(self) -> None:  # noqa: D401
        """Safety-net release on GC — RAII guarantee for the autograd path."""
        try:
            self.release()
        except Exception:  # noqa: BLE001 — destructors must not throw
            pass


class ChunkManager:
    """Runtime driver for a :class:`ChunkLayout`."""

    def __init__(
        self,
        model: "nn.Module",
        layout: ChunkLayout,
        n_persist: int,
        buffer_pool: "BufferPool | None",
        cpu_optim: "CpuFusedAdamAdapter | None" = None,
        gpu_optim: "GpuFusedAdamAdapter | None" = None,
        device: "torch.device | str | None" = None,
        world_size: int = 1,
        rank: int = 0,
        zero3_shard: bool = False,
        shape_preserving_placeholders: bool = False,
    ) -> None:
        if n_persist < 0 or n_persist > layout.N_chunk:
            raise ValueError(
                f"n_persist={n_persist} out of range [0, {layout.N_chunk}]"
            )
        if buffer_pool is not None and buffer_pool.S_chunk != layout.S_chunk:
            raise ValueError(
                f"buffer_pool.S_chunk ({buffer_pool.S_chunk}) "
                f"!= layout.S_chunk ({layout.S_chunk})"
            )
        # All-persistent layout: caller may pass buffer_pool=None; device must be supplied.
        if buffer_pool is None and device is None:
            raise ValueError(
                "device must be provided when buffer_pool is None "
                "(all-persistent layout has no pool to source it from)"
            )

        import torch

        self.model = model
        self.layout = layout
        self.buffer_pool = buffer_pool
        self.cpu_optim = cpu_optim
        self.gpu_optim = gpu_optim
        self.device = torch.device(
            device if device is not None else buffer_pool.device  # type: ignore[union-attr]
        )

        self.world_size: int = int(max(1, world_size))
        self.rank: int = int(max(0, rank))
        if self.rank >= self.world_size:
            raise ValueError(
                f"rank={self.rank} out of range for world_size={self.world_size}"
            )
        # Sharding requires both the flag and peers; ws==1 degrades to replication.
        self.zero3_shard: bool = bool(zero3_shard) and self.world_size > 1

        # When True, defer cross-rank grad sync to an outer DDP wrapper.
        self.skip_internal_grad_reduce: bool = False

        self._params_by_id: dict[ParamId, "nn.Parameter"] = {
            cast(ParamId, name): p for name, p in model.named_parameters()
        }

        self._persistent_ids: set[ChunkId] = set()
        self._non_persistent_ids: set[ChunkId] = set(
            cast(ChunkId, i) for i in range(layout.N_chunk)
        )

        self._persistent_buffers: dict[ChunkId, "torch.Tensor"] = {}

        self._cpu_slots: dict[ChunkId, list[_CpuParamSlot]] = {}

        # Custom pinned-memory pools (precise-size, no power-of-2 rounding waste).
        self._cpu_param_pool: "PinnedHostMemory | None" = None
        self._cpu_grad_pool: "PinnedHostMemory | None" = None

        self._chunk_shards: dict[ChunkId, _ChunkShardState] = {}

        self._empty_by_dtype: dict["torch.dtype", "torch.Tensor"] = {}

        # Opt-in: bind released param.data to a zero-stride view so autograd sees real shape.
        self._shape_preserving_placeholders: bool = bool(shape_preserving_placeholders)
        self._shape_scratch_by_dtype: dict["torch.dtype", "torch.Tensor"] = {}

        self._grad_remaining: dict[ChunkId, int] = {}
        self._grad_initial: dict[ChunkId, int] = {}
        self._cpu_step_ready_chunks: set[ChunkId] = set()
        self._cpu_step_events: dict[ChunkId, Any] = {}
        self._cpu_step_post_steps: dict[ChunkId, Any] = {}
        self._persistent_grads_synced: set[ChunkId] = set()

        self._grad_hook_handles: list[object] = []

        # storage-ptr -> chunk_id reverse lookup; storages survive view ops.
        self._storage_ptr_to_chunk: dict[int, ChunkId] = {}

        # Backward-window pin counter; non-zero defers offload.
        self._backward_refcount: dict[ChunkId, int] = {}

        self._deferred_offloads: set[ChunkId] = set()

        # Lease-idempotency: gather() may fire 2-3x per active window; pool counter must bump once.
        self._active_chunks: set[ChunkId] = set()

        # Per-chunk physical byte size for oversize-chunk routing through the pool side-table.
        self._chunk_bytes_by_id: dict[ChunkId, int] = {}

        self._closed: bool = False

        self.mark_persistent(n_persist)

    # ---- configuration -------------------------------------------------

    def mark_persistent(self, first_n: int) -> None:
        """Tag chunks ``[0, first_n) ∪ layout.mandatory_persistent`` as persistent."""
        if first_n < 0 or first_n > self.layout.N_chunk:
            raise ValueError(
                f"first_n={first_n} out of range [0, {self.layout.N_chunk}]"
            )
        # Single-source the definition through layout.effective_persistent_ids.
        new_persistent_ids = set(self.layout.effective_persistent_ids(first_n))
        new_non_persistent_ids = {
            cast(ChunkId, i)
            for i in range(self.layout.N_chunk)
            if cast(ChunkId, i) not in new_persistent_ids
        }
        # After materialization the residency split is baked in; flipping it would silently corrupt weights since gather/offload paths skip already-resident chunks.
        if (self._cpu_slots or self._persistent_buffers) and (
            new_persistent_ids != self._persistent_ids
            or new_non_persistent_ids != self._non_persistent_ids
        ):
            raise RuntimeError(
                "ChunkManager.mark_persistent() cannot change the residency "
                "split after chunks have been materialized; rebuild the "
                "manager first."
            )
        self._persistent_ids = new_persistent_ids
        self._non_persistent_ids = new_non_persistent_ids
        LOG.debug(
            "ChunkManager.mark_persistent: %d / %d chunks resident on GPU",
            first_n,
            self.layout.N_chunk,
        )

    # ---- M4.5: init-time chunk offload + per-param grad hooks ----------

    def materialize_offload(self) -> int:
        """Physically move non-persistent chunks' params to pinned CPU memory."""
        if self._cpu_param_pool is not None or self._cpu_slots:
            LOG.debug(
                "ChunkManager.materialize_offload: already materialized "
                "(%d chunks), no-op",
                len(self._cpu_slots),
            )
            return 0

        import torch

        from axolotl.integrations.protrain.chunk.pinned_alloc import PinnedHostMemory

        # 16-byte inter-chunk align covers any dtype up to fp64.
        _INTER_CHUNK_ALIGN = 16

        # Pass 1: planning — no allocations yet.
        chunk_plans: list[dict] = []
        total_param_pool_bytes = 0
        total_grad_pool_bytes = 0

        def _align_up(n: int, a: int) -> int:
            """Round ``n`` up to the next multiple of ``a`` (a > 0)."""
            return ((n + a - 1) // a) * a

        for cid_int in sorted(self._non_persistent_ids):
            cid = cast(ChunkId, cid_int)
            param_ids = self.layout.chunks[int(cid)]
            if not param_ids:
                continue

            # Pad each param offset up to its element_size; mixed-dtype chunks need this for .view(dtype).
            element_sizes: list[int] = []
            per_param_bytes: list[int] = []
            for pid in param_ids:
                param = self._params_by_id.get(pid)
                if param is None:
                    element_sizes.append(0)
                    per_param_bytes.append(0)
                    continue
                nbytes = int(param.numel()) * int(param.element_size())
                per_param_bytes.append(nbytes)
                element_sizes.append(int(param.element_size()))

            aligned_offsets: list[int] = []
            offset = 0
            for nbytes, esz in zip(per_param_bytes, element_sizes, strict=True):
                if nbytes == 0 or esz == 0:
                    aligned_offsets.append(offset)
                    continue
                offset = ((offset + esz - 1) // esz) * esz
                aligned_offsets.append(offset)
                offset += nbytes
            chunk_bytes = offset

            if chunk_bytes == 0:
                continue

            # Always shard under zero3; homogeneous chunks collapse to a single region.
            chunk_is_shardable = self.zero3_shard
            # list of (dtype, esize, start_off, end_off, is_trainable)
            dtype_regions: list[tuple] = []
            if chunk_is_shardable:
                cur_dtype = None
                cur_esize = 0
                cur_start = 0
                cur_end = 0
                cur_trainable: bool | None = None
                for pid, nbytes, off, esz in zip(
                    param_ids,
                    per_param_bytes,
                    aligned_offsets,
                    element_sizes,
                    strict=True,
                ):
                    if nbytes == 0 or esz == 0:
                        continue
                    param = self._params_by_id.get(pid)
                    if param is None:
                        continue
                    dtype_here = param.data.dtype
                    # Region must be uniformly trainable or uniformly frozen so grad allocation matches.
                    trainable_here = bool(param.requires_grad)
                    param_end = off + nbytes
                    if cur_dtype is None:
                        cur_dtype = dtype_here
                        cur_esize = esz
                        cur_start = off
                        cur_end = param_end
                        cur_trainable = trainable_here
                    elif dtype_here == cur_dtype and trainable_here == cur_trainable:
                        if param_end > cur_end:
                            cur_end = param_end
                        if off < cur_start:
                            cur_start = off
                    else:
                        dtype_regions.append(
                            (
                                cur_dtype,
                                cur_esize,
                                cur_start,
                                cur_end,
                                bool(cur_trainable),
                            )
                        )
                        cur_dtype = dtype_here
                        cur_esize = esz
                        cur_start = off
                        cur_end = param_end
                        cur_trainable = trainable_here
                if cur_dtype is not None:
                    dtype_regions.append(
                        (
                            cur_dtype,
                            cur_esize,
                            cur_start,
                            cur_end,
                            bool(cur_trainable),
                        )
                    )

            # No chunk without any regions is shardable (empty chunk).
            if chunk_is_shardable and not dtype_regions:
                chunk_is_shardable = False

            # Compute per-region partition layout (sharded path).
            region_plans: list[dict] = []
            total_shard_bytes = 0
            if chunk_is_shardable:
                for (
                    dtype_r,
                    esize_r,
                    start_off,
                    end_off,
                    trainable_r,
                ) in dtype_regions:
                    region_bytes = end_off - start_off
                    # Pad in element space; must stay byte-compatible with api/reshard.py.
                    elem_count = (region_bytes + esize_r - 1) // esize_r
                    padded_elems = (
                        (elem_count + self.world_size - 1) // self.world_size
                    ) * self.world_size
                    region_bytes_padded = padded_elems * esize_r
                    shard_bytes_r = region_bytes_padded // self.world_size
                    region_plans.append(
                        {
                            "dtype": dtype_r,
                            "esize": esize_r,
                            "chunk_offset": start_off,
                            "region_bytes": region_bytes,
                            "region_bytes_padded": region_bytes_padded,
                            "shard_bytes": shard_bytes_r,
                            "is_trainable": bool(trainable_r),
                        }
                    )
                    total_shard_bytes += shard_bytes_r

            # Replicated: full chunk_bytes pinned; sharded: total_shard_bytes only.
            param_pool_chunk_bytes = (
                total_shard_bytes if chunk_is_shardable else chunk_bytes
            )
            grad_pool_chunk_bytes = 0
            if chunk_is_shardable:
                for plan in region_plans:
                    if plan["is_trainable"]:
                        grad_pool_chunk_bytes += plan["shard_bytes"]
            else:
                for pid, nbytes in zip(param_ids, per_param_bytes, strict=True):
                    if nbytes == 0:
                        continue
                    p = self._params_by_id.get(pid)
                    if p is None or not p.requires_grad:
                        continue
                    grad_pool_chunk_bytes += nbytes

            param_pool_offset = _align_up(total_param_pool_bytes, _INTER_CHUNK_ALIGN)
            total_param_pool_bytes = param_pool_offset + param_pool_chunk_bytes
            grad_pool_offset = _align_up(total_grad_pool_bytes, _INTER_CHUNK_ALIGN)
            total_grad_pool_bytes = grad_pool_offset + grad_pool_chunk_bytes

            chunk_plans.append(
                {
                    "cid": cid,
                    "param_ids": param_ids,
                    "per_param_bytes": per_param_bytes,
                    "element_sizes": element_sizes,
                    "aligned_offsets": aligned_offsets,
                    "chunk_bytes": chunk_bytes,
                    "shardable": chunk_is_shardable,
                    "region_plans": region_plans,
                    "total_shard_bytes": total_shard_bytes,
                    "param_pool_offset": param_pool_offset,
                    "param_pool_chunk_bytes": param_pool_chunk_bytes,
                    "grad_pool_offset": grad_pool_offset,
                    "grad_pool_chunk_bytes": grad_pool_chunk_bytes,
                }
            )

        # ONE precise-size PinnedHostMemory per kind; skip when total bytes == 0.
        param_pool_buf: "torch.Tensor | None" = None
        grad_pool_buf: "torch.Tensor | None" = None
        if total_param_pool_bytes > 0:
            self._cpu_param_pool = PinnedHostMemory(
                n_buffer=1, S_chunk=total_param_pool_bytes
            )
            # Borrow slot 0 for manager lifetime; released in restore_to_gpu / __del__.
            param_pool_buf = self._cpu_param_pool.buffer(0)
        if total_grad_pool_bytes > 0:
            self._cpu_grad_pool = PinnedHostMemory(
                n_buffer=1, S_chunk=total_grad_pool_bytes
            )
            grad_pool_buf = self._cpu_grad_pool.buffer(0)

        # Pass 2: population — slice per-chunk views and populate slots / regions / grads.
        freed = 0
        from torch import nn as _nn

        for plan in chunk_plans:
            cid = plan["cid"]
            param_ids = plan["param_ids"]
            per_param_bytes = plan["per_param_bytes"]
            aligned_offsets = plan["aligned_offsets"]
            chunk_bytes = plan["chunk_bytes"]
            chunk_is_shardable = plan["shardable"]
            region_plans = plan["region_plans"]
            total_shard_bytes = plan["total_shard_bytes"]
            param_pool_offset = plan["param_pool_offset"]
            param_pool_chunk_bytes = plan["param_pool_chunk_bytes"]
            grad_pool_offset = plan["grad_pool_offset"]
            grad_pool_chunk_bytes = plan["grad_pool_chunk_bytes"]

            # Cache for oversize-chunk routing via BufferPool side-table.
            self._chunk_bytes_by_id[cid] = chunk_bytes

            assert param_pool_buf is not None or param_pool_chunk_bytes == 0
            chunk_param_view: "torch.Tensor | None" = None
            if param_pool_chunk_bytes > 0:
                assert param_pool_buf is not None
                chunk_param_view = param_pool_buf.narrow(
                    0, param_pool_offset, param_pool_chunk_bytes
                )
            chunk_grad_view: "torch.Tensor | None" = None
            if grad_pool_chunk_bytes > 0:
                assert grad_pool_buf is not None
                chunk_grad_view = grad_pool_buf.narrow(
                    0, grad_pool_offset, grad_pool_chunk_bytes
                )

            transient_full_chunk: "torch.Tensor | None" = None
            if chunk_is_shardable:
                # Unpinned scratch for the partition copy; released at iteration end.
                transient_full_chunk = torch.empty(chunk_bytes, dtype=torch.uint8)

            slots: list[_CpuParamSlot] = []
            trainable_count = 0
            grad_running_off = 0
            for pid, nbytes, off in zip(
                param_ids, per_param_bytes, aligned_offsets, strict=True
            ):
                param = self._params_by_id.get(pid)
                if param is None or nbytes == 0:
                    continue

                orig_data = param.data
                dtype = orig_data.dtype
                shape = orig_data.shape
                numel = orig_data.numel()
                element_size = orig_data.element_size()

                if chunk_is_shardable:
                    assert transient_full_chunk is not None
                    cpu_view = transient_full_chunk.narrow(0, off, nbytes)
                else:
                    assert chunk_param_view is not None
                    cpu_view = chunk_param_view.narrow(0, off, nbytes)
                cpu_param = cpu_view.view(dtype).view(shape)
                cpu_param.copy_(orig_data)

                # Release GPU storage; opt-in shape-preserving placeholder keeps param.size() correct for autograd.
                if self._shape_preserving_placeholders:
                    param.data = self._shape_preserving_placeholder(shape, dtype)
                else:
                    param.data = self._empty_placeholder(dtype)

                # Replicated only: per-slot grad shadow; sharded uses per-region buffer.
                cpu_grad: "torch.Tensor | None" = None
                if param.requires_grad:
                    trainable_count += 1
                    if not chunk_is_shardable:
                        assert chunk_grad_view is not None
                        grad_byte_view = chunk_grad_view.narrow(
                            0, grad_running_off, nbytes
                        )
                        cpu_grad = grad_byte_view.view(dtype).view(shape)
                        # Pre-zero for tests / first accumulate-grad consumers.
                        cpu_grad.zero_()
                        grad_running_off += nbytes

                # Sharded: slot.cpu_data is None; bytes live in per-region shards.
                slot_cpu_data: "torch.Tensor | None" = None
                if not chunk_is_shardable:
                    slot_cpu_data = cpu_param

                slots.append(
                    _CpuParamSlot(
                        param_id=pid,
                        cpu_data=slot_cpu_data,
                        cpu_grad=cpu_grad,
                        shape=shape,
                        dtype=dtype,
                        byte_offset=off,
                        numel=numel,
                        element_size=element_size,
                    )
                )
                freed += nbytes

            self._cpu_slots[cid] = slots
            self._grad_initial[cid] = trainable_count
            self._grad_remaining[cid] = trainable_count

            if chunk_is_shardable:
                assert transient_full_chunk is not None
                assert chunk_param_view is not None  # holds per-region shards

                regions: list[_DtypeRegion] = []
                region_param_off = 0
                region_grad_off = 0
                for region_plan in region_plans:
                    r_dtype = region_plan["dtype"]
                    r_esize = region_plan["esize"]
                    r_chunk_off = region_plan["chunk_offset"]
                    r_bytes = region_plan["region_bytes"]
                    r_bytes_padded = region_plan["region_bytes_padded"]
                    r_shard_bytes = region_plan["shard_bytes"]
                    r_is_trainable = region_plan["is_trainable"]

                    # Zero-init the padded region so peer ranks don't see uninitialized tail bytes.
                    region_scratch = torch.zeros(r_bytes_padded, dtype=torch.uint8)
                    region_scratch.narrow(0, 0, r_bytes).copy_(
                        transient_full_chunk.narrow(0, r_chunk_off, r_bytes)
                    )

                    my_off = self.rank * r_shard_bytes
                    cpu_region_shard = chunk_param_view.narrow(
                        0, region_param_off, r_shard_bytes
                    )
                    cpu_region_shard.copy_(
                        region_scratch.narrow(0, my_off, r_shard_bytes)
                    )
                    region_param_off += r_shard_bytes

                    # Frozen regions get no grad shard; otherwise Adam's weight decay would rewrite frozen bytes.
                    cpu_region_grad: "torch.Tensor | None" = None
                    if r_is_trainable:
                        assert chunk_grad_view is not None
                        cpu_region_grad = chunk_grad_view.narrow(
                            0, region_grad_off, r_shard_bytes
                        )
                        cpu_region_grad.zero_()
                        region_grad_off += r_shard_bytes

                    # Shard-level nn.Parameter — one flat Adam step per region.
                    shard_numel = r_shard_bytes // r_esize
                    shard_view = cpu_region_shard.view(r_dtype).view(shard_numel)
                    shard_param = _nn.Parameter(
                        shard_view, requires_grad=r_is_trainable
                    )
                    if r_is_trainable and cpu_region_grad is not None:
                        shard_grad_view = cpu_region_grad.view(r_dtype).view(
                            shard_numel
                        )
                        shard_param.grad = shard_grad_view

                    regions.append(
                        _DtypeRegion(
                            chunk_offset=r_chunk_off,
                            region_bytes=r_bytes,
                            region_bytes_padded=r_bytes_padded,
                            shard_bytes=r_shard_bytes,
                            dtype=r_dtype,
                            element_size=r_esize,
                            cpu_shard_bytes=cpu_region_shard,
                            cpu_shard_grad_bytes=cpu_region_grad,
                            shard_param=shard_param,
                            is_trainable=r_is_trainable,
                        )
                    )

                self._chunk_shards[cid] = _ChunkShardState(
                    regions=regions,
                    chunk_bytes=chunk_bytes,
                    shard_bytes=total_shard_bytes,
                )

            # Sharded path still fires per-param hooks for the counter decrement.
            for slot in slots:
                param = self._params_by_id[slot.param_id]
                if not param.requires_grad:
                    continue
                handle = param.register_post_accumulate_grad_hook(
                    self._make_grad_offload_hook(cid, slot)
                )
                self._grad_hook_handles.append(handle)

        precise_param = (
            self._cpu_param_pool.is_precise_size
            if self._cpu_param_pool is not None
            else True
        )
        precise_grad = (
            self._cpu_grad_pool.is_precise_size
            if self._cpu_grad_pool is not None
            else True
        )
        LOG.info(
            "ChunkManager.materialize_offload: offloaded %d non-persistent "
            "chunks to pinned CPU memory (param_pool=%.3f GB, grad_pool=%.3f "
            "GB; precise_size=%s/%s), freed %.3f GB on GPU",
            len(self._cpu_slots),
            total_param_pool_bytes / 1e9,
            total_grad_pool_bytes / 1e9,
            precise_param,
            precise_grad,
            freed / 1e9,
        )

        if self._shape_preserving_placeholders and self.model is not None:
            try:
                protrain_set = self.chunk_managed_param_names()
                registered, original_len = self.register_ddp_ignore_names(protrain_set)
                LOG.info(
                    "ChunkManager.materialize_offload: rebuilt "
                    "model._ddp_params_and_buffers_to_ignore from live "
                    "caller state "
                    "+ %d chunk-managed names (pre-protrain original: %s)",
                    registered,
                    "<unset>" if original_len is None else f"{original_len} names",
                )
            except Exception as _exc:  # noqa: BLE001 — defensive
                LOG.warning(
                    "ChunkManager.materialize_offload: failed to register "
                    "_ddp_params_and_buffers_to_ignore on model: %s",
                    _exc,
                )
        return freed

    def _close_cpu_pools(self) -> None:
        """Release unified pinned-host param/grad pools; callers must drop views first."""
        for attr in ("_cpu_param_pool", "_cpu_grad_pool"):
            pool = getattr(self, attr, None)
            if pool is None:
                continue
            try:
                pool.release_buffer(0)
            except Exception as exc:  # noqa: BLE001 — best-effort
                LOG.debug(
                    "ChunkManager._close_cpu_pools: release_buffer(0) failed on %s: %s",
                    attr,
                    exc,
                )
            try:
                pool.close()
            except Exception as exc:  # noqa: BLE001 — best-effort
                LOG.debug(
                    "ChunkManager._close_cpu_pools: close() failed on %s: %s",
                    attr,
                    exc,
                )
            setattr(self, attr, None)

    def register_ddp_ignore_names(self, names: set[str]) -> tuple[int, int | None]:
        """Register this manager's current DDP-ignore names on ``self.model``."""
        if self.model is None:
            return (0, None)
        return register_protrain_ddp_ignore_names(
            self.model,
            _DDP_IGNORE_OWNER_CHUNK,
            names,
        )

    def restore_to_gpu(self) -> int:
        """Inverse of :meth:`materialize_offload` — move every param back to GPU."""
        # Drain in-flight async CPU Adam so we snapshot a consistent post-step state.
        self.wait_cpu_optim()

        if not self._cpu_slots and not self._persistent_buffers:
            self._restore_protrain_ddp_ignore_snapshot()
            LOG.debug(
                "ChunkManager.restore_to_gpu: nothing offloaded "
                "(no _cpu_slots, no _persistent_buffers), no-op"
            )
            return 0

        import torch

        # Sharded restore needs a live process group for the per-region all_gather.
        if self.zero3_shard and self._chunk_shards:
            if not (
                torch.distributed.is_available() and torch.distributed.is_initialized()
            ):
                raise RuntimeError(
                    "ChunkManager.restore_to_gpu: zero3_shard=True but "
                    "torch.distributed is not initialized. Sharded "
                    "teardown needs a live process group to all_gather "
                    "the per-rank shards back into full chunks before "
                    "rebinding param.data. Call restore_to_gpu BEFORE "
                    "destroy_process_group()."
                )

        moved = 0

        # Route GPU allocations through default-stream heap for canonical allocator state.
        from axolotl.integrations.protrain.runtime.streams import (
            SingleStreamAllocator,
        )

        _on_cuda = self.device.type == "cuda" and torch.cuda.is_available()

        def _alloc_empty(shape, dtype):
            if _on_cuda:
                with SingleStreamAllocator():
                    return torch.empty(shape, dtype=dtype, device=self.device)
            return torch.empty(shape, dtype=dtype, device=self.device)

        _watchdog_on = _SLOW_OFFLOAD_REGATHER_S > 0.0

        def _maybe_warn_slow_restore(cid: ChunkId, elapsed_s: float) -> None:
            if _watchdog_on and elapsed_s >= _SLOW_OFFLOAD_REGATHER_S:
                LOG.warning(
                    "ProTrain SLOW_OFFLOAD_REGATHER: chunk_id=%d phase=%s "
                    "elapsed=%.3fs (threshold=%.2fs; H2D/D2H + NCCL all_gather). "
                    "Sharded=%s.",
                    int(cid),
                    "resume_restore",
                    elapsed_s,
                    _SLOW_OFFLOAD_REGATHER_S,
                    cid in self._chunk_shards,
                )

        # Non-persistent replicated chunks: per-slot copy from pinned CPU.
        for cid, slots in self._cpu_slots.items():
            if cid in self._chunk_shards:
                # Defer to the sharded reassembly pass below.
                continue
            t0 = time.perf_counter() if _watchdog_on else 0.0
            for slot in slots:
                param = self._params_by_id.get(slot.param_id)
                if param is None or slot.cpu_data is None:
                    continue
                gpu_tensor = _alloc_empty(slot.shape, slot.dtype)
                gpu_tensor.copy_(slot.cpu_data)
                param.data = gpu_tensor
                moved += slot.numel * slot.element_size
            if _watchdog_on:
                _maybe_warn_slow_restore(cid, time.perf_counter() - t0)

        # Sharded chunks: per-region all_gather, then per-slot rebind.
        if self.zero3_shard and self._chunk_shards:
            import torch.distributed as dist

            for cid, shard_state in self._chunk_shards.items():
                t0 = time.perf_counter() if _watchdog_on else 0.0
                chunk_buf = _alloc_empty(shard_state.chunk_bytes, torch.uint8)

                for region in shard_state.regions:
                    # Direct alloc — restore is one-shot teardown, don't consult the pool.
                    my_shard_gpu = _alloc_empty(region.shard_bytes, torch.uint8)
                    my_shard_gpu.copy_(region.cpu_shard_bytes, non_blocking=True)

                    gather_scratch = _alloc_empty(
                        region.region_bytes_padded, torch.uint8
                    )
                    dist.all_gather_into_tensor(gather_scratch, my_shard_gpu)

                    chunk_buf.narrow(0, region.chunk_offset, region.region_bytes).copy_(
                        gather_scratch.narrow(0, 0, region.region_bytes)
                    )

                # Per-slot fresh allocation so the next ChunkManager can rebuild under a new layout.
                slots = self._cpu_slots.get(cid, [])
                for slot in slots:
                    param = self._params_by_id.get(slot.param_id)
                    if param is None:
                        continue
                    nbytes = slot.numel * slot.element_size
                    if nbytes == 0:
                        continue
                    byte_view = chunk_buf.narrow(0, slot.byte_offset, nbytes)
                    typed = byte_view.view(slot.dtype).view(slot.shape)
                    gpu_tensor = _alloc_empty(slot.shape, slot.dtype)
                    gpu_tensor.copy_(typed)
                    param.data = gpu_tensor
                    moved += nbytes
                if _watchdog_on:
                    _maybe_warn_slow_restore(cid, time.perf_counter() - t0)

        # Persistent chunks: extract from resident pool buffer into standalone GPU storage.
        for cid, buf in self._persistent_buffers.items():
            t0 = time.perf_counter() if _watchdog_on else 0.0
            # Recompute the same aligned offsets materialize_offload used.
            param_ids = self.layout.chunks[int(cid)]
            offset = 0
            for pid in param_ids:
                param = self._params_by_id.get(pid)
                if param is None:
                    continue
                nbytes = int(param.numel()) * int(param.element_size())
                if nbytes == 0:
                    continue
                esz = int(param.element_size())
                offset = ((offset + esz - 1) // esz) * esz
                byte_view = buf.narrow(0, offset, nbytes)
                typed = byte_view.view(param.data.dtype).view(param.shape)
                gpu_tensor = _alloc_empty(param.shape, param.data.dtype)
                gpu_tensor.copy_(typed)
                param.data = gpu_tensor
                moved += nbytes
                offset += nbytes
            if _watchdog_on:
                _maybe_warn_slow_restore(cid, time.perf_counter() - t0)

        self.uninstall()

        # Drop view-holding state BEFORE closing pinned pools to avoid dangling pointers.
        self._cpu_slots.clear()
        self._chunk_shards.clear()
        self._persistent_buffers.clear()
        self._grad_initial.clear()
        self._grad_remaining.clear()
        self._chunk_bytes_by_id.clear()
        self._empty_by_dtype.clear()
        self._shape_scratch_by_dtype.clear()

        # Callers MUST drop external shard_param/cpu_data/cpu_grad views before this call.
        self._close_cpu_pools()
        self._restore_protrain_ddp_ignore_snapshot()

        LOG.info(
            "ChunkManager.restore_to_gpu: moved %.3f GB back to standalone "
            "GPU storage (non-persistent + persistent combined)",
            moved / 1e9,
        )
        return moved

    def _compute_chunk_bytes(self, chunk_id: ChunkId) -> int:
        """Return the physical byte size of ``chunk_id``'s param payload (cached)."""
        cached = self._chunk_bytes_by_id.get(chunk_id)
        if cached is not None:
            return cached
        param_ids = self.layout.chunks[int(chunk_id)]
        offset = 0
        for pid in param_ids:
            param = self._params_by_id.get(pid)
            if param is None:
                continue
            nbytes = int(param.numel()) * int(param.element_size())
            if nbytes == 0:
                continue
            esz = int(param.element_size())
            offset = ((offset + esz - 1) // esz) * esz
            offset += nbytes
        self._chunk_bytes_by_id[chunk_id] = offset
        return offset

    def _empty_placeholder(self, dtype: "torch.dtype") -> "torch.Tensor":
        """Return a zero-element GPU tensor of ``dtype`` (cached per dtype)."""
        import torch

        from axolotl.integrations.protrain.runtime.streams import (
            SingleStreamAllocator,
        )

        existing = self._empty_by_dtype.get(dtype)
        if existing is not None:
            return existing
        if self.device.type == "cuda" and torch.cuda.is_available():
            with SingleStreamAllocator():
                t = torch.empty(0, device=self.device, dtype=dtype)
        else:
            t = torch.empty(0, device=self.device, dtype=dtype)
        self._empty_by_dtype[dtype] = t
        return t

    def _shape_preserving_placeholder(
        self,
        shape: "torch.Size | tuple[int, ...]",
        dtype: "torch.dtype",
    ) -> "torch.Tensor":
        """Return a zero-stride view of ``shape``/``dtype`` so released params keep their real shape for autograd."""
        import torch

        from axolotl.integrations.protrain.runtime.streams import (
            SingleStreamAllocator,
        )

        scratch = self._shape_scratch_by_dtype.get(dtype)
        if scratch is None:
            if self.device.type == "cuda" and torch.cuda.is_available():
                with SingleStreamAllocator():
                    scratch = torch.empty(1, device=self.device, dtype=dtype)
            else:
                scratch = torch.empty(1, device=self.device, dtype=dtype)
            self._shape_scratch_by_dtype[dtype] = scratch

        if shape == torch.Size([]):
            return scratch.view(())
        return scratch.expand(tuple(shape))

    def chunk_managed_param_names(self) -> set[str]:
        """Return param names backed by released (non-persistent) chunks so DDP can be told to ignore them."""
        fallback_param_ids: list[ParamId] = []
        param_obj_ids: set[int] = set()
        for cid in self._non_persistent_ids:
            for slot in self._cpu_slots.get(cid, []):
                fallback_param_ids.append(slot.param_id)
                param = self._params_by_id.get(slot.param_id)
                if param is not None:
                    param_obj_ids.add(id(param))
        names: set[str] = set()
        resolved_param_ids: set[int] = set()
        for live_name, live_param in self.model.named_parameters():
            live_id = id(live_param)
            if live_id in param_obj_ids:
                names.add(live_name)
                resolved_param_ids.add(live_id)
        for param_id in fallback_param_ids:
            param = self._params_by_id.get(param_id)
            if param is not None and id(param) in resolved_param_ids:
                continue
            if str(param_id):
                names.add(str(param_id))
        return names

    def _make_grad_offload_hook(self, chunk_id: ChunkId, slot: _CpuParamSlot):
        """Build a post-accumulate grad hook for one trainable non-persistent param."""
        cm = self
        captured_slot = slot
        captured_cid = chunk_id

        def _hook(param: "nn.Parameter") -> None:
            if param.grad is None:
                return

            # Sharded fast-path: chunk-level reduce_scatter handles the grad; just decrement.
            shard_state_local = cm._chunk_shards.get(captured_cid)
            if shard_state_local is not None:
                remaining = max(cm._grad_remaining.get(captured_cid, 0) - 1, 0)
                cm._grad_remaining[captured_cid] = remaining
                return

            # Replicated: AVG reduce before drain. Skip when outer DDP owns sync.
            import torch as _torch
            import torch.distributed as _dist

            if (
                _dist.is_available()
                and _dist.is_initialized()
                and _dist.get_world_size() > 1
                and not cm.skip_internal_grad_reduce
            ):
                _dist.all_reduce(param.grad, op=_dist.ReduceOp.AVG)
            accumulating = captured_cid in cm._cpu_step_ready_chunks
            d2h_event = None
            if accumulating:
                prev_event = cm._cpu_step_events.pop(captured_cid, None)
                if prev_event is not None:
                    prev_event.synchronize()
                grad_to_add = param.grad
                if grad_to_add.device != captured_slot.cpu_grad.device:  # type: ignore[union-attr]
                    grad_to_add = grad_to_add.to(captured_slot.cpu_grad.device)  # type: ignore[union-attr]
                captured_slot.cpu_grad.add_(grad_to_add)  # type: ignore[union-attr]
            else:
                captured_slot.cpu_grad.copy_(param.grad, non_blocking=True)  # type: ignore[union-attr]
                # Record D2H event so the CPU-Adam worker can wait before reading the pinned shard.
                if param.grad.is_cuda:
                    d2h_event = _torch.cuda.Event(blocking=True)
                    d2h_event.record()
            param.grad = None

            remaining = cm._grad_remaining.get(captured_cid, 0) - 1
            cm._grad_remaining[captured_cid] = remaining
            if remaining == 0:
                # Fail fast on missing cpu_optim — would silently retain iter-0 weights.
                if cm.cpu_optim is None:
                    raise RuntimeError(
                        "ChunkManager: missing CPU optimizer for offloaded "
                        f"chunk {int(captured_cid)} — DeepSpeedCPUAdam was "
                        "not attached, so the offload step path cannot "
                        "update the CPU master weights. Install "
                        "deepspeed (with a matching CUDA toolchain) or "
                        "configure n_persist == N_chunk so no chunks are "
                        "offloaded."
                    )
                cm._mark_cpu_step_ready(
                    captured_cid,
                    d2h_event=d2h_event,
                    post_step=cm._make_post_cpu_step_repoint(captured_cid),
                )
                # Reset the counter now so the next backward fires again.
                cm._grad_remaining[captured_cid] = cm._grad_initial.get(captured_cid, 0)

        return _hook

    def _mark_cpu_step_ready(
        self,
        chunk_id: ChunkId,
        *,
        d2h_event: Any = None,
        post_step: Any = None,
    ) -> None:
        self._cpu_step_ready_chunks.add(chunk_id)
        if d2h_event is not None:
            self._cpu_step_events[chunk_id] = d2h_event
        if post_step is not None:
            self._cpu_step_post_steps[chunk_id] = post_step

    def _make_post_cpu_step_repoint(self, chunk_id: ChunkId):
        """Build the after-step callback that repoints ``.data`` back to GPU."""
        cm = self
        captured_cid = chunk_id

        def _repoint() -> None:
            slots = cm._cpu_slots.get(captured_cid, [])
            for slot in slots:
                param = cm._params_by_id.get(slot.param_id)
                if param is None:
                    continue
                # Only trainable slots — frozen params would race in-flight backward kernels.
                if not param.requires_grad:
                    continue
                if param.data.device.type != "cpu":
                    continue
                if cm._shape_preserving_placeholders:
                    param.data = cm._shape_preserving_placeholder(
                        slot.shape, slot.dtype
                    )
                else:
                    param.data = cm._empty_placeholder(slot.dtype)
                # Clear grad: otherwise iter N+1 autograd would accumulate GPU grad on CPU tensor.
                param.grad = None

        return _repoint

    def _ensure_cpu_grads_attached(self, chunk_id: ChunkId) -> None:
        """Prepare the non-persistent chunk for its CPU Adam step."""
        slots = self._cpu_slots.get(chunk_id, [])
        for slot in slots:
            param = self._params_by_id.get(slot.param_id)
            if param is None:
                continue
            # Skip frozen: rebinding mid-backward races in-flight kernels (PEFT chunk-sharing).
            if not param.requires_grad:
                continue
            param.data = slot.cpu_data
            param.grad = slot.cpu_grad

    # ---- gather / offload ---------------------------------------------

    def gather(
        self,
        chunk_id: ChunkId,
        phase: str = "forward_regather",
        stream: "torch.cuda.Stream | None" = None,
    ) -> None:
        """Make ``chunk_id``'s params GPU-resident; WARN-logs slow gathers.

        ``phase`` tags the call site for the OFFLOAD-regather watchdog:
        ``"forward_regather"`` (default — initial forward gather of an offloaded
        chunk), ``"backward_regather"`` (re-gather during backward via
        :meth:`gather_for_backward` or the saved-tensor unpack fallback), or
        ``"resume_restore"`` (one-shot teardown from :meth:`restore_to_gpu`).

        ``stream`` (optional) routes the H2D copy + NCCL ``all_gather_into_tensor``
        onto a caller-provided CUDA stream. Used by the scheduler's dedicated
        ``_offload_stream`` to overlap backward re-gather with backward compute.
        When ``None`` the gather runs on whatever stream is current at the call
        site (preserves prior behavior for prefetch / compute callers).
        """
        slow_g = _SLOW_GATHER_THRESHOLD_S > 0.0
        slow_or = _SLOW_OFFLOAD_REGATHER_S > 0.0

        if not slow_g and not slow_or:
            return self._gather_impl(chunk_id, stream=stream, phase=phase)

        t0 = time.perf_counter()
        try:
            self._gather_impl(chunk_id, stream=stream, phase=phase)
        finally:
            elapsed = time.perf_counter() - t0
            if slow_g and elapsed >= _SLOW_GATHER_THRESHOLD_S:
                # WARN so the slow-chunk + elapsed time survive default log filters;
                # narrows the n_offload>0 first-iter hang to a single chunk.
                LOG.warning(
                    "ChunkManager.gather: chunk_id=%d took %.3fs "
                    "(threshold=%.1fs). Sharded=%s active=%s pool_resident=%s. "
                    "If this fires on iter 1 it pinpoints the first-iter hang.",
                    int(chunk_id),
                    elapsed,
                    _SLOW_GATHER_THRESHOLD_S,
                    chunk_id in self._chunk_shards,
                    chunk_id in self._active_chunks,
                    (
                        self.buffer_pool.lookup_resident(chunk_id) is not None
                        if self.buffer_pool is not None
                        else False
                    ),
                )
            if slow_or and elapsed >= _SLOW_OFFLOAD_REGATHER_S:
                # Per-chunk OFFLOAD re-gather wall time (H2D copy + NCCL all_gather).
                # Distinct WARN line so log scrapers can grep SLOW_OFFLOAD_REGATHER.
                LOG.warning(
                    "ProTrain SLOW_OFFLOAD_REGATHER: chunk_id=%d phase=%s "
                    "elapsed=%.3fs (threshold=%.2fs; H2D/D2H + NCCL all_gather). "
                    "Sharded=%s active=%s pool_resident=%s.",
                    int(chunk_id),
                    phase,
                    elapsed,
                    _SLOW_OFFLOAD_REGATHER_S,
                    chunk_id in self._chunk_shards,
                    chunk_id in self._active_chunks,
                    (
                        self.buffer_pool.lookup_resident(chunk_id) is not None
                        if self.buffer_pool is not None
                        else False
                    ),
                )

    def _gather_impl(
        self,
        chunk_id: ChunkId,
        stream: "torch.cuda.Stream | None" = None,
        phase: str = "forward_regather",
    ) -> None:
        """Untimed gather body; split out so the public wrapper can attach watchdog timing.

        ``stream`` (optional) is honored only when provided AND CUDA is available;
        the H2D copy + sharded all_gather are wrapped in ``torch.cuda.stream(stream)``
        so they execute on the caller's stream rather than the default compute stream.

        ``phase`` is threaded to :meth:`_gather_sharded` so the
        ``SLOW_SHARDED_GATHER`` watchdog can tag forward vs backward gathers.
        """
        if chunk_id in self._persistent_ids:
            return

        if chunk_id not in self._cpu_slots:
            return

        if stream is not None:
            try:
                import torch as _torch
            except ImportError:  # pragma: no cover
                _torch = None
            if _torch is not None and _torch.cuda.is_available():
                with _torch.cuda.stream(stream):
                    self._gather_impl_body(chunk_id, phase=phase)
                return
        self._gather_impl_body(chunk_id, phase=phase)

    def _gather_impl_body(
        self, chunk_id: ChunkId, phase: str = "forward_regather"
    ) -> None:
        """Original untimed gather body, kept stream-context-agnostic for both call paths."""
        if self.buffer_pool is None:
            raise RuntimeError(
                "ProTrain invariant violated: "
                "gather() reached the non-persistent path with no buffer_pool; "
                "all-persistent layouts must early-return above"
            )

        # Wait for any in-flight CPU-Adam worker on this chunk; gather mid-step would SIGSEGV.
        cpu_optim = self.cpu_optim
        if cpu_optim is not None:
            wait_fn = getattr(cpu_optim, "wait", None)
            if wait_fn is not None:
                wait_fn(chunk_id)

        shard_state = self._chunk_shards.get(chunk_id)

        trace_internals = _GATHER_INTERNALS_TRACE

        # Active fast path: tag-lookup-only re-gather (lease-idempotent).
        if chunk_id in self._active_chunks:
            resident_buf = self.buffer_pool.lookup_resident(chunk_id)
            if resident_buf is None:
                raise RuntimeError(
                    "ProTrain invariant violated: "
                    f"chunk {chunk_id} marked active but pool has no resident "
                    "tag — _active_chunks invariant violated"
                )
            if trace_internals:
                t0_sr = time.perf_counter()
            self._rebind_params_to_buffer(chunk_id, resident_buf, needs_copy=False)
            if trace_internals:
                LOG.warning(
                    "[rank=%d] chunk_id=%d phase=%s sub_op=split_rebind_active_fastpath "
                    "took %.3fs",
                    self.rank,
                    int(chunk_id),
                    phase,
                    time.perf_counter() - t0_sr,
                )
            return

        if trace_internals:
            t0_bytes = time.perf_counter()
        chunk_bytes = self._compute_chunk_bytes(chunk_id)
        if trace_internals:
            LOG.warning(
                "[rank=%d] chunk_id=%d phase=%s sub_op=compute_chunk_bytes took %.3fs",
                self.rank,
                int(chunk_id),
                phase,
                time.perf_counter() - t0_bytes,
            )

        if trace_internals:
            t0_lookup = time.perf_counter()
        resident_buf = self.buffer_pool.acquire_if_resident(chunk_id)
        if trace_internals:
            LOG.warning(
                "[rank=%d] chunk_id=%d phase=%s sub_op=acquire_if_resident took %.3fs",
                self.rank,
                int(chunk_id),
                phase,
                time.perf_counter() - t0_lookup,
            )
        if resident_buf is not None:
            self._active_chunks.add(chunk_id)
            if trace_internals:
                t0_sr = time.perf_counter()
            self._rebind_params_to_buffer(chunk_id, resident_buf, needs_copy=False)
            if trace_internals:
                LOG.warning(
                    "[rank=%d] chunk_id=%d phase=%s sub_op=split_rebind_resident_hit "
                    "took %.3fs",
                    self.rank,
                    int(chunk_id),
                    phase,
                    time.perf_counter() - t0_sr,
                )
            return

        # Cache miss: acquire fresh slot. Oversize chunks route through _large_buffers.
        if trace_internals:
            t0_acq = time.perf_counter()
        buf = self.buffer_pool.acquire(chunk_id, chunk_bytes=chunk_bytes)
        if trace_internals:
            LOG.warning(
                "[rank=%d] chunk_id=%d phase=%s sub_op=buffer_pool_acquire took %.3fs",
                self.rank,
                int(chunk_id),
                phase,
                time.perf_counter() - t0_acq,
            )
        self._active_chunks.add(chunk_id)
        if shard_state is not None:
            self._gather_sharded(chunk_id, buf, shard_state, phase=phase)
            if trace_internals:
                t0_sr = time.perf_counter()
            self._rebind_params_to_buffer(chunk_id, buf, needs_copy=False)
            if trace_internals:
                LOG.warning(
                    "[rank=%d] chunk_id=%d phase=%s sub_op=split_rebind_post_sharded "
                    "took %.3fs",
                    self.rank,
                    int(chunk_id),
                    phase,
                    time.perf_counter() - t0_sr,
                )
            return

        # Replicated path: per-slot H2D copies directly into the buffer.
        if trace_internals:
            t0_sr = time.perf_counter()
        self._rebind_params_to_buffer(chunk_id, buf, needs_copy=True)
        if trace_internals:
            LOG.warning(
                "[rank=%d] chunk_id=%d phase=%s sub_op=split_rebind_replicated_h2d "
                "took %.3fs",
                self.rank,
                int(chunk_id),
                phase,
                time.perf_counter() - t0_sr,
            )

    def _gather_sharded(
        self,
        chunk_id: ChunkId,
        buf: "torch.Tensor",
        shard_state: "_ChunkShardState",
        phase: str = "forward_regather",
    ) -> None:
        """ZeRO-3 all_gather path: one collective per dtype region.

        ``phase`` tags the call site for the SLOW_SHARDED_GATHER watchdog
        ("forward_regather" / "backward_regather" / "resume_restore"). The
        watchdog times the per-region all_gather hot loop so Mode C
        ``zero3_shard`` stalls on the NCCL collective surface independently
        of OFFLOAD H2D re-gather and the broader gather() wrapper timing.
        """
        import torch
        import torch.distributed as dist

        from axolotl.integrations.protrain.runtime.streams import (
            SingleStreamAllocator,
        )

        slow_sg = _SLOW_SHARDED_GATHER_S > 0.0
        t0_sg = time.perf_counter() if slow_sg else 0.0

        trace_internals = _GATHER_INTERNALS_TRACE

        # Route scratch allocations through default-stream heap; tie lifetime via record_stream.
        cur_stream: "torch.cuda.Stream | None" = None
        on_cuda = buf.device.type == "cuda" and torch.cuda.is_available()
        if on_cuda:
            cur_stream = torch.cuda.current_stream(device=buf.device)
        # Skip the wrap on default stream / CPU-only paths.
        wrap_alloc = (
            on_cuda
            and cur_stream is not None
            and (cur_stream != torch.cuda.default_stream(device=buf.device))
        )

        for region_idx, region in enumerate(shard_state.regions):
            # Staging: this rank's shard on GPU.
            if trace_internals:
                t0_alloc_shard = time.perf_counter()
            if wrap_alloc:
                with SingleStreamAllocator():
                    my_shard_gpu = torch.empty(
                        region.shard_bytes, dtype=torch.uint8, device=buf.device
                    )
                # record_stream ties lifetime to the consuming stream.
                my_shard_gpu.record_stream(cur_stream)  # type: ignore[arg-type]
            else:
                my_shard_gpu = torch.empty(
                    region.shard_bytes, dtype=torch.uint8, device=buf.device
                )
            if trace_internals:
                LOG.warning(
                    "[rank=%d] chunk_id=%d phase=%s sub_op=alloc_shard_gpu "
                    "region=%d took %.3fs",
                    self.rank,
                    int(chunk_id),
                    phase,
                    region_idx,
                    time.perf_counter() - t0_alloc_shard,
                )
                t0_h2d = time.perf_counter()
            my_shard_gpu.copy_(region.cpu_shard_bytes, non_blocking=True)
            if trace_internals:
                LOG.warning(
                    "[rank=%d] chunk_id=%d phase=%s sub_op=h2d_copy "
                    "region=%d took %.3fs",
                    self.rank,
                    int(chunk_id),
                    phase,
                    region_idx,
                    time.perf_counter() - t0_h2d,
                )

            # Gather output scratch: region_bytes_padded (may be > region_bytes).
            if trace_internals:
                t0_alloc_gs = time.perf_counter()
            if wrap_alloc:
                with SingleStreamAllocator():
                    gather_scratch = torch.empty(
                        region.region_bytes_padded,
                        dtype=torch.uint8,
                        device=buf.device,
                    )
                gather_scratch.record_stream(cur_stream)  # type: ignore[arg-type]
            else:
                gather_scratch = torch.empty(
                    region.region_bytes_padded,
                    dtype=torch.uint8,
                    device=buf.device,
                )
            if trace_internals:
                LOG.warning(
                    "[rank=%d] chunk_id=%d phase=%s sub_op=alloc_gather_scratch "
                    "region=%d took %.3fs",
                    self.rank,
                    int(chunk_id),
                    phase,
                    region_idx,
                    time.perf_counter() - t0_alloc_gs,
                )
                t0_ag = time.perf_counter()
            dist.all_gather_into_tensor(gather_scratch, my_shard_gpu)
            if trace_internals:
                LOG.warning(
                    "[rank=%d] chunk_id=%d phase=%s sub_op=all_gather_issue "
                    "region=%d took %.3fs",
                    self.rank,
                    int(chunk_id),
                    phase,
                    region_idx,
                    time.perf_counter() - t0_ag,
                )

            if trace_internals:
                t0_narrow = time.perf_counter()
            buf.narrow(0, region.chunk_offset, region.region_bytes).copy_(
                gather_scratch.narrow(0, 0, region.region_bytes)
            )
            if trace_internals:
                LOG.warning(
                    "[rank=%d] chunk_id=%d phase=%s sub_op=narrow_copy_to_buf "
                    "region=%d took %.3fs",
                    self.rank,
                    int(chunk_id),
                    phase,
                    region_idx,
                    time.perf_counter() - t0_narrow,
                )

        if slow_sg:
            elapsed_sg = time.perf_counter() - t0_sg
            if elapsed_sg >= _SLOW_SHARDED_GATHER_S:
                # Distinct WARN line so log scrapers can grep SLOW_SHARDED_GATHER.
                LOG.warning(
                    "ProTrain SLOW_SHARDED_GATHER: chunk_id=%d phase=%s "
                    "elapsed=%.3fs (threshold=%.2fs; NCCL all_gather_into_tensor "
                    "across %d region(s) for zero3_shard). Mode C bs=2 forward "
                    "hang site; expect this to be silent post-fix.",
                    int(chunk_id),
                    phase,
                    elapsed_sg,
                    _SLOW_SHARDED_GATHER_S,
                    len(shard_state.regions),
                )

    def _rebind_params_to_buffer(
        self,
        chunk_id: ChunkId,
        buf: "torch.Tensor",
        needs_copy: bool,
    ) -> None:
        """Copy CPU shards into ``buf`` (if needed) and rebind each param's data."""
        slots = self._cpu_slots.get(chunk_id, [])
        if not slots:
            return

        trace_internals = _GATHER_INTERNALS_TRACE

        if needs_copy:
            if trace_internals:
                t0_inner_h2d = time.perf_counter()
            for slot in slots:
                nbytes = slot.numel * slot.element_size
                dst_bytes = buf.narrow(0, slot.byte_offset, nbytes)
                dst_typed = dst_bytes.view(slot.dtype).view(slot.shape)
                dst_typed.copy_(slot.cpu_data, non_blocking=True)
            if trace_internals:
                LOG.warning(
                    "[rank=%d] chunk_id=%d phase=rebind sub_op=replicated_slot_h2d "
                    "slots=%d took %.3fs",
                    self.rank,
                    int(chunk_id),
                    len(slots),
                    time.perf_counter() - t0_inner_h2d,
                )

        if trace_internals:
            t0_bind = time.perf_counter()
        # Rebind .data unconditionally — previous offload() nulled it.
        for slot in slots:
            param = self._params_by_id.get(slot.param_id)
            if param is None:
                continue
            nbytes = slot.numel * slot.element_size
            byte_view = buf.narrow(0, slot.byte_offset, nbytes)
            typed = byte_view.view(slot.dtype).view(slot.shape)
            param.data = typed
        if trace_internals:
            LOG.warning(
                "[rank=%d] chunk_id=%d phase=rebind sub_op=bind_param_data_to_chunk_view "
                "slots=%d took %.3fs",
                self.rank,
                int(chunk_id),
                len(slots),
                time.perf_counter() - t0_bind,
            )

        # Register storage ptr → chunk_id for OffloadedBlock._pack.
        try:
            ptr = buf.untyped_storage().data_ptr()
        except Exception:  # noqa: BLE001 — defensive on unusual backends
            ptr = 0
        if ptr:
            self._storage_ptr_to_chunk[ptr] = chunk_id

    def offload(self, chunk_id: ChunkId) -> None:
        """Release ``chunk_id``'s GPU storage (non-persistent only)."""
        if chunk_id in self._persistent_ids:
            return
        if self.buffer_pool is None:
            raise RuntimeError(
                "ProTrain invariant violated: "
                "offload() reached the non-persistent path with no buffer_pool; "
                "all-persistent layouts must early-return above"
            )

        # Defer if any BackwardHandle still pins this chunk; drain on handle drop.
        if self._backward_refcount.get(chunk_id, 0) > 0:
            self._deferred_offloads.add(chunk_id)
            return

        # Deregister storage-ptr reverse lookup before nulling param.data.
        slots = self._cpu_slots.get(chunk_id, [])
        for slot in slots:
            param = self._params_by_id.get(slot.param_id)
            if param is None:
                continue
            try:
                ptr = param.data.untyped_storage().data_ptr()
            except Exception:  # noqa: BLE001
                ptr = 0
            if ptr and self._storage_ptr_to_chunk.get(ptr) == chunk_id:
                self._storage_ptr_to_chunk.pop(ptr, None)
                # One ptr per chunk; break.
                break

        for slot in slots:
            param = self._params_by_id.get(slot.param_id)
            if param is None:
                continue
            # CPU-bound param.data is mid-CPU-Adam-step; post_step will repoint.
            if param.data.device.type == "cpu":
                continue
            if self._shape_preserving_placeholders:
                param.data = self._shape_preserving_placeholder(slot.shape, slot.dtype)
            else:
                param.data = self._empty_placeholder(slot.dtype)
        self.buffer_pool.release(chunk_id)
        # Symmetric with gather()'s _active_chunks.add; deferred path keeps the lease.
        self._active_chunks.discard(chunk_id)

    def reduce_grads_and_offload(
        self, chunk_id: ChunkId, *, force: bool = False
    ) -> None:
        """Reduce-scatter grads and D2H-copy the chunk's grad shard back to CPU."""
        import torch

        if chunk_id in self._persistent_ids:
            # ProTrain owns reduction (no outer DDP): coalesce same-dtype grads per chunk.
            if (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
                and torch.distributed.get_world_size() > 1
                and not self.skip_internal_grad_reduce
            ):
                self._coalesced_all_reduce_persistent_grads(chunk_id)
            return

        shard_state = self._chunk_shards.get(chunk_id)
        if shard_state is not None:
            finalized = self._reduce_scatter_and_offload_shard(
                chunk_id, shard_state, force=force
            )
            if finalized:
                self.offload(chunk_id)
            return

        # Non-persistent replicated: per-param hooks already drained grads; release the buffer.
        self.offload(chunk_id)

    def reduce_grads_and_offload_from_backward(self, chunk_id: ChunkId) -> None:
        """Finalize a backward block without launching sharded CPU optimizer work."""
        if chunk_id in self._persistent_ids:
            return

        if chunk_id in self._chunk_shards:
            self.offload(chunk_id)
            return

        self.reduce_grads_and_offload(chunk_id)

    def sync_persistent_grads_for_step(self) -> None:
        """Synchronize persistent grads not reached by block-owned finalization."""
        import torch

        if (
            not self._persistent_ids
            or self.skip_internal_grad_reduce
            or not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
            or torch.distributed.get_world_size() <= 1
        ):
            return

        for cid in sorted(self._persistent_ids):
            if cid in self._persistent_grads_synced:
                continue
            self._coalesced_all_reduce_persistent_grads(cid)

    def reset_optimizer_step_tracking(self) -> None:
        self._persistent_grads_synced.clear()

    def _coalesced_all_reduce_persistent_grads(self, chunk_id: ChunkId) -> None:
        """Bucket persistent-chunk grads by dtype and issue one all_reduce per bucket."""
        import torch.distributed as dist
        from torch._utils import (
            _flatten_dense_tensors,
            _unflatten_dense_tensors,
        )

        # Per-dtype groups; param order matters for unflatten to land on the right grads.
        grads_by_dtype: dict[
            "torch.dtype", list[tuple["torch.Tensor", "torch.Tensor"]]
        ] = {}
        for pid in self.layout.chunks[int(chunk_id)]:
            param = self._params_by_id.get(pid)
            if param is None or param.grad is None:
                continue
            grads_by_dtype.setdefault(param.grad.dtype, []).append(
                (param.grad, param.grad)  # (input_view, target_for_writeback)
            )

        did_sync = False
        for _dtype, pairs in grads_by_dtype.items():
            if not pairs:
                continue
            grads = [p[0] for p in pairs]
            if len(grads) == 1:
                # Skip flatten/unflatten for a single grad.
                dist.all_reduce(grads[0], op=dist.ReduceOp.AVG)
                did_sync = True
                continue

            # Flatten → reduce → copy back into original storage (unflatten aliases the flat buffer).
            flat = _flatten_dense_tensors(grads)
            dist.all_reduce(flat, op=dist.ReduceOp.AVG)
            did_sync = True
            for orig, view in zip(
                grads, _unflatten_dense_tensors(flat, grads), strict=True
            ):
                orig.copy_(view)
        if did_sync:
            self._persistent_grads_synced.add(chunk_id)

    def _reduce_scatter_and_offload_shard(
        self, chunk_id: ChunkId, shard_state: "_ChunkShardState", *, force: bool = False
    ) -> bool:
        """Sharded path: reduce_scatter chunk grads, D2H shard, kick CPU Adam."""
        import torch
        import torch.distributed as dist

        from axolotl.integrations.protrain.runtime.streams import (
            SingleStreamAllocator,
        )

        slots = self._cpu_slots.get(chunk_id, [])
        if not slots:
            return False

        if self._grad_remaining.get(chunk_id, 0) > 0 and not force:
            return False

        device = self.device
        any_grad = False
        for slot in slots:
            p = self._params_by_id.get(slot.param_id)
            if p is not None and p.grad is not None:
                device = p.grad.device
                any_grad = True
                break
        if not any_grad:
            return bool(force)

        # current_stream is a syscall; compute once outside the loop.
        on_cuda = device.type == "cuda" and torch.cuda.is_available()
        cur_stream: "torch.cuda.Stream | None" = None
        wrap_alloc = False
        if on_cuda:
            cur_stream = torch.cuda.current_stream(device=device)
            wrap_alloc = cur_stream != torch.cuda.default_stream(device=device)

        d2h_event = None
        any_trainable_region = False
        accumulating = chunk_id in self._cpu_step_ready_chunks
        if accumulating:
            prev_event = self._cpu_step_events.pop(chunk_id, None)
            if prev_event is not None:
                prev_event.synchronize()
        for region in shard_state.regions:
            # Frozen regions have no grad shard; reducing here would let weight-decay mutate frozen bytes.
            if not region.is_trainable:
                continue
            any_trainable_region = True

            r_start = region.chunk_offset
            r_end = r_start + region.region_bytes

            # Padded buffer matches reduce_scatter input length; trailing pad stays zero.
            if wrap_alloc:
                with SingleStreamAllocator():
                    region_grad = torch.zeros(
                        region.region_bytes_padded,
                        dtype=torch.uint8,
                        device=device,
                    )
                region_grad.record_stream(cur_stream)  # type: ignore[arg-type]
            else:
                region_grad = torch.zeros(
                    region.region_bytes_padded,
                    dtype=torch.uint8,
                    device=device,
                )
            for slot in slots:
                if slot.byte_offset < r_start:
                    continue
                if slot.byte_offset >= r_end:
                    break
                p = self._params_by_id.get(slot.param_id)
                if p is None or p.grad is None:
                    continue
                nbytes = slot.numel * slot.element_size
                rel_off = slot.byte_offset - r_start
                dst_bytes = region_grad.narrow(0, rel_off, nbytes)
                dst_typed = dst_bytes.view(slot.dtype).view(slot.shape)
                dst_typed.copy_(p.grad)
                p.grad = None

            shard_numel_r = region.shard_bytes // region.element_size
            full_numel_r = region.region_bytes_padded // region.element_size
            region_grad_typed = region_grad.view(region.dtype).view(full_numel_r)
            if wrap_alloc:
                with SingleStreamAllocator():
                    my_shard_grad_gpu = torch.empty(
                        shard_numel_r, dtype=region.dtype, device=device
                    )
                my_shard_grad_gpu.record_stream(cur_stream)  # type: ignore[arg-type]
            else:
                my_shard_grad_gpu = torch.empty(
                    shard_numel_r, dtype=region.dtype, device=device
                )
            dist.reduce_scatter_tensor(
                my_shard_grad_gpu,
                region_grad_typed,
                op=dist.ReduceOp.AVG,
            )

            # Re-bind .grad to pinned CPU view if zero_grad(set_to_none=True) cleared it.
            if region.cpu_shard_grad_bytes is None:
                raise RuntimeError(
                    "ProTrain invariant violated: "
                    "region.cpu_shard_grad_bytes is None during reduce-scatter "
                    "grad rebind; CPU-side shard grad buffer must be allocated"
                )
            if region.shard_param.grad is None:
                region.shard_param.grad = region.cpu_shard_grad_bytes.view(
                    region.dtype
                ).view(shard_numel_r)

            if my_shard_grad_gpu.is_cuda:
                if accumulating:
                    grad_to_add = my_shard_grad_gpu.to(region.shard_param.grad.device)  # type: ignore[union-attr]
                    region.shard_param.grad.add_(grad_to_add)  # type: ignore[union-attr]
                else:
                    region.shard_param.grad.copy_(  # type: ignore[union-attr]
                        my_shard_grad_gpu, non_blocking=True
                    )
                ev = torch.cuda.Event(blocking=True)
                ev.record()
                # Last region's event suffices: all D2Hs share the same stream.
                d2h_event = ev
            else:
                if accumulating:
                    region.shard_param.grad.add_(my_shard_grad_gpu)  # type: ignore[union-attr]
                else:
                    region.shard_param.grad.copy_(my_shard_grad_gpu)  # type: ignore[union-attr]

        # Raise before resetting counter so missing cpu_optim re-fires next backward.
        if any_trainable_region and self.cpu_optim is None:
            raise RuntimeError(
                "ChunkManager: missing CPU optimizer for offloaded "
                f"chunk {int(chunk_id)} — DeepSpeedCPUAdam was not "
                "attached, so the sharded reduce_scatter/offload path "
                "cannot update the CPU master weights. Install "
                "deepspeed (with a matching CUDA toolchain) or "
                "configure n_persist == N_chunk so no chunks are "
                "offloaded."
            )

        self._grad_remaining[chunk_id] = self._grad_initial.get(chunk_id, 0)

        if self.cpu_optim is not None:
            self._mark_cpu_step_ready(chunk_id, d2h_event=d2h_event, post_step=None)
        return True

    # ---- optimizer driver ---------------------------------------------

    def persistent_step(self) -> None:
        """Run the synchronous GPU FusedAdam step over persistent chunks."""
        if self.gpu_optim is None:
            return
        self.gpu_optim.step()

    def wait_cpu_optim(self) -> None:
        """Block until every in-flight CPU Adam step has finished."""
        if self.cpu_optim is not None:
            self.cpu_optim.wait_all()

    def wait_cpu_optim_all(self) -> None:
        """Alias of :meth:`wait_cpu_optim` for the public optim wrapper."""
        self.wait_cpu_optim()

    def step_ready_cpu_chunks(self) -> None:
        """Launch CPU Adam once for each chunk finalized since the last optimizer step."""
        if not self._cpu_step_ready_chunks:
            return
        if self.cpu_optim is None:
            raise RuntimeError(
                "ChunkManager: missing CPU optimizer for finalized offloaded "
                "chunk grads; cannot apply the optimizer step."
            )
        ready = sorted(self._cpu_step_ready_chunks)
        self._cpu_step_ready_chunks.clear()
        for chunk_id in ready:
            if chunk_id not in self._chunk_shards:
                self._ensure_cpu_grads_attached(chunk_id)
            d2h_event = self._cpu_step_events.pop(chunk_id, None)
            post_step = self._cpu_step_post_steps.pop(chunk_id, None)
            self.cpu_optim.step_async(
                chunk_id,
                d2h_event=d2h_event,
                post_step=post_step,
            )

    # ---- cleanup -------------------------------------------------------

    def uninstall(self) -> None:
        """Remove every registered per-param grad hook. Idempotent."""
        for handle in self._grad_hook_handles:
            try:
                handle.remove()  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001 — best-effort
                LOG.debug("ChunkManager.uninstall: hook remove failed: %s", exc)
        self._grad_hook_handles.clear()

    def _restore_protrain_ddp_ignore_snapshot(self) -> None:
        """Remove this manager's DDP-ignore entries from the live model list."""
        model = self.model
        if model is None:
            return
        if not hasattr(model, _DDP_ORIGINAL_IGNORE_ATTR) and not hasattr(
            model, _DDP_IGNORE_OWNERS_ATTR
        ):
            return
        try:
            removed, remaining = restore_protrain_ddp_ignore_names(
                model,
                _DDP_IGNORE_OWNER_CHUNK,
            )
            LOG.info(
                "ChunkManager: removed %d chunk-managed DDP-ignore name(s); "
                "%d live name(s) remain on model._ddp_params_and_buffers_to_ignore",
                removed,
                remaining,
            )
        except Exception as exc:  # noqa: BLE001 — best-effort
            LOG.debug(
                "ChunkManager._restore_protrain_ddp_ignore_snapshot failed: %s",
                exc,
            )

    def close(self) -> None:
        """Tear down every manager-owned resource. Idempotent."""
        if self._closed:
            return
        self._closed = True

        if self.cpu_optim is not None:
            try:
                self.cpu_optim.shutdown()
            except Exception as exc:  # noqa: BLE001 — best-effort
                LOG.debug("ChunkManager.close: cpu_optim.shutdown failed: %s", exc)

        try:
            self.uninstall()
        except Exception as exc:  # noqa: BLE001 — best-effort
            LOG.debug("ChunkManager.close: uninstall failed: %s", exc)

        self._cpu_slots.clear()
        self._chunk_shards.clear()
        self._persistent_buffers.clear()
        self._storage_ptr_to_chunk.clear()
        self._active_chunks.clear()
        self._backward_refcount.clear()
        self._deferred_offloads.clear()
        self._grad_remaining.clear()
        self._grad_initial.clear()
        self._cpu_step_ready_chunks.clear()
        self._cpu_step_events.clear()
        self._cpu_step_post_steps.clear()
        self._persistent_grads_synced.clear()
        self._chunk_bytes_by_id.clear()
        self._empty_by_dtype.clear()
        self._shape_scratch_by_dtype.clear()

        try:
            self._close_cpu_pools()
        except Exception as exc:  # noqa: BLE001 — best-effort
            LOG.debug("ChunkManager.close: _close_cpu_pools failed: %s", exc)

        if self.buffer_pool is not None:
            try:
                self.buffer_pool.close()
            except Exception as exc:  # noqa: BLE001 — best-effort
                LOG.debug("ChunkManager.close: buffer_pool.close failed: %s", exc)
            self.buffer_pool = None

        self.cpu_optim = None
        self.gpu_optim = None

        try:
            removed, remaining = restore_all_protrain_ddp_ignore_names(self.model)
            if removed:
                LOG.info(
                    "ChunkManager.close: removed %d total ProTrain DDP-ignore "
                    "name(s); %d caller-owned name(s) remain",
                    removed,
                    remaining,
                )
        except Exception as exc:  # noqa: BLE001 — best-effort
            LOG.debug(
                "ChunkManager.close: snapshot restore failed: %s",
                exc,
            )

    def __del__(self) -> None:  # noqa: D401
        try:
            self.uninstall()
        except Exception:  # noqa: BLE001 — destructors must not throw
            pass
        try:
            # GC safety net for the unified pinned pools.
            self._close_cpu_pools()
        except Exception:  # noqa: BLE001 — destructors must not throw
            pass

    # ---- M2 / Option B: backward-window pinning -----------------------

    def chunk_id_for_storage_ptr(self, ptr: int) -> "ChunkId | None":
        """Look up the chunk whose pool buffer storage starts at ``ptr``."""
        return self._storage_ptr_to_chunk.get(ptr)

    def gather_for_backward(
        self,
        chunk_id: ChunkId,
        stream: "torch.cuda.Stream | None" = None,
    ) -> "BackwardHandle":
        """Re-gather a chunk for the backward pass and pin it via refcount.

        ``stream`` (optional) routes the underlying H2D + NCCL all_gather onto
        a dedicated stream (e.g. the scheduler's ``_offload_stream``) so the
        re-gather overlaps backward compute instead of serializing with it.
        """
        self.gather(chunk_id, phase="backward_regather", stream=stream)
        self._backward_refcount[chunk_id] = self._backward_refcount.get(chunk_id, 0) + 1
        return BackwardHandle(chunk_id, self)

    def _release_backward_handle(self, chunk_id: ChunkId) -> None:
        """Decrement ``chunk_id``'s refcount and drain any deferred offload."""
        cur = self._backward_refcount.get(chunk_id, 0)
        if cur <= 1:
            self._backward_refcount.pop(chunk_id, None)
            if chunk_id in self._deferred_offloads:
                self._deferred_offloads.discard(chunk_id)
                self.offload(chunk_id)
        else:
            self._backward_refcount[chunk_id] = cur - 1

    def drain_deferred_offloads(self) -> int:
        """Flush every deferred offload whose backward refcount is now zero."""
        drained = 0
        for cid in tuple(self._deferred_offloads):
            if self._backward_refcount.get(cid, 0) > 0:
                continue
            # Pop before offload so the deferral guard sees refcount==0.
            self._deferred_offloads.discard(cid)
            self.offload(cid)
            drained += 1
        return drained

    # ---- introspection for tests --------------------------------------

    def sharded_chunk_ids(self) -> list[ChunkId]:
        """Return the list of chunks currently held in ZeRO-3 sharded form."""
        return sorted(self._chunk_shards.keys())

    def shard_bytes_for(self, chunk_id: ChunkId) -> int:
        """Return this rank's total pinned CPU shard bytes for ``chunk_id``."""
        s = self._chunk_shards.get(chunk_id)
        return 0 if s is None else s.shard_bytes

    def per_rank_cpu_bytes(self) -> int:
        """Total pinned CPU bytes this rank holds across every sharded chunk."""
        total = 0
        for shard_state in self._chunk_shards.values():
            for region in shard_state.regions:
                total += int(region.cpu_shard_bytes.numel())
                if region.cpu_shard_grad_bytes is not None:
                    total += int(region.cpu_shard_grad_bytes.numel())
        return total

    def replicated_cpu_bytes(self) -> int:
        """Total pinned CPU bytes this rank holds in replicated (non-sharded) mode."""
        total = 0
        for slots in self._cpu_slots.values():
            for s in slots:
                if s.cpu_data is not None:
                    total += s.numel * s.element_size
                if s.cpu_grad is not None:
                    total += s.numel * s.element_size
        return total

    # ---- CPU-state snapshot / restore (phase-2 rollback support) -------

    def snapshot_cpu_state(self) -> dict[ChunkId, dict[str, Any]]:
        """Deep-clone the CPU-resident weight bytes for every non-persistent chunk."""
        snap: dict[ChunkId, dict[str, Any]] = {}
        for cid in sorted(self._non_persistent_ids):
            entry: dict[str, Any] = {"slots": None, "regions": None}
            shard_state = self._chunk_shards.get(cid)
            if shard_state is not None:
                # Clone every region — frozen regions too, for exact restore.
                region_snaps: list["torch.Tensor"] = []
                for region in shard_state.regions:
                    region_snaps.append(region.cpu_shard_bytes.detach().clone())
                entry["regions"] = region_snaps
            slots = self._cpu_slots.get(cid)
            if slots is not None and shard_state is None:
                slot_snaps: list["torch.Tensor | None"] = []
                for slot in slots:
                    if slot.cpu_data is None:
                        slot_snaps.append(None)
                    else:
                        slot_snaps.append(slot.cpu_data.detach().clone())
                entry["slots"] = slot_snaps
            snap[cid] = entry
        return snap

    def restore_cpu_state(self, snapshot: dict[ChunkId, dict[str, Any]]) -> None:
        """Inverse of :meth:`snapshot_cpu_state` — copy bytes back in place."""
        restored_cids: list[ChunkId] = []
        for cid, entry in snapshot.items():
            shard_state = self._chunk_shards.get(cid)
            region_snaps = entry.get("regions")
            if region_snaps is not None and shard_state is not None:
                if len(region_snaps) != len(shard_state.regions):
                    continue
                for region, region_snap in zip(
                    shard_state.regions, region_snaps, strict=True
                ):
                    region.cpu_shard_bytes.copy_(region_snap)
                restored_cids.append(cid)
                continue
            slot_snaps = entry.get("slots")
            slots = self._cpu_slots.get(cid)
            if slot_snaps is None or slots is None:
                continue
            if len(slot_snaps) != len(slots):
                continue
            for slot, slot_snap in zip(slots, slot_snaps, strict=True):
                if slot_snap is None or slot.cpu_data is None:
                    continue
                slot.cpu_data.copy_(slot_snap)
            restored_cids.append(cid)
        # Force-offload orphan-leased chunks (e.g. LoRA + non-block params) before invalidating tags.
        if self.buffer_pool is not None and restored_cids:
            # Snapshot the active set first; offload mutates _active_chunks mid-iteration.
            still_active = [cid for cid in restored_cids if cid in self._active_chunks]
            for cid in still_active:
                self.offload(cid)

        # Invalidate resident-tag entries so the next gather re-copies fresh CPU bytes.
        if self.buffer_pool is not None and restored_cids:
            for cid in restored_cids:
                self.buffer_pool.invalidate_tag(cid)

    # ---- internals -----------------------------------------------------

    def _ensure_persistent_buffer(self, chunk_id: ChunkId) -> "torch.Tensor":
        """Lazily materialize the resident GPU buffer for a persistent chunk."""
        existing = self._persistent_buffers.get(chunk_id)
        if existing is not None:
            return existing
        import torch

        from axolotl.integrations.protrain.runtime.streams import (
            SingleStreamAllocator,
        )

        # Size for oversize chunks: use max(S_chunk, chunk_bytes) so narrow() never overflows.
        chunk_bytes = self._compute_chunk_bytes(chunk_id)
        buf_bytes = max(int(self.layout.S_chunk), int(chunk_bytes))
        if self.device.type == "cuda" and torch.cuda.is_available():
            with SingleStreamAllocator():
                buf = torch.empty(
                    buf_bytes,
                    dtype=torch.uint8,
                    device=self.device,
                )
        else:
            buf = torch.empty(
                buf_bytes,
                dtype=torch.uint8,
                device=self.device,
            )
        self._persistent_buffers[chunk_id] = buf
        return buf


__all__ = ["BackwardHandle", "ChunkManager"]
