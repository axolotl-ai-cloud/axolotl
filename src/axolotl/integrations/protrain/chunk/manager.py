"""Per-rank chunk manager driving the persistent / non-persistent split.

The :class:`ChunkManager` owns the runtime behavior of a :class:`ChunkLayout`:

* Persistent chunks (``chunk_id < n_persist``) stay resident on GPU,
  updated in place by the GPU FusedAdam adapter.
* Non-persistent chunks are sharded across ranks, offloaded to CPU as
  pinned host tensors, gathered into a pool buffer on demand, and
  reduce-scatter'd + D2H-copied on the backward sweep.

All ``torch.distributed`` calls are guarded with
``torch.distributed.is_initialized()`` so single-rank unit tests don't
require an initialized process group.

M4.5 runtime-primitives additions
---------------------------------

:meth:`materialize_offload` physically moves every non-persistent chunk's
param data from GPU to pinned CPU memory and replaces the GPU storage
with an empty placeholder tensor — this is what closes the paper's
"non-persistent chunks live on CPU" promise end-to-end (Gap 1). The
method is idempotent and must be called exactly once after the chunk
manager is constructed but before the first :meth:`gather` / any
forward pass. :func:`protrain_model_wrapper` drives this from step 4.5
of its construction sequence.

:meth:`_offload_grad` — per-parameter post-accumulate grad hook installed
on every trainable non-persistent param by :meth:`materialize_offload`
(Gap 2). Fires the instant PyTorch autograd accumulates a grad, copies
it to a pinned CPU grad shard, nulls ``param.grad`` on GPU, and — once
every param in the chunk has contributed — enqueues the async CPU
FusedAdam step. This is what keeps GPU grad pressure ≈ zero for
non-persistent chunks during backward, matching ZeRO-Offload's invariant.

Paper references: §3.1.1, §5; ZeRO-Offload's per-param hook pattern.

M7: true ZeRO-3 chunk sharding
------------------------------

When ``zero3_shard=True`` is set on construction (driven automatically
by ``protrain_model_wrapper`` when ``world_size > 1`` AND no outer DDP
wrapper is detected), every non-persistent chunk's bytes are partitioned
across ranks on CPU: each rank keeps only ``ceil(chunk_bytes / world_size)``
pinned bytes — the ``rank``-th slice of the full chunk's byte layout.

* :meth:`gather` in sharded mode H2D-uploads this rank's CPU shard then
  issues ``torch.distributed.all_gather_into_tensor`` to reconstruct the
  full chunk into the pool buffer — every rank gets a bit-identical full
  copy for forward / backward compute.
* :meth:`reduce_grads_and_offload` for non-persistent chunks in sharded
  mode flattens the chunk's GPU grads into a contiguous buffer, issues
  ``torch.distributed.reduce_scatter_tensor(op=AVG)`` so each rank
  receives only its slice of the reduced-average grad, then D2H-copies
  the slice to the rank's pinned CPU grad shard and kicks the CPU
  FusedAdam step against the shard (CPU Adam is built over a single
  shard-flat ``nn.Parameter`` — see ``materialize_offload``).

The sharded path handles BOTH homogeneous-dtype and mixed-dtype
chunks. Each chunk is modelled as an ordered list of
:class:`_DtypeRegion` entries — one per maximal-length contiguous
same-dtype byte run — and each region is independently partitioned
across ranks. Gather/reduce issues one collective per region: a
homogeneous chunk lands exactly one collective (identical to the
pre-followup behaviour), a Llama block with fp32 RMSNorm scales
between fp16 linear layers lands 3. Shard boundaries are padded up to
``lcm(region_element_size, world_size)`` so every ``.view(dtype)``
after ``all_gather`` lands on a clean element boundary. Params
straddling a shard boundary within a region are partitioned across
two ranks' shards and reassembled on gather by ``all_gather``.

Persistent chunks are FULLY REPLICATED even in sharded mode — they're
small, live on GPU, and the FusedAdam step runs locally on each rank.
The persistent branch of :meth:`reduce_grads_and_offload` still uses
per-param ``all_reduce(op=AVG)`` when ``zero3_shard=True`` (unchanged
from the non-sharded path).

Paper references: §1 (parallelism foundation), §2A (chunks), §5
(low-level overlaps).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from axolotl.integrations.protrain.types import (
    ChunkId,
    ChunkLayout,
    ParamId,
)
from axolotl.utils.logging import get_logger

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


class _CpuParamSlot:
    """Per-parameter bookkeeping for a non-persistent chunk.

    Holds the pinned CPU tensor containing the fp16 (or whatever dtype)
    parameter data, the original shape, dtype, and byte offset inside
    the chunk's flat byte buffer — everything :meth:`ChunkManager.gather`
    needs to rebind ``param.data`` to a GPU view after the H2D copy.

    In the ZeRO-3 sharded path (``zero3_shard=True``) each param's
    ``cpu_data`` / ``cpu_grad`` may be ``None`` when the param lies
    outside this rank's shard range — the bytes live on a peer rank
    and will be reconstructed on ``gather`` via ``all_gather``. The
    ``byte_offset`` / ``numel`` / ``element_size`` fields are
    authoritative regardless; they describe the full-chunk layout
    shared by every rank.
    """

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
    """One contiguous same-dtype byte region inside a sharded chunk.

    A chunk with homogeneous dtype maps to a single region spanning the
    whole chunk. A chunk with mixed dtypes (e.g. fp16 attention +
    fp32 RMSNorm scales) maps to ONE REGION PER maximal-length
    contiguous run of same-dtype params — a standard Llama fp16 block
    with fp32 layernorms produces ~3 regions per block.

    Each region is partitioned across ranks independently: rank ``r``
    owns the byte range ``[r * shard_bytes, (r + 1) * shard_bytes)``
    within the region, where ``shard_bytes = region_bytes_padded /
    world_size`` and ``region_bytes_padded`` is rounded up to
    ``lcm(element_size, world_size)`` so shard slices land on clean
    element boundaries. The collective (``all_gather_into_tensor`` on
    gather, ``reduce_scatter_tensor`` on reduce) is issued ONCE PER
    REGION — correctness-first; a mixed-dtype chunk with 3 regions
    issues 3 collectives per gather/reduce. This trades peak throughput
    for correctness: the alternative (one collective coalescing regions
    across dtypes) would need careful pack/unpack buffers at each rank
    and was judged out-of-scope for the M7 follow-up.

    Fields
    ------
    chunk_offset:
        Byte offset of this region inside the chunk's padded byte
        layout. All params in the region have ``byte_offset ∈
        [chunk_offset, chunk_offset + region_bytes)``.
    region_bytes:
        Size of the region (before world_size padding). May be padded
        per-param for element alignment but excludes any inter-region
        alignment padding ``materialize_offload`` adds at the region's
        tail.
    region_bytes_padded:
        ``region_bytes`` rounded up to ``lcm(element_size, world_size)``.
        Equals ``shard_bytes * world_size``.
    shard_bytes:
        Bytes this rank owns within the region: ``region_bytes_padded
        / world_size``.
    dtype / element_size:
        The common dtype of every param in the region and its
        ``dtype.itemsize``.
    cpu_shard_bytes / cpu_shard_grad_bytes:
        Pinned ``uint8`` tensors holding THIS RANK's slice of the
        region's data / grad. Both are ``shard_bytes`` long.
        ``cpu_shard_grad_bytes`` is ``None`` for fully-frozen regions —
        we never reduce/copy grads into them, so allocating the buffer
        would only waste CPU memory.
    shard_param:
        An ``nn.Parameter`` whose ``.data`` views ``cpu_shard_bytes``
        as ``dtype``. The CPU FusedAdam adapter is built against this
        param — one flat Adam step per region. Constructed with
        ``requires_grad`` matching the region's trainable state, and
        with ``.grad`` left ``None`` for fully-frozen regions so the
        optimizer's standard ``param.grad is None`` skip-clause keeps
        weight decay / moment updates from touching frozen bytes
        (PEFT/LoRA + base-weight freezing correctness).
    is_trainable:
        ``True`` iff at least one param contributing bytes to this
        region has ``requires_grad=True``. Region segmentation in
        :meth:`ChunkManager.materialize_offload` splits on this
        boundary in addition to dtype, so every region is uniformly
        trainable or uniformly frozen.
    """

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
    """Per-chunk ZeRO-3 shard bookkeeping (populated when ``zero3_shard=True``).

    A chunk is modelled as an ordered list of :class:`_DtypeRegion`
    entries, each describing one maximal-length contiguous same-dtype
    byte span within the chunk. For homogeneous-dtype chunks this
    reduces to a single region covering the whole chunk; for
    mixed-dtype chunks we get one region per contiguous same-dtype
    run. Each region is independently partitioned across ranks and
    participates in its own ``all_gather_into_tensor`` /
    ``reduce_scatter_tensor`` collective during gather/reduce.

    ``chunk_bytes`` is the total byte footprint of the chunk including
    any inter-region alignment padding (equal to the sum of the
    regions' ``region_bytes_padded`` plus any leading/trailing pad).

    ``shard_bytes`` is the SUM of per-region ``shard_bytes`` — the
    total number of CPU-pinned bytes THIS RANK holds for the chunk.
    Exposed primarily for tests and for the CPU-footprint assertion in
    ``test_multi_gpu_7b.py::test_protrain_4gpu_zero3_sharding``.
    """

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
        """Whether this chunk is genuinely in the sharded path.

        True whenever at least one region exists. Useful for test
        assertions that the sharded path engaged (vs. silently
        falling back to replicated mode, which would leave
        ``_chunk_shards`` empty for the chunk).
        """
        return bool(self.regions)


class BackwardHandle:
    """RAII refcount handle for a chunk pinned across a backward window.

    Returned by :meth:`ChunkManager.gather_for_backward` and consumed
    by :class:`OffloadedBlock` (M2 of the Option B rollout). Each
    handle represents one outstanding reference to ``chunk_id``'s
    GPU pool slot. When the last live handle for a chunk is dropped:

    1. The manager's per-chunk refcount hits zero.
    2. If ``reduce_grads_and_offload`` ran while the count was non-zero
       (the slot couldn't be safely released because saved tensors
       still aliased it), the deferred offload runs now.

    Lifetime is driven by the autograd engine: ``OffloadedBlock._unpack``
    attaches the handle to the unpack-returned view as a private
    attribute, autograd holds the view until the consuming Node's
    ``apply()`` completes, then drops it; Python ref-counting cascades
    the drop to the handle's ``__del__``.

    ``release()`` is the explicit-drop path (idempotent). Tests use
    it to deterministically simulate handle drops without relying on
    GC timing. ``__del__`` is the safety net for the GC-driven path.
    """

    __slots__ = ("_chunk_id", "_manager", "_released")

    def __init__(self, chunk_id: ChunkId, manager: "ChunkManager") -> None:
        """Bind the handle to ``chunk_id`` on ``manager``; refcount already bumped."""
        self._chunk_id = chunk_id
        self._manager = manager
        self._released = False

    def release(self) -> None:
        """Drop this handle, decrementing the manager's refcount.

        Idempotent. Safe to call multiple times; subsequent calls are
        no-ops. The manager handles the deferred-offload drain when
        the count hits zero.
        """
        if self._released:
            return
        self._released = True
        # ``_release_backward_handle`` performs the decrement +
        # deferred-drain. We hold a hard reference to the manager so
        # __del__ during interpreter shutdown still works (the
        # manager is reachable via this handle's __slots__ until we
        # explicitly clear it below).
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
        # __del__ must not raise. ``release`` is itself defensively
        # try/except'd; the only remaining risk is the manager weakref
        # being collected before us during interpreter shutdown.
        try:
            self.release()
        except Exception:  # noqa: BLE001 — destructors must not throw
            pass


class ChunkManager:
    """Runtime driver for a :class:`ChunkLayout`.

    Parameters
    ----------
    model
        The already-initialized ``nn.Module`` whose ``named_parameters()``
        cover every ``ParamId`` in ``layout``.
    layout
        Output of :func:`axolotl.integrations.protrain.chunk.layout.build_layout`.
    n_persist
        Number of leading chunks kept resident on GPU. The rest are
        offloaded / sharded.
    buffer_pool
        Pre-allocated GPU chunk buffers for the non-persistent path.
        May be ``None`` in the all-persistent layout (every chunk
        resident on GPU, ``n_persist == layout.N_chunk``); in that
        case no method that needs the pool ever fires (gather/offload
        early-return for persistent chunks, ``_ensure_persistent_buffer``
        sources its device from ``self.device``).
    cpu_optim
        Optional CPU FusedAdam adapter for non-persistent chunks. If
        provided, :meth:`reduce_grads_and_offload` triggers its
        ``step_async`` the moment grads land on CPU.
    gpu_optim
        Optional GPU FusedAdam adapter for the persistent chunk set;
        invoked by :meth:`persistent_step`.
    device
        The CUDA device where non-persistent chunks land when gathered.
        Defaults to ``buffer_pool.device`` when a pool is provided;
        otherwise must be supplied explicitly (the all-persistent
        wrapper passes the resolved device directly).
    world_size, rank
        Collective-comms context, defaulting to ``1`` / ``0`` for the
        single-rank unit-test path. When ``world_size > 1`` and
        ``zero3_shard=True``, non-persistent chunks are partitioned
        across ranks on CPU and ``gather``/``reduce_grads_and_offload``
        become ``all_gather_into_tensor`` / ``reduce_scatter_tensor``
        respectively (M7 true ZeRO-3 path).
    zero3_shard
        When True, activate the sharded non-persistent-chunk path
        described in the module docstring. When False (the default), the
        manager behaves identically to the M4.5 / M6 snapshot: every
        rank holds a full copy of each non-persistent chunk on CPU and
        cross-rank grad sync uses per-param ``all_reduce(op=AVG)``
        (ZeRO-2-ish, composes cleanly under an outer DDP wrapper).
    """

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
        # When the layout is all-persistent (n_persist == N_chunk) the
        # caller may legitimately pass ``buffer_pool=None`` to skip the
        # dormant pool allocation. In that case ``device`` MUST be
        # supplied explicitly — there's no pool to source it from.
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

        # ZeRO-3 sharding context. ``world_size`` and ``rank`` default
        # to the single-rank case; when either is > default AND
        # ``zero3_shard`` is True, :meth:`materialize_offload` creates
        # per-rank CPU shards and :meth:`gather` /
        # :meth:`reduce_grads_and_offload` take the collectives path.
        self.world_size: int = int(max(1, world_size))
        self.rank: int = int(max(0, rank))
        if self.rank >= self.world_size:
            raise ValueError(
                f"rank={self.rank} out of range for world_size={self.world_size}"
            )
        # Sharding is only physically active when BOTH the flag is set
        # and we have peers to talk to. With ``world_size == 1`` a
        # "sharded" chunk would be the full chunk (a rank of 1 talking
        # to itself) — degrading cleanly to the ZeRO-2-style replication
        # path keeps the unit tests for zero3_shard=True viable on
        # single-GPU hosts.
        self.zero3_shard: bool = bool(zero3_shard) and self.world_size > 1

        # When True, :meth:`reduce_grads_and_offload` and the per-param
        # grad-offload hook skip their internal ``dist.all_reduce`` calls
        # and trust an outer layer (typically ``DistributedDataParallel``
        # wrapped over the protrain'd module) to own cross-rank grad
        # sync. Toggled by ``protrain_model_wrapper`` at compose-time —
        # see the Multi-GPU section of ``DESIGN.md``. Mutually exclusive
        # with ``zero3_shard=True``: the sharded path is the grad-sync
        # point in its own right (reduce_scatter), so an outer DDP
        # wouldn't compose anyway.
        self.skip_internal_grad_reduce: bool = False

        # Param lookup by id for gather/offload payload construction.
        self._params_by_id: dict[ParamId, "nn.Parameter"] = {
            cast(ParamId, name): p for name, p in model.named_parameters()
        }

        # Persistent / non-persistent split; populated in ``mark_persistent``.
        self._persistent_ids: set[ChunkId] = set()
        self._non_persistent_ids: set[ChunkId] = set(
            cast(ChunkId, i) for i in range(layout.N_chunk)
        )

        # Per-chunk resident GPU flat tensor — populated only for persistent
        # chunks (non-persistent chunks borrow from the buffer pool).
        self._persistent_buffers: dict[ChunkId, "torch.Tensor"] = {}

        # Per-chunk CPU slots: materialize_offload populates this dict
        # mapping chunk_id -> list[_CpuParamSlot] ordered as the params
        # appear in ``layout.chunks[chunk_id]``.
        self._cpu_slots: dict[ChunkId, list[_CpuParamSlot]] = {}

        # App B.2 — custom pinned-memory allocator for the offload-time
        # host shadows. PyTorch's ``pin_memory=True`` flows through
        # ``CUDAHostAllocator`` which rounds up to the next power of two
        # (paper App B.2 explicitly calls this out as wasteful). We
        # instead allocate ONE precise-size :class:`PinnedHostMemory`
        # region for every non-persistent chunk's param bytes, and ONE
        # for every trainable param's grad shadow bytes — sized to the
        # exact aligned-byte total — and slice per-chunk views out of
        # them. Populated by :meth:`materialize_offload`; freed by
        # :meth:`restore_to_gpu` (or the manager's GC fallback).
        # ``None`` until materialize, or when the layout has no
        # non-persistent chunks at all (all-persistent shape — no host
        # shadows are needed).
        self._cpu_param_pool: "PinnedHostMemory | None" = None
        self._cpu_grad_pool: "PinnedHostMemory | None" = None

        # Per-chunk sharded state (ZeRO-3 path). Populated by
        # :meth:`materialize_offload` only when ``self.zero3_shard`` is
        # True and the chunk qualifies for sharding (homogeneous dtype).
        # Unset entries signal the chunk falls back to the replicated
        # path even in sharded mode.
        self._chunk_shards: dict[ChunkId, _ChunkShardState] = {}

        # Empty GPU sentinel (one per dtype) — reused for all param.data
        # "placeholders" after offload so we don't allocate a fresh 0-byte
        # tensor per param (cheap but not free).
        self._empty_by_dtype: dict["torch.dtype", "torch.Tensor"] = {}

        # Per-chunk grad-drain counter: decremented by _offload_grad for
        # every trainable param in the chunk; when it hits zero we kick
        # off the async CPU Adam step (Gap 2).
        self._grad_remaining: dict[ChunkId, int] = {}
        # How many trainable params a chunk started with, used to reset
        # _grad_remaining at the top of every backward pass (we clone this
        # dict on demand).
        self._grad_initial: dict[ChunkId, int] = {}

        # Hook handles stored so ``uninstall`` / ``__del__`` can remove
        # them deterministically and we don't leak closures over ``self``.
        self._grad_hook_handles: list[object] = []

        # M2 / Option B state: storage-pointer -> chunk_id reverse
        # lookup populated at gather time and cleared at offload time.
        # ``OffloadedBlock._pack`` queries this to detect "is this
        # saved tensor a view of a chunk-managed param?". Using
        # storage identity matches what autograd actually saved
        # (storages survive view ops); using param-id would force a
        # weakref-from-tensor path that PyTorch doesn't offer.
        self._storage_ptr_to_chunk: dict[int, ChunkId] = {}

        # Per-chunk refcount of outstanding ``BackwardHandle``s. When
        # non-zero, ``reduce_grads_and_offload(cid)`` defers the
        # actual offload into ``_deferred_offloads``; when the last
        # handle is dropped, the deferred offload drains. This is
        # the §3.4 backward-window pin counter.
        self._backward_refcount: dict[ChunkId, int] = {}

        # Set of chunks whose offload was deferred because their
        # backward refcount was non-zero at reduce time. Drained by
        # ``_release_backward_handle`` once the last handle drops.
        self._deferred_offloads: set[ChunkId] = set()

        self.mark_persistent(n_persist)

    # ---- configuration -------------------------------------------------

    def mark_persistent(self, first_n: int) -> None:
        """Tag chunks ``[0, first_n) ∪ layout.mandatory_persistent`` as persistent.

        ``first_n`` is the user-chosen prefix length (the search's
        ``cfg.n_persist``). The runtime resident set augments the prefix
        with ``layout.mandatory_persistent`` — chunks the block-
        granularity scheduler cannot gather on its own (typically chunks
        containing non-block params like ``model.norm.weight`` or an
        untied ``lm_head``); see :class:`ChunkLayout` for the
        runtime-correctness rationale.

        Idempotent — safe to call after a searcher re-pick at the start
        of a new epoch. Allocations for already-materialized buffers are
        NOT changed here (the first-time materialization happens lazily
        in :meth:`gather` / :meth:`_ensure_persistent_buffer`), so
        repeated calls with the same ``first_n`` are cheap.
        """
        if first_n < 0 or first_n > self.layout.N_chunk:
            raise ValueError(
                f"first_n={first_n} out of range [0, {self.layout.N_chunk}]"
            )
        # Use ChunkLayout.effective_persistent_ids so the search /
        # runtime / cost-model definitions of "persistent" are
        # single-sourced.
        new_persistent_ids = set(self.layout.effective_persistent_ids(first_n))
        new_non_persistent_ids = {
            cast(ChunkId, i)
            for i in range(self.layout.N_chunk)
            if cast(ChunkId, i) not in new_persistent_ids
        }
        # CodeRabbit R2-04 fix: once chunks have been materialized into
        # CPU placeholder slots or persistent GPU buffers, the residency
        # split is baked into the runtime state — a previously offloaded
        # chunk newly tagged persistent would early-return in ``gather``
        # while its params still point at empty GPU placeholders, and a
        # previously persistent chunk newly tagged non-persistent would
        # have no ``_cpu_slots`` to drain grads into. Reject the change
        # so the failure surfaces immediately rather than as silent
        # weight corruption many steps later.
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
        """Physically move non-persistent chunks' params to pinned CPU memory.

        Two-phase implementation (paper App B.2):

        1. **Planning pass.** Walk every non-persistent chunk and compute
           its byte plan — per-param aligned byte offsets (BUG 2 fix
           preserved per-chunk), the local ``chunk_bytes`` total, and
           (under ``zero3_shard``) the per-region partition layout.
           Compute the SUM of pinned-host bytes needed across every
           chunk for params and (separately) for trainable grads.
        2. **Allocation pass.** Allocate ONE
           :class:`PinnedHostMemory` region for the param pool and ONE
           for the grad pool — sized to the precise sum of per-chunk
           aligned bytes. Then walk the plans again, slice per-chunk
           views out of the unified pools, and populate
           ``_cpu_slots`` / ``_chunk_shards`` / per-param grad shadows
           on top of those views.

        App B.2 paper-fidelity: PyTorch's ``torch.empty(pin_memory=True)``
        routes through ``CUDAHostAllocator`` which rounds up to the next
        power of two. ``PinnedHostMemory`` calls ``cudaHostAlloc``
        directly via ctypes for an exact byte count, avoiding the
        rounding waste. The two-pass structure is what lets us pre-size
        the unified region precisely.

        Inter-chunk alignment: each chunk's start offset within the
        unified pool is padded up to a 16-byte boundary. 16 covers
        every dtype up through fp64 (8-byte itemsize doubled for
        future-proofing); the per-param BUG 2 alignment computed
        WITHIN a chunk continues to use the chunk's max element size,
        so the alignment guarantee carried by ``slot.byte_offset`` is
        unchanged from the previous per-chunk-allocation layout. The
        16-byte inter-chunk pad costs at most 15 bytes per chunk —
        negligible vs. the per-power-of-2 round-up the App B.2
        switch eliminates.

        Per-chunk action (Pass 2):

        * For each param: copy ``param.data`` (GPU) into its CPU slot
          (a view into the unified param pool), then replace
          ``param.data`` with an empty GPU placeholder.
        * For each *trainable* param (replicated mode) OR each
          trainable region (sharded mode): assign a view into the
          unified grad pool as the grad shadow, and register a
          ``register_post_accumulate_grad_hook`` that drains the
          grad to CPU on the fly (Gap 2).

        Returns
        -------
        int
            Bytes freed on the GPU by the offload. Sum of
            ``param.numel() * param.element_size()`` across every
            offloaded param.

        Idempotent: a second call is a no-op (detected via
        ``self._cpu_param_pool`` already being non-None).
        """
        if self._cpu_param_pool is not None or self._cpu_slots:
            LOG.debug(
                "ChunkManager.materialize_offload: already materialized "
                "(%d chunks), no-op",
                len(self._cpu_slots),
            )
            return 0

        import torch

        from axolotl.integrations.protrain.chunk.pinned_alloc import PinnedHostMemory

        # Inter-chunk alignment for slicing per-chunk byte ranges out of
        # the unified pool. 16 bytes covers any dtype up to fp64 (8
        # bytes) with a doubling for future-proofing — enough that the
        # per-chunk byte-view ``narrow(0, chunk_start, chunk_bytes)``
        # always lands on an element boundary regardless of which
        # dtype mix the chunk holds.
        _INTER_CHUNK_ALIGN = 16

        # ---- Pass 1: planning ------------------------------------------
        # For each non-persistent chunk, compute everything we need to
        # know to slice the unified pinned pools later: per-param
        # aligned offsets, chunk_bytes, region partition (if sharded),
        # trainable flags. No allocations yet.
        #
        # Each plan is a dict shaped like::
        #
        #   {
        #     "cid": ChunkId,
        #     "param_ids": list[ParamId],
        #     "per_param_bytes": list[int],
        #     "aligned_offsets": list[int],
        #     "chunk_bytes": int,
        #     "shardable": bool,
        #     "region_plans": list[dict],  # populated iff shardable
        #     "total_shard_bytes": int,    # sum across regions
        #     "param_pool_offset": int,    # filled in Pass 2
        #     "grad_pool_offset": int,     # filled in Pass 2 (replicated path)
        #     "region_grad_offsets": list[int],  # filled in Pass 2 (sharded path)
        #   }
        chunk_plans: list[dict] = []
        # Running totals for the unified pools.
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

            # Per-param byte sizes + element sizes (for BUG 2 alignment).
            # BUG 2 FIX: each param's byte_offset must be aligned to its
            # element_size, otherwise ``byte_view.view(dtype)`` raises
            # ``RuntimeError: offset is not aligned``. This bites when a
            # chunk contains a mix of 2-byte (fp16/bf16) and 4-byte
            # (fp32) params — e.g. Llama's fp16 attention weights sitting
            # next to fp32 RMSNorm scales — because the running offset
            # lands on an odd multiple of 2 when an fp16 param precedes
            # an fp32 one. We pad each param's starting offset up to a
            # multiple of its element_size before laying it down; this
            # guarantees alignment for any dtype mix up to 8 bytes
            # (fp64). The padding bytes stay zero; no code ever reads a
            # padding region — the only readers are the per-param typed
            # views and the per-param H2D copy which only touches
            # ``nbytes``.
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

            # Per-param aligned offsets within the chunk (BUG 2 fix).
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

            # Decide shardability + compute dtype regions.
            # When ``zero3_shard`` is on we always try to shard — even
            # mixed-dtype chunks. The chunk is modelled as an ordered
            # list of maximal-length contiguous same-dtype regions;
            # each region is sharded independently (its own
            # ``all_gather`` / ``reduce_scatter`` collective). For a
            # homogeneous chunk this reduces to a single region
            # spanning the whole chunk and behaves identically to the
            # pre-M7-followup path.
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
                    # CodeRabbit R07 fix: split regions on requires_grad
                    # in addition to dtype so each region is uniformly
                    # trainable or uniformly frozen.
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
                    # Pad in element space so each rank owns a whole
                    # number of elements. The previous formula padded in
                    # bytes via ``lcm(esize, world_size)``, which split
                    # mid-element when ``gcd(esize, world_size) > 1``
                    # (e.g. esize=4, world_size=8 → pad_unit=8 bytes,
                    # 1 byte/rank = ½ element). Must stay byte-compatible
                    # with ``api/reshard.py::_padded_region_bytes``: the
                    # loader's regions_per_chunk match recomputes this and
                    # any drift trips the layout-signature check.
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

            # ---- Pool-byte accounting --------------------------------
            # Per-chunk param-pool footprint:
            #   - Replicated path: ``chunk_bytes`` (full-chunk pinned).
            #   - Sharded path:    ``total_shard_bytes`` (sum of per-rank
            #     region shards). The full-chunk buffer is transient,
            #     allocated unpinned at materialize time only to perform
            #     the initial partition.
            # Per-chunk grad-pool footprint:
            #   - Replicated path: sum of ``nbytes`` across trainable
            #     params (each gets its own shape-typed grad shadow).
            #   - Sharded path:    sum of ``shard_bytes`` across
            #     trainable regions.
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

            # Reserve aligned starting offsets in the unified pools.
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

        # ---- Pool allocation ------------------------------------------
        # ONE precise-size :class:`PinnedHostMemory` per kind. The
        # allocator calls ``cudaHostAlloc`` directly with the exact
        # byte count — no power-of-2 round-up. ``n_buffer=1`` plus
        # ``S_chunk=total_bytes`` gives a single big slot we slice
        # per-chunk via ``narrow`` in the population pass.
        #
        # Empty-pool guard: if no non-persistent chunk needs param /
        # grad bytes, skip the allocation. ``PinnedHostMemory`` rejects
        # zero-byte slot sizes; the manager simply leaves the pool
        # ``None`` and every downstream consumer sees an empty slot
        # list / region list.
        param_pool_buf: "torch.Tensor | None" = None
        grad_pool_buf: "torch.Tensor | None" = None
        if total_param_pool_bytes > 0:
            self._cpu_param_pool = PinnedHostMemory(
                n_buffer=1, S_chunk=total_param_pool_bytes
            )
            # Borrow slot 0 for the lifetime of this manager. Released
            # via ``release_buffer(0)`` in :meth:`restore_to_gpu`'s
            # teardown (or the manager's GC fallback) — see
            # :meth:`_close_cpu_pools`.
            param_pool_buf = self._cpu_param_pool.buffer(0)
        if total_grad_pool_bytes > 0:
            self._cpu_grad_pool = PinnedHostMemory(
                n_buffer=1, S_chunk=total_grad_pool_bytes
            )
            grad_pool_buf = self._cpu_grad_pool.buffer(0)

        # ---- Pass 2: population ---------------------------------------
        # Walk the plans, slice per-chunk views out of the unified pools,
        # populate slots / regions / grad shadows.
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

            # Per-chunk view into the param pool.
            #   - Replicated: full-chunk byte tensor backed by pinned
            #     memory; slot.cpu_data slices are views into it.
            #   - Sharded:    per-rank shard region; per-region
            #     ``cpu_shard_bytes`` are views into it. The full-chunk
            #     image used for the initial partition is allocated
            #     unpinned (transient) below.
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

            # ---- Replicated path ------------------------------------
            # ``chunk_param_view`` IS the full-chunk pinned bytes; per-
            # param slot.cpu_data = chunk_param_view.narrow(byte_offset,
            # nbytes).view(dtype).view(shape). Per-param grad shadow =
            # chunk_grad_view.narrow(running_grad_offset, nbytes)
            # .view(dtype).view(shape).
            #
            # ---- Sharded path ---------------------------------------
            # ``chunk_param_view`` IS this rank's contiguous shard
            # region (sum of per-region shard_bytes). Within it we
            # carve per-region sub-ranges. The transient
            # ``transient_chunk_bytes`` buffer below holds the full
            # chunk image just long enough to perform the per-region
            # partition (region_scratch → cpu_region_shard.copy_).
            transient_full_chunk: "torch.Tensor | None" = None
            if chunk_is_shardable:
                # Unpinned scratch: only used for the partition copy.
                # Pinning would just waste pinned host memory because
                # the buffer is released at the end of this iteration.
                transient_full_chunk = torch.empty(chunk_bytes, dtype=torch.uint8)

            # ---- Per-param copy + rebind ----------------------------
            slots: list[_CpuParamSlot] = []
            trainable_count = 0
            # Per-trainable-param running offset into the chunk's grad
            # view (replicated path only). Grad shadows are packed
            # contiguously in the order they appear in the chunk.
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

                # The per-param byte view is taken from EITHER the
                # transient full-chunk buffer (sharded — partitioned
                # away after this loop) OR the chunk's slice of the
                # unified param pool (replicated — permanent storage).
                if chunk_is_shardable:
                    assert transient_full_chunk is not None
                    cpu_view = transient_full_chunk.narrow(0, off, nbytes)
                else:
                    assert chunk_param_view is not None
                    cpu_view = chunk_param_view.narrow(0, off, nbytes)
                cpu_param = cpu_view.view(dtype).view(shape)
                cpu_param.copy_(orig_data)

                # Release GPU storage by rebinding .data to an empty
                # placeholder of the same dtype.
                param.data = self._empty_placeholder(dtype)

                # Pinned CPU grad shadow for trainable params (replicated
                # only). In sharded mode the per-region shard buffer
                # covers every trainable param's grad bytes, so the
                # per-slot shadow stays ``None`` (and the per-param
                # hook short-circuits to the counter-only path).
                cpu_grad: "torch.Tensor | None" = None
                if param.requires_grad:
                    trainable_count += 1
                    if not chunk_is_shardable:
                        assert chunk_grad_view is not None
                        # Slice the chunk's grad view at the running
                        # offset, reshape to (dtype, shape). The view
                        # shares storage with the unified grad pool;
                        # writes through ``cpu_grad.copy_(...)`` land
                        # directly in pinned memory.
                        grad_byte_view = chunk_grad_view.narrow(
                            0, grad_running_off, nbytes
                        )
                        cpu_grad = grad_byte_view.view(dtype).view(shape)
                        # Zero the freshly-allocated grad shadow — it
                        # may be read by upstream consumers (reference
                        # comparisons in tests, the first
                        # accumulate-grad sequence) before any backward
                        # has fired. Pre-zero to match the
                        # ``torch.zeros`` semantics of the prior
                        # per-param allocation.
                        cpu_grad.zero_()
                        grad_running_off += nbytes

                # For sharded chunks ``slot.cpu_data`` is None — the
                # bytes live in per-region shards across
                # ``self._chunk_shards`` and the chunk_param_view
                # holds those shards, not the full chunk.
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

            # ---- Sharded path: per-region shard partition + bookkeeping
            if chunk_is_shardable:
                assert transient_full_chunk is not None
                assert chunk_param_view is not None  # holds per-region shards

                regions: list[_DtypeRegion] = []
                # Per-region running offsets within the chunk's slice
                # of the unified pools. Regions are packed contiguously
                # in the order they were planned.
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

                    # Build the padded region image in a transient
                    # scratch buffer: copy the valid region_bytes from
                    # the full-chunk transient into [0, region_bytes),
                    # the trailing pad stays zero. This keeps peer
                    # ranks that receive the padded tail from seeing
                    # uninitialized bytes on the first ``gather``.
                    region_scratch = torch.zeros(r_bytes_padded, dtype=torch.uint8)
                    region_scratch.narrow(0, 0, r_bytes).copy_(
                        transient_full_chunk.narrow(0, r_chunk_off, r_bytes)
                    )

                    # This rank's shard of the region — VIEW into the
                    # unified param pool, NOT a fresh allocation.
                    my_off = self.rank * r_shard_bytes
                    cpu_region_shard = chunk_param_view.narrow(
                        0, region_param_off, r_shard_bytes
                    )
                    cpu_region_shard.copy_(
                        region_scratch.narrow(0, my_off, r_shard_bytes)
                    )
                    region_param_off += r_shard_bytes

                    # CodeRabbit R07 fix: only allocate the pinned grad
                    # shard for trainable regions. Frozen-only regions
                    # never receive a reduce/copy in
                    # :meth:`reduce_grads_and_offload`; binding a
                    # zero-grad view as ``shard_param.grad`` would
                    # let Adam's weight-decay rewrite frozen bytes.
                    cpu_region_grad: "torch.Tensor | None" = None
                    if r_is_trainable:
                        assert chunk_grad_view is not None
                        cpu_region_grad = chunk_grad_view.narrow(
                            0, region_grad_off, r_shard_bytes
                        )
                        cpu_region_grad.zero_()
                        region_grad_off += r_shard_bytes

                    # Shard-level nn.Parameter for this region — one
                    # flat Adam step per region.
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
                # ``transient_full_chunk`` falls out of scope here;
                # Python GC reclaims its (unpinned) bytes.

            # ---- Step 4: per-param grad hooks for trainable params ----
            # In sharded mode the hook still fires per-param — we need
            # the counter decrement so :meth:`reduce_grads_and_offload`
            # can tell when every param in the chunk has an accumulated
            # grad. The hook body takes a different fast-path for
            # sharded slots (see :meth:`_make_grad_offload_hook`).
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
        return freed

    def _close_cpu_pools(self) -> None:
        """Release the unified pinned-host param/grad pools.

        Idempotent. Drops every PinnedHostMemory borrow held on
        ``buffer(0)`` and calls ``close()`` on each pool. Called from
        :meth:`restore_to_gpu` (deterministic teardown) and from the
        manager's ``__del__`` (GC safety net via PinnedHostMemory's own
        ``__del__``).

        Lifetime hazard: every per-slot ``cpu_data`` / ``cpu_grad``
        and per-region ``cpu_shard_bytes`` / ``cpu_shard_grad_bytes``
        is a view into the unified pool. Callers MUST drop those views
        (clear ``_cpu_slots`` / ``_chunk_shards``) BEFORE invoking
        this method — otherwise the view tensors become dangling
        pointers into freed pinned memory. ``restore_to_gpu`` does
        this by calling ``_cpu_slots.clear()`` etc. AFTER copying the
        bytes back to GPU.
        """
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

    def restore_to_gpu(self) -> int:
        """Inverse of :meth:`materialize_offload` — move every param back to GPU.

        For each non-persistent chunk in ``self._cpu_slots``: allocate a
        fresh standalone GPU tensor of each param's recorded shape +
        dtype, copy from the pinned CPU slot, and rebind ``param.data``
        to the new tensor. For each persistent chunk that has a
        materialized resident buffer: copy each param's typed view out
        of the pool buffer into a standalone GPU tensor and rebind.

        After the pass every parameter once again owns its own GPU
        storage — exactly as it did before ``materialize_offload`` ran —
        so a fresh :class:`ChunkManager` constructed against the same
        model can re-run ``materialize_offload`` from scratch under a
        new ``CostConfig`` (different ``n_persist`` / ``n_buffer`` /
        ``S_chunk``). This is the foundation for the phase-2 profiler's
        bootstrap-then-rebuild flow (paper §3.2 calibration loop).

        Sharded path (``zero3_shard=True``)
        -----------------------------------
        For sharded chunks ``slot.cpu_data is None`` — the bytes live
        in per-rank slices across ``self._chunk_shards``. Each chunk
        is reassembled by issuing one
        :func:`torch.distributed.all_gather_into_tensor` per
        :class:`_DtypeRegion`: this rank's pinned CPU shard is
        H2D-staged into a GPU buffer (mirroring the materialize-time
        partition step), every rank's contribution is gathered into a
        ``region_bytes_padded``-sized GPU scratch, and the valid
        ``region_bytes`` prefix is copied into the chunk's reassembly
        buffer at the region's recorded ``chunk_offset``. Once every
        region is in place the chunk-sized buffer holds the same byte
        layout the replicated path would have produced; per-slot
        rebind then proceeds exactly as in the non-sharded branch.

        The collective is a no-op when ``world_size == 1`` (every shard
        IS the full region) but ``materialize_offload`` does not engage
        the sharded path under ``world_size == 1`` to begin with — see
        ``__init__``'s ``self.zero3_shard = ... and self.world_size > 1``
        guard — so this method only runs the all_gather when there are
        actually peer ranks to talk to.

        Returns
        -------
        int
            Bytes copied back to standalone GPU storage. 0 on a manager
            that was never offloaded.

        Raises
        ------
        RuntimeError
            When ``zero3_shard`` is True but ``torch.distributed`` is
            not initialized. The sharded path requires a live process
            group to issue the per-region ``all_gather_into_tensor``;
            calling restore on a manager whose distributed context has
            already been torn down is a programmer error.

        Idempotent: a second call with no offload materialized is a no-op.
        """
        # Wait for any in-flight async CPU Adam steps to finish so we
        # snapshot a consistent post-step state, not a half-applied one.
        # Without this barrier, a CpuFusedAdamAdapter.step_async() worker
        # could be mid-write to the same shard tensors restore_to_gpu
        # reads, producing corrupted weights — or restore could clear
        # shard state out from under the still-running worker.
        # ``wait_cpu_optim`` is a no-op when ``self.cpu_optim is None``
        # (no DeepSpeedCPUAdam — replicated path or unavailable).
        self.wait_cpu_optim()

        if not self._cpu_slots and not self._persistent_buffers:
            LOG.debug(
                "ChunkManager.restore_to_gpu: nothing offloaded "
                "(no _cpu_slots, no _persistent_buffers), no-op"
            )
            return 0

        import torch

        # Pre-flight: sharded restore needs a live process group for
        # the per-region all_gather. Catch the misuse here with a clean
        # error rather than letting torch.distributed raise an opaque
        # "default process group not initialized" deep in the call stack.
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

        # App B.2: route every GPU allocation made by this teardown path
        # through the default-stream heap. ``restore_to_gpu`` runs at
        # manager teardown and is never invoked from inside a
        # non-default stream context today, but the wraps cost almost
        # nothing and keep the heap-routing invariant uniform across
        # the chunk manager. No ``record_stream`` calls needed: every
        # allocation here is consumed only by default-stream copies and
        # the freshly-allocated tensors are immediately written to via
        # ``copy_`` on the same stream they were allocated on.
        from axolotl.integrations.protrain.runtime.streams import (
            SingleStreamAllocator,
        )

        # Helper closure to keep the per-site wrap a one-liner.
        _on_cuda = self.device.type == "cuda" and torch.cuda.is_available()

        def _alloc_empty(shape, dtype):
            if _on_cuda:
                with SingleStreamAllocator():
                    return torch.empty(shape, dtype=dtype, device=self.device)
            return torch.empty(shape, dtype=dtype, device=self.device)

        # ---- Non-persistent chunks: copy from pinned CPU slots --------
        # For sharded chunks ``slot.cpu_data is None`` — those are
        # handled by the sharded reassembly block below. For replicated
        # (non-sharded) chunks, slot.cpu_data is the full-shape pinned
        # tensor and the per-slot copy is the inverse of materialize.
        for cid, slots in self._cpu_slots.items():
            if cid in self._chunk_shards:
                # Defer to the sharded reassembly pass below.
                continue
            for slot in slots:
                param = self._params_by_id.get(slot.param_id)
                if param is None or slot.cpu_data is None:
                    continue
                gpu_tensor = _alloc_empty(slot.shape, slot.dtype)
                gpu_tensor.copy_(slot.cpu_data)
                param.data = gpu_tensor
                moved += slot.numel * slot.element_size

        # ---- Sharded chunks: per-region all_gather, then per-slot rebind
        # Reverses ``materialize_offload``'s shard-time partition (lines
        # ~753-836). For each region we reconstruct the full
        # ``region_bytes_padded`` byte image on GPU via
        # ``all_gather_into_tensor``, then copy the valid
        # ``[0, region_bytes)`` prefix into a chunk-sized GPU scratch at
        # the region's ``chunk_offset``. After every region for the
        # chunk is in place, walk the chunk's slots and rebind each
        # param.data to a fresh standalone GPU tensor sliced from the
        # scratch at ``slot.byte_offset``. This is the exact inverse of
        # the materialize-time
        #   "full chunk_bytes -> per-region scratch -> per-rank shard"
        # data flow.
        if self.zero3_shard and self._chunk_shards:
            import torch.distributed as dist

            for cid, shard_state in self._chunk_shards.items():
                # Chunk-sized GPU scratch holding the reassembled bytes.
                # Must use the manager's device so the per-slot rebind
                # below produces tensors on the same device as the
                # rest of the model.
                chunk_buf = _alloc_empty(shard_state.chunk_bytes, torch.uint8)

                for region in shard_state.regions:
                    # Stage this rank's CPU shard onto GPU. Mirrors the
                    # gather-time copy in ``_gather_sharded`` but drives
                    # the all_gather directly into a freshly allocated
                    # transient (we do NOT consult the buffer pool here
                    # — restore is a one-shot teardown and the pool may
                    # already be torn down by the caller).
                    my_shard_gpu = _alloc_empty(region.shard_bytes, torch.uint8)
                    my_shard_gpu.copy_(region.cpu_shard_bytes, non_blocking=True)

                    # Padded gather output: region_bytes_padded ==
                    # shard_bytes * world_size, so this matches the
                    # all_gather_into_tensor contract exactly (output
                    # length == input length * world_size).
                    gather_scratch = _alloc_empty(
                        region.region_bytes_padded, torch.uint8
                    )
                    dist.all_gather_into_tensor(gather_scratch, my_shard_gpu)

                    # Copy only the VALID prefix into the chunk
                    # reassembly buffer at the region's chunk offset.
                    # The trailing pad bytes (region_bytes_padded -
                    # region_bytes) are never read by any slot's
                    # byte_offset slice, so leaving them
                    # uninitialized in chunk_buf is correct.
                    chunk_buf.narrow(0, region.chunk_offset, region.region_bytes).copy_(
                        gather_scratch.narrow(0, 0, region.region_bytes)
                    )

                # All regions are in place: rebind each slot to a
                # fresh standalone GPU tensor. Per-slot fresh
                # allocation matches the non-sharded branch's
                # invariant — every param owns its own storage after
                # restore so the next ChunkManager can rebuild from
                # scratch under a new layout. We could keep params
                # pointing into ``chunk_buf`` to save bytes, but a
                # subsequent materialize_offload would then see params
                # whose .data aliases each other and corrupt its
                # alignment-padding pass.
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

        # ---- Persistent chunks: extract from the resident pool buffer
        # back into standalone GPU storage. The pool buffer itself can
        # then be released by clearing _persistent_buffers — params are
        # no longer pointing into it.
        for cid, buf in self._persistent_buffers.items():
            # We need the per-param byte offsets used at gather time.
            # _cpu_slots is the canonical record but persistent chunks
            # were never offloaded so it has no entry for them. Recompute
            # the same aligned-offset layout that materialize_offload
            # would have used (offsets are a function of the chunk's
            # param sequence + dtypes, not the offload itself).
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
                # Same alignment rule as materialize_offload (line ~550).
                offset = ((offset + esz - 1) // esz) * esz
                byte_view = buf.narrow(0, offset, nbytes)
                typed = byte_view.view(param.data.dtype).view(param.shape)
                gpu_tensor = _alloc_empty(param.shape, param.data.dtype)
                gpu_tensor.copy_(typed)
                param.data = gpu_tensor
                moved += nbytes
                offset += nbytes

        # ---- Drop hook handles + per-chunk state ----------------------
        # uninstall() removes the post-accumulate-grad hooks installed
        # by materialize_offload. After this the per-param hook bindings
        # are gone; a subsequent materialize_offload on a fresh manager
        # will install a new set.
        self.uninstall()

        # Clear every dict that materialize_offload populated so the
        # next ChunkManager doesn't see stale entries (shouldn't happen
        # — restore_to_gpu is meant to precede this manager's GC — but
        # be defensive). Order matters: drop view-holding state
        # (_cpu_slots, _chunk_shards) BEFORE attempting to close the
        # unified pinned pools, otherwise per-slot / per-region views
        # would be left as dangling pointers into freed pinned memory.
        self._cpu_slots.clear()
        self._chunk_shards.clear()
        self._persistent_buffers.clear()
        self._grad_initial.clear()
        self._grad_remaining.clear()
        # Empty placeholders are still referenced by params we just
        # rebound — the rebind dropped the param.data reference, so the
        # placeholders are unreferenced from torch's perspective. Drop
        # the dict so the next gather builds fresh ones if needed.
        self._empty_by_dtype.clear()

        # Release + close the unified pinned pools.
        #
        # Lifetime contract: ``_cpu_slots`` and ``_chunk_shards`` were
        # cleared above, so the manager no longer holds any narrow-view
        # into the pools. The borrow we took at materialize time
        # (``buffer(0)``) is released here, after which
        # ``PinnedHostMemory.close()`` calls ``cudaFreeHost`` and the
        # pinned region is reclaimed.
        #
        # Caveat: ``_DtypeRegion.shard_param`` tensors may still be
        # held externally (e.g. by a CPU FusedAdam adapter constructed
        # before restore). Since those views share storage with the
        # pinned region, freeing it here would create a
        # use-after-free at the optimizer's next read. Callers MUST
        # tear down the optimizer (or any other consumer of the
        # shard_params / cpu_data / cpu_grad views) BEFORE calling
        # ``restore_to_gpu`` in the rebuild flow. Documented as a
        # precondition of the bootstrap-then-rebuild loop in
        # :meth:`materialize_offload`'s docstring.
        self._close_cpu_pools()

        LOG.info(
            "ChunkManager.restore_to_gpu: moved %.3f GB back to standalone "
            "GPU storage (non-persistent + persistent combined)",
            moved / 1e9,
        )
        return moved

    def _empty_placeholder(self, dtype: "torch.dtype") -> "torch.Tensor":
        """Return a zero-element GPU tensor of ``dtype`` (cached per dtype)."""
        import torch

        from axolotl.integrations.protrain.runtime.streams import (
            SingleStreamAllocator,
        )

        existing = self._empty_by_dtype.get(dtype)
        if existing is not None:
            return existing
        # App B.2: cached one-per-dtype zero-element placeholder; route
        # through the default-stream heap for consistency with the rest
        # of the chunk-manager allocations even though the byte
        # footprint is trivial. No record_stream needed: the placeholder
        # is process-lived and never the consumer of any kernel work
        # (it's a 0-element ``param.data`` sentinel).
        if self.device.type == "cuda" and torch.cuda.is_available():
            with SingleStreamAllocator():
                t = torch.empty(0, device=self.device, dtype=dtype)
        else:
            t = torch.empty(0, device=self.device, dtype=dtype)
        self._empty_by_dtype[dtype] = t
        return t

    def _make_grad_offload_hook(self, chunk_id: ChunkId, slot: _CpuParamSlot):
        """Build a post-accumulate grad hook for one trainable non-persistent param.

        Captures ``chunk_id`` + ``slot`` by closure. On fire:

        1. Copy ``param.grad`` into the pinned CPU grad shard.
        2. Null out ``param.grad`` to free GPU storage immediately.
        3. Decrement the chunk's grad counter; if zero, enqueue the
           async CPU Adam step so it overlaps with the remaining GPU
           backward compute (§5).
        """
        cm = self
        # Keep a strong ref to the slot so the param lifetime isn't what
        # keeps it alive.
        captured_slot = slot
        captured_cid = chunk_id

        def _hook(param: "nn.Parameter") -> None:
            if param.grad is None:
                return

            # ---- M7 sharded fast-path ----------------------------------
            # When this chunk has a shard state, the per-param hook does
            # NOT:
            #   * all_reduce the grad (done at chunk level via reduce_scatter)
            #   * copy the grad to CPU (reduce_scatter drains to CPU)
            #   * kick CPU Adam (deferred to reduce_grads_and_offload)
            #   * null the grad (it needs to live on GPU until the
            #     chunk-level reduce_scatter collects every param's grad)
            # We still decrement the chunk counter so the block-level
            # scheduler knows backward-for-this-chunk is done.
            shard_state_local = cm._chunk_shards.get(captured_cid)
            if shard_state_local is not None:
                remaining = cm._grad_remaining.get(captured_cid, 0) - 1
                cm._grad_remaining[captured_cid] = remaining
                return

            # ---- Replicated (non-sharded) path: original M4.5 logic ----
            # Multi-rank data-parallel path: reduce the GPU grad across
            # ranks (AVG = sum / world_size) BEFORE draining to the CPU
            # shard. Guarded on world_size > 1 AND ``skip_internal_grad_reduce``
            # being False — the M6 DDP-composed stack sets the flag to
            # True so DDP's own bucketed allreduce handles this sync
            # and we don't do a second per-param reduce here. In a bare
            # non-DDP distributed run the flag is False and this is the
            # sole grad-sync point.
            import torch as _torch
            import torch.distributed as _dist

            if (
                _dist.is_available()
                and _dist.is_initialized()
                and _dist.get_world_size() > 1
                and not cm.skip_internal_grad_reduce
            ):
                _dist.all_reduce(param.grad, op=_dist.ReduceOp.AVG)
            # copy_ supports cross-device; non_blocking=True is safe
            # because the destination is pinned host memory.
            captured_slot.cpu_grad.copy_(param.grad, non_blocking=True)  # type: ignore[union-attr]
            # BUG 1 FIX: record a CUDA event on the current stream the
            # moment the async D2H is dispatched. The CPU Adam worker
            # thread will synchronize on this event before reading the
            # pinned grad shard — without the wait, the worker can race
            # the D2H and read uninitialized/partial bytes the moment
            # the ThreadPoolExecutor pops its queue (DeepSpeedCPUAdam
            # holds no implicit CUDA-side ordering). Recording the event
            # here (after copy_) captures the D2H completion exactly;
            # the event itself is cheap to record.
            d2h_event = None
            if param.grad.is_cuda:
                d2h_event = _torch.cuda.Event(blocking=True)
                d2h_event.record()
            # Null the grad so PyTorch frees the GPU storage right away —
            # this is the whole point of the per-param hook.
            param.grad = None

            remaining = cm._grad_remaining.get(captured_cid, 0) - 1
            cm._grad_remaining[captured_cid] = remaining
            if remaining == 0:
                # All of the chunk's trainable params are drained. The
                # CPU FusedAdam adapter is responsible for actually
                # updating the offloaded weights — without it, the CPU
                # master shards never advance and every offloaded chunk
                # silently retains its iter-0 weights forever.
                #
                # CodeRabbit R2-05 fix: fail fast the FIRST time an
                # offloaded chunk reaches its CPU-step path with no
                # ``cpu_optim`` attached. Prior code skipped the
                # ``step_async`` and just reset ``_grad_remaining`` so
                # the next backward could fire again — which masked the
                # missing optimizer behind silently stale weights.
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
                # Install the CPU shards onto the param objects and kick
                # off the async step — the adapter was built against the
                # GPU param refs but consumes grads from our CPU shards,
                # so we temporarily repoint ``.data`` and ``.grad`` for it.
                cm._ensure_cpu_grads_attached(captured_cid)
                # BUG 4 FIX: after the worker thread runs
                # ``optim.step()`` the CPU shards hold the updated
                # weights, but ``param.data`` still points at the
                # CPU tensor (we repointed it in
                # _ensure_cpu_grads_attached). Install a post_step
                # callback that repoints ``param.data`` back to the
                # GPU empty placeholder so any intermediate code
                # reading ``.data`` between iters (clip_grad_norm_,
                # checkpoint save, Trainer metric hooks) sees a
                # zero-element GPU tensor — matching the invariant
                # the rest of the runtime relies on. The CPU master
                # weights are still held by ``slot.cpu_data`` so
                # the next gather() flows the updated values back
                # to GPU via its H2D copy.
                cm.cpu_optim.step_async(
                    captured_cid,
                    d2h_event=d2h_event,
                    post_step=cm._make_post_cpu_step_repoint(captured_cid),
                )
                # Reset the counter now so the next backward fires again.
                cm._grad_remaining[captured_cid] = cm._grad_initial.get(captured_cid, 0)

        return _hook

    def _make_post_cpu_step_repoint(self, chunk_id: ChunkId):
        """Build the after-step callback that repoints ``.data`` back to GPU.

        BUG 4 FIX: between the end of iter N's optimizer step and the
        start of iter N+1's gather, ``param.data`` must be a GPU tensor
        (zero-element is fine — it's the same empty-placeholder used
        elsewhere in the runtime). If we leave it pointing at the CPU
        master shard, any caller between iters (clip_grad_norm_, Trainer
        logging, checkpoint save) sees a CPU tensor where a GPU tensor
        was expected. The CPU shard continues to hold the post-step
        weights; the next :meth:`gather` H2D-copies them into the GPU
        buffer.
        """
        cm = self
        captured_cid = chunk_id

        def _repoint() -> None:
            slots = cm._cpu_slots.get(captured_cid, [])
            for slot in slots:
                param = cm._params_by_id.get(slot.param_id)
                if param is None:
                    continue
                param.data = cm._empty_placeholder(slot.dtype)
                # Also clear grad: we've consumed it in the CPU step,
                # and leaving param.grad pointing at the CPU grad shard
                # means iter N+1's autograd would accumulate new GPU
                # grad onto a CPU tensor → "expected same device" fail.
                param.grad = None

        return _repoint

    def _ensure_cpu_grads_attached(self, chunk_id: ChunkId) -> None:
        """Prepare the non-persistent chunk for its CPU Adam step.

        The CPU FusedAdam adapter was built over the GPU ``nn.Parameter``
        objects (see ``protrain_optimizer_wrapper``). For the CPU step to
        consume the drained grads, we temporarily:

        * Point each param's ``.data`` at its CPU shard (so Adam updates
          the CPU master in place).
        * Point each param's ``.grad`` at its CPU grad shard.

        This matches DeepSpeed's CPU-offload pattern where the optimizer
        holds param references but those references are repointed at CPU
        storage for the step's duration. ``gather`` will re-point ``.data``
        back at the GPU buffer after the step (the CPU shard's updated
        bytes flow back via the gather's H2D copy).
        """
        slots = self._cpu_slots.get(chunk_id, [])
        for slot in slots:
            param = self._params_by_id.get(slot.param_id)
            if param is None:
                continue
            # Swap .data to point at the CPU master so the CPU Adam kernel
            # has somewhere to read/write. This is a view of pinned memory;
            # no allocation.
            param.data = slot.cpu_data
            param.grad = slot.cpu_grad

    # ---- gather / offload ---------------------------------------------

    def gather(self, chunk_id: ChunkId) -> None:
        """Make ``chunk_id``'s params GPU-resident.

        Persistent chunks: no-op — they were never offloaded.

        Non-persistent chunks (replicated path): acquire a GPU buffer
        from the pool, copy the chunk's CPU bytes into it (skipping the
        copy if the chunk is already resident-tagged in the pool), and
        rebind every param's ``.data`` to a GPU view.

        Non-persistent chunks (sharded path, ``zero3_shard=True`` AND
        chunk has a shard state): each rank H2D-uploads its
        ``shard_bytes`` CPU shard into a slice of the pool buffer, then
        issues ``torch.distributed.all_gather_into_tensor`` to fill the
        full-chunk buffer from every rank's contribution. After the
        collective the buffer holds the full chunk on every rank, and
        params are rebound exactly as in the replicated path.

        Unlike the M2 stub signature, this method no longer returns the
        tensor — the side effect is the ``param.data`` rebind, and the
        raw buffer is owned by the pool.
        """
        if chunk_id in self._persistent_ids:
            return

        if chunk_id not in self._cpu_slots:
            # materialize_offload wasn't called, or this chunk had no
            # params — nothing to do.
            return

        # Past the persistent early-return: every code path below
        # routes through ``self.buffer_pool``. The all-persistent
        # construction path (``buffer_pool=None``) cannot reach here
        # because every chunk would have hit the ``_persistent_ids``
        # branch above. Assert for type narrowing + defense in depth.
        assert self.buffer_pool is not None, (
            "gather() reached the non-persistent path with no buffer_pool; "
            "all-persistent layouts must early-return above"
        )

        shard_state = self._chunk_shards.get(chunk_id)

        # Forward→backward reuse fast path (paper §3.1.1: "buffer-cached
        # chunks skip re-gather in backward"). The buffer pool preserves
        # the chunk's tag on ``release`` and only drops it when the slot
        # is re-acquired for a different chunk (see BufferPool.acquire's
        # eviction branch). Consequently:
        #
        # * If ``lookup_resident(chunk_id)`` returns a buffer, the slot's
        #   bytes are still the SAME bytes the previous gather wrote
        #   there — every rank's full-chunk reconstruction is intact and
        #   we can skip both the H2D copy (replicated path) AND the
        #   ``all_gather_into_tensor`` collective (sharded path).
        # * If it returns None, an intervening ``acquire`` for some
        #   other chunk evicted the tag (and overwrote the bytes); we
        #   take the full miss path below.
        #
        # The skip is the single biggest throughput win on PCIe-bound
        # 4-GPU 3090 setups (Item 5 profiling pass): each avoided
        # all_gather is ~290MB of cross-PCIe motion at the 10-12 GB/s
        # NCCL ring ceiling. Skipping it costs nothing in correctness:
        # the sharded gather's only output is the full-chunk byte image
        # in the pool buffer, and ``lookup_resident`` is the proof that
        # image is still there.
        resident_buf = self.buffer_pool.lookup_resident(chunk_id)
        if resident_buf is not None:
            # Re-claim the slot (idempotent if already in-use; pops the
            # free list if it was released after forward).
            buf = self.buffer_pool.acquire(chunk_id)
            self._rebind_params_to_buffer(chunk_id, buf, needs_copy=False)
            return

        # Cache miss: the slot was evicted or never populated. Acquire a
        # fresh slot (which evicts some OTHER chunk's tag if the free
        # list is non-empty), then either (a) issue per-region
        # all_gathers in sharded mode or (b) per-slot H2D copies in
        # replicated mode.
        buf = self.buffer_pool.acquire(chunk_id)
        if shard_state is not None:
            self._gather_sharded(chunk_id, buf, shard_state)
            self._rebind_params_to_buffer(chunk_id, buf, needs_copy=False)
            return

        # Replicated path: per-slot H2D copies directly into the buffer.
        self._rebind_params_to_buffer(chunk_id, buf, needs_copy=True)

    def _gather_sharded(
        self,
        chunk_id: ChunkId,
        buf: "torch.Tensor",
        shard_state: "_ChunkShardState",
    ) -> None:
        """ZeRO-3 all_gather path: reconstruct the full chunk on GPU.

        One :func:`all_gather_into_tensor` collective per
        :class:`_DtypeRegion` — homogeneous chunks issue exactly one
        collective (matches the pre-followup single-region fast path);
        mixed-dtype chunks issue N collectives, one per dtype region.

        For each region:

        1. H2D copy this rank's pinned ``shard_bytes`` slice into a
           GPU staging tensor.
        2. all_gather_into_tensor to a padded per-region scratch
           tensor (``region_bytes_padded`` bytes).
        3. Copy the valid ``region_bytes`` prefix into the pool buffer
           at ``chunk_offset``. The scratch is freed when it falls out
           of scope.

        Step 3 is what keeps the pool buffer's byte layout identical
        to the replicated path — ``_rebind_params_to_buffer`` can
        then index every param at its original byte_offset without
        caring whether sharding was engaged.
        """
        import torch
        import torch.distributed as dist

        from axolotl.integrations.protrain.runtime.streams import (
            SingleStreamAllocator,
        )

        # App B.2 wire-up: this method is called from
        # ``ChunkManager.gather`` which the scheduler invokes inside
        # ``with torch.cuda.stream(self._prefetch_stream):`` (see
        # ``Scheduler._gather_on_prefetch_stream``). Without the
        # SingleStreamAllocator wrapper below, ``torch.empty`` would
        # land on the prefetch-stream's heap — and the next default-
        # stream allocation could not reuse the bytes after these
        # transient scratch tensors fall out of scope, fragmenting the
        # allocator. Routing them through the default-stream heap and
        # then issuing ``record_stream`` to whatever stream is current
        # at call time keeps the allocator-free path correctly gated
        # on the consuming work.
        cur_stream: "torch.cuda.Stream | None" = None
        on_cuda = buf.device.type == "cuda" and torch.cuda.is_available()
        if on_cuda:
            cur_stream = torch.cuda.current_stream(device=buf.device)
        # Skip the wrap when the caller is already on the default
        # stream — the allocations would already land on the right heap
        # and the ``record_stream`` calls would be no-ops. This keeps
        # the CPU-only / synchronous-fallback paths zero-overhead.
        wrap_alloc = (
            on_cuda
            and cur_stream is not None
            and (cur_stream != torch.cuda.default_stream(device=buf.device))
        )

        for region in shard_state.regions:
            # Staging: this rank's shard on GPU.
            if wrap_alloc:
                with SingleStreamAllocator():
                    my_shard_gpu = torch.empty(
                        region.shard_bytes, dtype=torch.uint8, device=buf.device
                    )
                # Tie the buffer's lifetime to whichever stream is
                # actually about to consume it — the prefetch stream
                # in steady-state, the default stream in the
                # synchronous fallback. Without record_stream the
                # default-stream allocator could free the storage
                # while the H2D/all_gather below is still in flight.
                my_shard_gpu.record_stream(cur_stream)  # type: ignore[arg-type]
            else:
                my_shard_gpu = torch.empty(
                    region.shard_bytes, dtype=torch.uint8, device=buf.device
                )
            my_shard_gpu.copy_(region.cpu_shard_bytes, non_blocking=True)

            # Gather output scratch: region_bytes_padded (may be > region_bytes).
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
            dist.all_gather_into_tensor(gather_scratch, my_shard_gpu)

            # Write the valid-bytes prefix into the pool buffer at the
            # region's chunk offset. The pool buffer is S_chunk wide
            # and already zero-sentinelled on the first acquire; the
            # narrow() slice here covers exactly the original region
            # bytes the params' byte_offsets index into.
            buf.narrow(0, region.chunk_offset, region.region_bytes).copy_(
                gather_scratch.narrow(0, 0, region.region_bytes)
            )

    def _rebind_params_to_buffer(
        self,
        chunk_id: ChunkId,
        buf: "torch.Tensor",
        needs_copy: bool,
    ) -> None:
        """Copy CPU shards into ``buf`` (if needed) and rebind each param's data.

        ``buf`` is the pool-owned GPU uint8 tensor of length ``S_chunk``.
        For each param slot we slice off
        ``slot.byte_offset .. +slot.numel*slot.element_size``, reinterpret
        it as the param's dtype, reshape to the param's shape, and
        assign to ``param.data``. ``slot.byte_offset`` already includes
        any per-param alignment padding applied by
        :meth:`materialize_offload` (BUG 2 fix), so the GPU buffer layout
        mirrors the pinned CPU layout exactly.
        """
        slots = self._cpu_slots.get(chunk_id, [])
        if not slots:
            return

        if needs_copy:
            for slot in slots:
                nbytes = slot.numel * slot.element_size
                # Slice the buffer at this param's recorded
                # (alignment-padded) byte offset — same offset used for
                # the pinned CPU layout in materialize_offload — and view
                # as the param's dtype+shape for an element-typed copy.
                dst_bytes = buf.narrow(0, slot.byte_offset, nbytes)
                dst_typed = dst_bytes.view(slot.dtype).view(slot.shape)
                dst_typed.copy_(slot.cpu_data, non_blocking=True)

        # Rebind .data unconditionally — even on the no-copy path, a
        # previous offload() nulled out param.data, and re-acquiring from
        # the pool keeps the GPU bytes but requires re-pointing the
        # param at them.
        for slot in slots:
            param = self._params_by_id.get(slot.param_id)
            if param is None:
                continue
            nbytes = slot.numel * slot.element_size
            # Slice the chunk buffer at this param's byte offset (with
            # alignment padding already baked in) and view as
            # (dtype, shape).
            byte_view = buf.narrow(0, slot.byte_offset, nbytes)
            typed = byte_view.view(slot.dtype).view(slot.shape)
            param.data = typed

        # M2: register the chunk's flat buffer storage in the reverse
        # lookup so OffloadedBlock._pack can identify saved tensors
        # that view this chunk. Every per-param view rebound above
        # shares ``buf``'s storage (``narrow`` + ``view`` keep
        # storage identity), so a single entry per chunk suffices.
        try:
            ptr = buf.untyped_storage().data_ptr()
        except Exception:  # noqa: BLE001 — defensive on unusual backends
            ptr = 0
        if ptr:
            self._storage_ptr_to_chunk[ptr] = chunk_id

    def offload(self, chunk_id: ChunkId) -> None:
        """Release ``chunk_id``'s GPU storage (non-persistent only).

        Null out every param.data back to the empty sentinel, then return
        the buffer to the pool. The pool keeps the resident tag (so a
        backward-pass gather within the reuse window can skip the H2D
        re-copy) — but the param-level bindings are severed here so
        nothing tries to read stale GPU bytes after the pool reassigns
        the slot to a different chunk.

        BUG FIX: skip the ``param.data = empty_placeholder`` re-bind when
        ``param.data`` is already on CPU. In the replicated non-sharded
        path the per-param grad hook calls ``_ensure_cpu_grads_attached``
        right before kicking the async CPU Adam step — that points
        ``param.data`` at the pinned CPU shard so DeepSpeedCPUAdam can
        read/write it. The block-granularity scheduler then calls
        ``reduce_grads_and_offload`` → ``offload`` on the SAME main
        thread that just enqueued the step. If we re-bind ``param.data``
        back to a GPU placeholder here, the worker thread (which hasn't
        called ``step()`` yet) sees ``p.device == cuda`` and trips
        DeepSpeedCPUAdam's ``"CPUAdam param is on cuda:N and must be
        'cpu'"`` assertion. The post_step callback registered by the
        grad hook (``_make_post_cpu_step_repoint``) is the canonical
        place that returns ``param.data`` to the empty GPU placeholder
        AFTER the CPU step completes, so leaving it on CPU here is
        correct: the next gather repoints it onto the GPU buffer view
        before any compute runs against it.
        """
        if chunk_id in self._persistent_ids:
            return
        # Past the persistent early-return: ``buffer_pool`` is required
        # for the release call below. The all-persistent construction
        # path (``buffer_pool=None``) cannot reach here because every
        # chunk hits the early-return above. Narrow for mypy + assert
        # for defense in depth.
        assert self.buffer_pool is not None, (
            "offload() reached the non-persistent path with no buffer_pool; "
            "all-persistent layouts must early-return above"
        )

        # M2 / Option B: defer the offload if any BackwardHandle is
        # still outstanding for this chunk. The unpack hook returned a
        # view into the pool buffer that autograd is still consuming;
        # releasing the slot now would let an intervening
        # ``acquire(other)`` evict the bytes mid-backward (see §3.4
        # point 2). The drain runs in ``_release_backward_handle``
        # when the last handle drops to zero.
        if self._backward_refcount.get(chunk_id, 0) > 0:
            self._deferred_offloads.add(chunk_id)
            return

        # M2: deregister the storage-ptr reverse lookup BEFORE we
        # null param.data and release the buffer. The pool may keep
        # the slot tagged for forward→backward reuse, but the
        # OFFLOAD pack hook should only resolve a chunk_id from a
        # storage_ptr while the chunk's params are actively bound to
        # a buffer view; clearing here matches that invariant. (The
        # next gather will re-register.)
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
                # One ptr-per-chunk by construction (every param view
                # shares the same buffer storage); break early.
                break

        for slot in slots:
            param = self._params_by_id.get(slot.param_id)
            if param is None:
                continue
            # Don't clobber a CPU-bound param.data: the grad hook just
            # repointed it for the pending CPU Adam step and the
            # post-step repoint will null it back to a GPU placeholder.
            if param.data.device.type == "cpu":
                continue
            param.data = self._empty_placeholder(slot.dtype)
        self.buffer_pool.release(chunk_id)

    def reduce_grads_and_offload(self, chunk_id: ChunkId) -> None:
        """Reduce-scatter grads and D2H-copy the chunk's grad shard back to CPU.

        Persistent chunks: run the reduction (if distributed is live)
        and leave the result on GPU — the GPU optimizer consumes it in
        :meth:`persistent_step`.

        Non-persistent chunks: the per-param post-accumulate-grad hooks
        installed by :meth:`materialize_offload` already drained each
        param's grad to CPU and kicked off the async CPU FusedAdam step
        at the moment the last param's grad landed (§5, ZeRO-Offload).
        All that's left for the block-granularity scheduler to do is
        release the chunk's buffer — the grad work is already in flight.
        """
        import torch

        if chunk_id in self._persistent_ids:
            # Persistent chunks keep their grads GPU-resident for the
            # FusedAdam step.
            #
            # Distributed grad-sync policy. When another layer above
            # ProTrain owns the cross-rank reduction (the M6 stack wraps
            # the protrain'd module in ``DistributedDataParallel``, which
            # fires its own bucketed allreduce via autograd hooks), this
            # in-manager all_reduce would be a redundant second sync —
            # so ``self.skip_internal_grad_reduce`` (set by the wrapper
            # when it detects DDP composition) tells us to leave the
            # grads alone.
            #
            # In the non-DDP distributed path (e.g. a bare ZeRO-3 run
            # or Mode-A-no-DDP / Mode-C-no-DDP) the flag is False and
            # we own the cross-rank reduction. To minimize NCCL launch
            # latency on small persistent chunks (Item 5 profiling
            # showed ~19 ops × 17MB unbucketed on a Llama-3B 4-GPU run,
            # ~30 ms / 1300 ms iter), we COALESCE every same-dtype grad
            # in the chunk into a single flat buffer and issue one
            # ``all_reduce`` per dtype group. PyTorch's
            # ``_flatten_dense_tensors`` / ``_unflatten_dense_tensors``
            # is the same primitive DDP uses internally; it handles
            # the contiguous-buffer staging and the per-tensor view
            # restoration without any copy back when the grads were
            # already contiguous (the common case).
            #
            # Mixed-dtype chunks (e.g. fp16 attention weights next to
            # fp32 layernorm scales in a Llama block) issue ONE
            # all_reduce per dtype run, not one per param. Homogeneous
            # chunks issue exactly one collective — the structurally
            # cleanest case.
            if (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
                and torch.distributed.get_world_size() > 1
                and not self.skip_internal_grad_reduce
            ):
                self._coalesced_all_reduce_persistent_grads(chunk_id)
            return

        # ---- Non-persistent sharded path -------------------------------
        shard_state = self._chunk_shards.get(chunk_id)
        if shard_state is not None:
            self._reduce_scatter_and_offload_shard(chunk_id, shard_state)
            self.offload(chunk_id)
            return

        # Non-persistent, replicated: grad offload is owned by
        # _offload_grad (per-param hooks). The block-granularity
        # scheduler here releases the chunk buffer AND nulls the
        # param.data placeholder so the GPU storage is fully freed and
        # the params are in a clean state for the next gather.
        self.offload(chunk_id)

    def _coalesced_all_reduce_persistent_grads(self, chunk_id: ChunkId) -> None:
        """Bucket persistent-chunk grads by dtype and issue one all_reduce per bucket.

        Replaces the per-param ``dist.all_reduce`` loop that dominated
        launch latency on the Mode-C / Mode-A-no-DDP path (Item 5
        profiling: 19 ops × 17MB unbucketed → ~30 ms/iter). Equivalent
        to PyTorch DDP's internal bucketed allreduce (which uses the
        same ``_flatten_dense_tensors`` primitive).

        Algorithm:

        1. Group every live ``param.grad`` in ``chunk_id`` by dtype.
        2. For each dtype group: flatten into one contiguous buffer,
           ``all_reduce(op=AVG)`` it once, then unflatten back to
           per-param views and copy each view into the original
           ``param.grad``. The copy_back handles the case where
           ``_flatten_dense_tensors`` materialized a fresh buffer (it
           always does — the input grads' storage is independent).

        Mixed-dtype chunks (Llama: fp16 weights + fp32 RMSNorm scales)
        issue one collective per dtype run, exactly like the sharded
        path's per-region collectives. Empty chunks issue zero
        collectives.
        """
        import torch.distributed as dist
        from torch._utils import (
            _flatten_dense_tensors,
            _unflatten_dense_tensors,
        )

        # Collect all live grads for this chunk, grouped by dtype.
        # Maintaining param-order within each dtype group is important:
        # the unflatten step relies on the order matching the input
        # tensors so the typed views land back on the right grads.
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

        for _dtype, pairs in grads_by_dtype.items():
            if not pairs:
                continue
            grads = [p[0] for p in pairs]
            if len(grads) == 1:
                # Single-grad dtype group: skip the flatten/unflatten
                # round-trip entirely (it would be a wasteful copy +
                # copy_back for no bandwidth saving). One all_reduce
                # on the grad in-place matches the legacy path's
                # behaviour exactly.
                dist.all_reduce(grads[0], op=dist.ReduceOp.AVG)
                continue

            # Flatten -> one collective -> unflatten back into the
            # original grads' storage. ``_flatten_dense_tensors`` always
            # returns a fresh contiguous buffer; the unflattened views
            # alias INTO that buffer, so we must copy each view back to
            # the corresponding original ``param.grad`` (autograd /
            # FusedAdam read from the original storage, not the
            # flattened one).
            flat = _flatten_dense_tensors(grads)
            dist.all_reduce(flat, op=dist.ReduceOp.AVG)
            for orig, view in zip(
                grads, _unflatten_dense_tensors(flat, grads), strict=True
            ):
                # ``copy_`` works in-place on ``orig``'s storage. Same
                # device by construction (every grad in this group was
                # already on the same device as the param).
                orig.copy_(view)

    def _reduce_scatter_and_offload_shard(
        self, chunk_id: ChunkId, shard_state: "_ChunkShardState"
    ) -> None:
        """Sharded path: reduce_scatter chunk grads, D2H shard, kick CPU Adam.

        One :func:`reduce_scatter_tensor` collective per
        :class:`_DtypeRegion` — homogeneous chunks issue exactly one
        collective; mixed-dtype chunks issue N collectives, one per
        dtype region. D2H into a per-region pinned grad shard, then
        kick the region's CPU FusedAdam step.

        Precondition: every trainable param in the chunk has a GPU grad
        (backward drained the chunk). Postcondition: every GPU grad is
        nulled, every region's CPU shard grad holds its slice of the
        ``AVG``-reduced cross-rank grad, and the CPU Adam step for
        the chunk has been submitted to the async worker (once; the
        adapter bundles all regions' shard_params under the chunk key).
        """
        import torch
        import torch.distributed as dist

        from axolotl.integrations.protrain.runtime.streams import (
            SingleStreamAllocator,
        )

        slots = self._cpu_slots.get(chunk_id, [])
        if not slots:
            return

        # Device from the first live param.grad (all params in a chunk
        # share a device by construction).
        device = self.device
        any_grad = False
        for slot in slots:
            p = self._params_by_id.get(slot.param_id)
            if p is not None and p.grad is not None:
                device = p.grad.device
                any_grad = True
                break
        if not any_grad:
            return

        # App B.2: this method runs from ``Scheduler.post_block_backward``
        # which today does not wrap in a non-default stream context, so
        # the per-region scratch buffers below would already land on the
        # default-stream heap. We wrap defensively anyway: if a future
        # caller invokes this from inside a non-default stream context
        # (e.g. a swap-stream-driven gradient finalizer), the wrap keeps
        # the heap selection canonical and ``record_stream`` ties the
        # buffer's lifetime to the actually-consuming stream. Computed
        # once outside the loop because ``current_stream`` is a syscall
        # we don't want to repeat per region.
        on_cuda = device.type == "cuda" and torch.cuda.is_available()
        cur_stream: "torch.cuda.Stream | None" = None
        wrap_alloc = False
        if on_cuda:
            cur_stream = torch.cuda.current_stream(device=device)
            wrap_alloc = cur_stream != torch.cuda.default_stream(device=device)

        # Build an index from slot.byte_offset -> slot so we can quickly
        # locate every param whose bytes land inside a given region.
        # Slots are ordered by byte_offset within a chunk (the
        # aligned-offsets pass in ``materialize_offload`` preserves
        # input order), so a linear scan per region is fine.

        d2h_event = None
        any_trainable_region = False
        for region in shard_state.regions:
            # CodeRabbit R07 fix: skip frozen-only regions outright.
            # Their ``shard_param`` was constructed with
            # ``requires_grad=False`` and ``cpu_shard_grad_bytes=None``;
            # there is nothing to reduce or D2H here. Running the
            # collective + binding a zero-grad view as
            # ``shard_param.grad`` would re-introduce the original
            # bug — Adam's weight-decay path would mutate frozen
            # bytes against a silently-zero grad. The trainability
            # flag is authoritative because region segmentation in
            # :meth:`materialize_offload` splits on ``requires_grad``,
            # so any param contributing bytes to a frozen region is
            # guaranteed itself frozen and will never produce a grad.
            if not region.is_trainable:
                continue
            any_trainable_region = True

            r_start = region.chunk_offset
            r_end = r_start + region.region_bytes

            # Stage a padded per-region grad buffer on GPU so
            # reduce_scatter's input length matches
            # region_bytes_padded. Trailing (padding) bytes stay zero.
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
                # Param offset relative to the region's start.
                rel_off = slot.byte_offset - r_start
                dst_bytes = region_grad.narrow(0, rel_off, nbytes)
                dst_typed = dst_bytes.view(slot.dtype).view(slot.shape)
                dst_typed.copy_(p.grad)
                # Null the GPU grad now that we've captured its bytes.
                p.grad = None

            # reduce_scatter_tensor requires matching typed views on
            # input (padded full region) and output (this rank's
            # region shard). Use the region's dtype.
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

            # Re-bind shard_param.grad to its canonical pinned-CPU view
            # if a caller (e.g. HF Trainer with default args) cleared
            # it via ``optim.zero_grad(set_to_none=True)``. The Adam
            # adapter operates on the persistent ``cpu_shard_grad_bytes``
            # pinned buffer; we just need ``.grad`` to point at it again
            # so ``.copy_()`` lands in the right place.
            #
            # ``cpu_shard_grad_bytes`` is non-None here because the
            # ``region.is_trainable`` guard above filtered out the
            # frozen-region case where it stays None. The cast below
            # is purely for the type-checker.
            assert region.cpu_shard_grad_bytes is not None
            if region.shard_param.grad is None:
                region.shard_param.grad = region.cpu_shard_grad_bytes.view(
                    region.dtype
                ).view(shard_numel_r)

            if my_shard_grad_gpu.is_cuda:
                region.shard_param.grad.copy_(  # type: ignore[union-attr]
                    my_shard_grad_gpu, non_blocking=True
                )
                ev = torch.cuda.Event(blocking=True)
                ev.record()
                d2h_event = ev  # last region's event is enough — the
                # CPU Adam worker waits on it before running Adam;
                # because prior regions' D2Hs were launched on the
                # same default stream the last event is at-or-after
                # all previous region copies.
            else:
                region.shard_param.grad.copy_(my_shard_grad_gpu)  # type: ignore[union-attr]

        # CodeRabbit R2-05 fix: if we just reduce_scatter'd / D2H'd grads
        # for at least one trainable region but no CPU optimizer is
        # attached, the offloaded master weights would silently never
        # advance. Raise BEFORE resetting ``_grad_remaining`` so the
        # next backward fires the same condition again rather than
        # silently masking the bad state. Distinct from the R07
        # frozen-region guard above (which is about ``is_trainable``
        # per region — purely a routing concern within this loop):
        # this check fires when at least one trainable region exists
        # and the chunk-level ``cpu_optim`` hook is missing entirely.
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

        # Reset the hook counter so the next backward's per-param
        # decrements land correctly.
        self._grad_remaining[chunk_id] = self._grad_initial.get(chunk_id, 0)

        # Kick async CPU Adam for this chunk — the adapter was built
        # against every region's shard_param for this chunk, so one
        # step_async call updates every region's slice at once.
        if self.cpu_optim is not None:
            self.cpu_optim.step_async(chunk_id, d2h_event=d2h_event, post_step=None)

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

    # ---- cleanup -------------------------------------------------------

    def uninstall(self) -> None:
        """Remove every registered per-param grad hook. Idempotent."""
        for handle in self._grad_hook_handles:
            try:
                handle.remove()  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001 — best-effort
                LOG.debug("ChunkManager.uninstall: hook remove failed: %s", exc)
        self._grad_hook_handles.clear()

    def __del__(self) -> None:  # noqa: D401
        try:
            self.uninstall()
        except Exception:  # noqa: BLE001 — destructors must not throw
            pass
        try:
            # GC safety net: release the unified pinned pools if
            # ``restore_to_gpu`` was never called. By the time
            # ``__del__`` fires, the manager's ``_cpu_slots`` /
            # ``_chunk_shards`` dicts are about to be reclaimed too,
            # so dropping the borrow + closing here is safe in the
            # common single-owner case. If external code still holds
            # shard_param / cpu_data views at GC time, the
            # PinnedHostMemory destructor's "live borrows → leak"
            # guard kicks in (no cudaFreeHost) — but our balanced
            # release means the borrow IS already at zero, so the
            # close proceeds. Document the lifetime hazard in
            # :meth:`materialize_offload`.
            self._close_cpu_pools()
        except Exception:  # noqa: BLE001 — destructors must not throw
            pass

    # ---- M2 / Option B: backward-window pinning -----------------------

    def chunk_id_for_storage_ptr(self, ptr: int) -> "ChunkId | None":
        """Look up the chunk whose pool buffer storage starts at ``ptr``.

        ``OffloadedBlock._pack`` calls this to detect whether a saved
        tensor aliases a chunk-managed param view. The reverse lookup
        is populated by ``_rebind_params_to_buffer`` at gather time
        and cleared by ``offload`` (modulo the deferred path).

        Returns ``None`` if no chunk is currently registered at the
        given pointer — either the saved tensor is a pure activation
        (SWAP's domain) or the chunk has already been offloaded.
        """
        return self._storage_ptr_to_chunk.get(ptr)

    def gather_for_backward(self, chunk_id: ChunkId) -> "BackwardHandle":
        """Re-gather a chunk for the backward pass and pin it via refcount.

        Used by ``OffloadedBlock._unpack`` to re-materialize a chunk
        whose forward-side gather buffer was offloaded in
        ``post_block_forward``. The semantics:

        1. ``gather(chunk_id)`` — idempotent if the chunk is already
           resident; takes the H2D / all_gather path otherwise.
        2. Increment the per-chunk ``_backward_refcount``.
        3. Return a :class:`BackwardHandle` whose ``__del__`` /
           ``release`` decrements the count and drains any deferred
           offload queued for the chunk.

        The refcount is what keeps the pool slot from being evicted
        by an unrelated ``acquire`` call mid-backward (see §3.4 point
        2 of BLOCK_MODE_OFFLOAD_DESIGN). Multiple unpack calls for the
        same chunk in the same backward pass each get their own handle
        and refcount stays at the high-water mark until they all drop.
        """
        self.gather(chunk_id)
        self._backward_refcount[chunk_id] = self._backward_refcount.get(chunk_id, 0) + 1
        return BackwardHandle(chunk_id, self)

    def _release_backward_handle(self, chunk_id: ChunkId) -> None:
        """Decrement ``chunk_id``'s refcount and drain any deferred offload.

        Called by :meth:`BackwardHandle.release` (and indirectly by
        ``BackwardHandle.__del__``). When the count hits zero AND a
        prior ``offload(cid)`` / ``reduce_grads_and_offload(cid)`` was
        deferred (because the count was non-zero when it ran), the
        actual offload runs now — closing the §3.4 deferred-offload
        loop without scheduler involvement.
        """
        cur = self._backward_refcount.get(chunk_id, 0)
        if cur <= 1:
            # Refcount hits zero: drop the entry to keep the dict tidy
            # and run any deferred offload before we release.
            self._backward_refcount.pop(chunk_id, None)
            if chunk_id in self._deferred_offloads:
                self._deferred_offloads.discard(chunk_id)
                # The deferred offload was queued from offload() OR
                # reduce_grads_and_offload(). Either way, the
                # block-level reduce already ran (or didn't apply);
                # the only thing left is the buffer release + param
                # data nulling that ``offload`` does. Re-entering
                # offload() here is safe because the refcount is now
                # zero (we just popped it) and the ``> 0`` guard
                # won't redirect us back into deferral.
                self.offload(chunk_id)
        else:
            self._backward_refcount[chunk_id] = cur - 1

    def drain_deferred_offloads(self) -> int:
        """Flush every deferred offload whose backward refcount is now zero.

        Defensive end-of-iteration drain (M3, §3.3 of
        BLOCK_MODE_OFFLOAD_DESIGN). Today's Python ref-counting on
        :class:`BackwardHandle` already drains via ``__del__`` when the
        last unpack-returned view is collected, so in steady state this
        method is a no-op. It exists to:

        * make the drain timing explicit and composable with future
          schedulers that might want a deterministic flush point
          (e.g. before ``optimizer.step``);
        * give debug paths an assertable invariant — after
          ``Scheduler.drain``, ``_deferred_offloads`` MUST be empty if
          every backward handle has dropped, otherwise something
          leaked a strong reference into the autograd graph.

        Chunks whose refcount is still > 0 are intentionally left in
        ``_deferred_offloads``; the eventual handle drop will trigger
        :meth:`_release_backward_handle` which will offload them then.

        Returns
        -------
        int
            Number of chunks actually offloaded by this drain (i.e.
            chunks whose deferred offload was queued AND whose refcount
            was zero at call time). Useful for telemetry / asserts.
        """
        # Snapshot to avoid concurrent mutation: ``offload`` clears the
        # entry from ``_deferred_offloads`` via the path through
        # ``_release_backward_handle`` semantics OR directly when its
        # ``> 0`` guard fails-through.
        drained = 0
        for cid in tuple(self._deferred_offloads):
            if self._backward_refcount.get(cid, 0) > 0:
                continue
            # Pop before calling offload so the offload path's deferral
            # guard sees a clean refcount of zero (it would otherwise
            # re-add the entry, masking the drain). This mirrors the
            # _release_backward_handle path: discard, then call offload.
            self._deferred_offloads.discard(cid)
            self.offload(cid)
            drained += 1
        return drained

    # ---- introspection for tests --------------------------------------

    def sharded_chunk_ids(self) -> list[ChunkId]:
        """Return the list of chunks currently held in ZeRO-3 sharded form.

        Useful for test assertions: a non-empty list confirms the
        ``zero3_shard`` path engaged at ``materialize_offload`` time.
        """
        return sorted(self._chunk_shards.keys())

    def shard_bytes_for(self, chunk_id: ChunkId) -> int:
        """Return this rank's total pinned CPU shard bytes for ``chunk_id``.

        Sum across every :class:`_DtypeRegion` in the chunk. Returns
        0 when the chunk is not sharded (persistent, or ``zero3_shard``
        was off at materialize time).
        """
        s = self._chunk_shards.get(chunk_id)
        return 0 if s is None else s.shard_bytes

    def per_rank_cpu_bytes(self) -> int:
        """Total pinned CPU bytes this rank holds across every sharded chunk.

        Sums BOTH the per-region shard buffer (``cpu_shard_bytes``) and
        the per-region grad buffer (``cpu_shard_grad_bytes``) when
        present. ``cpu_shard_bytes`` is allocated for every sharded
        region; ``cpu_shard_grad_bytes`` is allocated only for trainable
        regions (frozen-only regions skip it as part of the CodeRabbit
        R07 fix — no Adam step, no need for the pinned grad shard).
        Convenience accessor for the 4-GPU sharding test which asserts
        per-rank CPU footprint roughly equals
        ``total_non_persistent_bytes / world_size`` and for benchmark
        scripts reporting Mode-C host RAM.
        """
        total = 0
        for shard_state in self._chunk_shards.values():
            for region in shard_state.regions:
                total += int(region.cpu_shard_bytes.numel())
                if region.cpu_shard_grad_bytes is not None:
                    total += int(region.cpu_shard_grad_bytes.numel())
        return total

    def replicated_cpu_bytes(self) -> int:
        """Total pinned CPU bytes this rank holds in replicated (non-sharded) mode.

        Sums ``(numel * element_size)`` for every per-param ``cpu_data``
        and ``cpu_grad`` slot across every non-persistent chunk. Mirrors
        :meth:`per_rank_cpu_bytes` (which is for ZeRO-3-style sharding)
        for the replicated-offload layout where every rank holds the
        full chunk in pinned host memory. Used by benchmark scripts so
        they do not have to reach into the private ``_cpu_slots``
        mapping.
        """
        total = 0
        for slots in self._cpu_slots.values():
            for s in slots:
                if s.cpu_data is not None:
                    total += s.numel * s.element_size
                if s.cpu_grad is not None:
                    total += s.numel * s.element_size
        return total

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

        # Source the device from ``self.device`` rather than
        # ``self.buffer_pool.device`` so this works in the
        # all-persistent layout where ``buffer_pool is None``.
        # ``self.device`` is canonical (always set in __init__) and
        # equal to ``buffer_pool.device`` when a pool exists.
        #
        # App B.2: persistent chunk buffers are long-lived — they sit
        # GPU-resident for the entire training run — so routing their
        # allocation through the default-stream heap unifies them with
        # the buffer-pool slots and the rest of the chunk-manager state.
        # No ``record_stream`` needed (long-lived, no cross-stream
        # release race).
        if self.device.type == "cuda" and torch.cuda.is_available():
            with SingleStreamAllocator():
                buf = torch.empty(
                    self.layout.S_chunk,
                    dtype=torch.uint8,
                    device=self.device,
                )
        else:
            buf = torch.empty(
                self.layout.S_chunk,
                dtype=torch.uint8,
                device=self.device,
            )
        self._persistent_buffers[chunk_id] = buf
        return buf


__all__ = ["BackwardHandle", "ChunkManager"]
