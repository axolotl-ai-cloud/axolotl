"""Param-offload-aware block wrapper (Option B, §3.2 of BLOCK_MODE_OFFLOAD_DESIGN).

OFFLOAD mode in the four-way ProTrain block strategy: a non-persistent
chunk's owning block runs WITHOUT activation recompute. Forward
proceeds normally; activations stay on GPU; the chunk gets offloaded
after the block's forward (saved tensors no longer pin GPU storage
because the saved-tensors-hooks below replaced them with metadata
handles); backward re-gathers the chunk via
:meth:`ChunkManager.gather_for_backward` and the unpack hook re-views
the gathered pool buffer at the original storage offset.

Compared to ``SwappedBlock`` (SWAP), the structural template is
identical — same ``saved_tensors_hooks`` context, same wrap-then-attach
pattern — but the SEMANTICS differ:

================  ===================  ============================
Saved-tensor      ``SwappedBlock``     ``OffloadedBlock``
================  ===================  ============================
Pack does         D2H copy to slot     record (cid, offset, shape)
Unpack does       H2D from slot        re-view gathered pool buffer
Pool used         pinned host          ChunkManager.buffer_pool (GPU)
Bytes copied      one D2H per save     zero (handle is metadata only)
================  ===================  ============================

The pack hook MUST drop its strong reference to the GPU tensor — that
is the whole point. Returning the tensor as-is (the SWAP pass-through
fallback) would defeat the design: autograd's saved-tensor table would
still pin the chunk buffer's GPU storage, ``post_block_forward`` could
not safely release it, and OFFLOAD would degrade to plain NONE on a
non-persistent chunk (the failure mode that motivated this design).

Lifetime / ordering invariants
------------------------------
* ``pre_block_backward(N)`` MUST fire before the autograd engine
  invokes any unpack hook for tensors saved during block N's forward.
  M3's scheduler integration guarantees this — the wrapper module's
  forward-pre hook fires before autograd starts decoding the block's
  saved tensors. Breaking this ordering is the single most subtle
  failure mode of OFFLOAD; the unpack hook would call
  ``gather_for_backward`` itself but cross-rank collectives in the
  sharded path require every rank to participate at the same step.
* Saved tensors that are NOT param-aliasing pass through the hook
  unchanged. Pure activations are SWAP's job, not ours; the hook
  detects "is this a param view?" by storage-pointer lookup against
  ``ChunkManager._storage_ptr_to_chunk`` (populated at gather time,
  cleared at offload time).
* The :class:`BackwardHandle` returned by ``gather_for_backward``
  refcounts the chunk buffer slot. Its lifetime MUST outlive the
  autograd engine's reference to the unpack-returned view. We pin the
  handle to the view via a private attribute on the view tensor —
  PyTorch's autograd holds the unpacked tensor object until the
  consuming Node's ``apply()`` returns; once autograd drops it,
  Python ref-counting frees both the view and (transitively) the
  attached ``BackwardHandle``, whose ``__del__`` decrements the
  manager's refcount and potentially drains a deferred offload.

Why a private attribute and not a weakref-keyed dict? Simplicity. A
weakref dict would require a finalizer keyed off the view's id, which
adds a global-state invariant and a teardown hazard (managers
constructed in tests would leak the dict across the process). Setting
``view._protrain_backward_handle = handle`` is local to the view's
lifetime, costs nothing at allocation, and the attribute is dropped
automatically when the view is GC'd.

Cold path / hot path
--------------------
``attach_runtime`` injects the chunk manager + scheduler post-
construction. Until that call, the wrapper passes the block forward
through unchanged — no saved_tensors_hooks context is installed, so
saved tensors live on GPU as they normally would. This preserves the
"constructible without runtime" surface the test fixtures rely on,
matching the SWAP wrapper's degradation behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from axolotl.integrations.protrain.block.strategy import BlockMode
from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    from axolotl.integrations.protrain.chunk.manager import ChunkManager
    from axolotl.integrations.protrain.types import ChunkId

LOG = get_logger(__name__)


#: Retained for ``SwappedBlock`` parity and test-override compatibility.
#: NOT consulted by ``OffloadedBlock._pack``: gating saves on size would
#: pass through small chunk-managed params (e.g. biases / LayerNorm
#: weights), pinning the chunk buffer past offload and defeating the
#: design. The chunk-storage lookup is the sole gate; non-chunk tensors
#: pass through regardless of size.
SIZE_THRESHOLD_BYTES: int = 1 << 20  # 1 MiB


@dataclass(slots=True, frozen=True)
class _ParamHandle:
    """Metadata handle that survives autograd's saved-tensor table.

    Replaces the strong tensor reference autograd would otherwise hold
    after a save_for_backward of a chunk-managed param's view. The
    pack hook records ``(chunk_id, storage_offset, shape, stride,
    dtype, requires_grad)``; the unpack hook re-gathers the chunk and
    reconstructs the view at ``storage_offset`` with the original
    ``shape``/``stride``/``dtype``.

    ``storage_offset`` is in BYTES from the start of the chunk's
    underlying ``UntypedStorage``. We store bytes (not elements)
    because the chunk buffer is allocated as ``torch.uint8`` and the
    individual params overlay it via dtype-typed views — the byte
    offset is the dtype-agnostic invariant.

    ``stride`` is in ELEMENTS of the param's dtype (matching
    ``torch.Tensor.stride()`` semantics). Capturing it is load-bearing:
    PyTorch's ``F.linear`` saves ``weight`` with stride ``(1, in_dim)``
    rather than the ``(in_dim, 1)`` of a row-major contiguous tensor,
    because it transposes the weight internally for the matmul. If
    ``_unpack`` reconstructed the view with the wrong stride, the
    autograd backward kernels would read the storage in the wrong
    element order — silently producing incorrect upstream gradients.
    """

    chunk_id: "ChunkId"
    storage_offset: int  # byte offset within the chunk's storage
    shape: torch.Size
    stride: tuple[int, ...]  # in elements of dtype
    dtype: torch.dtype
    requires_grad: bool


class OffloadedBlock(nn.Module):
    """Wrap an ``nn.Module`` so its saved param tensors are metadata-only.

    Construction is unconditional. Gating happens via the searcher's
    ``n_offload`` decision (the cost model + admissibility filters).

    The chunk manager is injected post-construction via
    :meth:`attach_runtime`. Until that call, the wrapper passes the
    block forward through unchanged — no saved_tensors_hooks context
    is installed, so saved tensors live on GPU as they normally would.
    This matches ``SwappedBlock``'s behavior so test fixtures that
    construct wrappers without runtime see clean degradation.
    """

    #: Retained for ``SwappedBlock`` parity and test-override compatibility;
    #: not consulted by ``_pack``. See module-level docstring.
    SIZE_THRESHOLD_BYTES: int = SIZE_THRESHOLD_BYTES

    def __init__(self, block: nn.Module) -> None:
        """Wrap ``block`` in identity-mode; runtime wired by :meth:`attach_runtime`."""
        super().__init__()
        self.block = block
        self._protrain_wrapped_mode: BlockMode = BlockMode.OFFLOAD
        self._chunk_manager: "ChunkManager | None" = None
        self._scheduler: Any = None  # M3 owns the scheduler interface contract
        self._warned_no_runtime = False

    def attach_runtime(
        self,
        chunk_manager: "ChunkManager",
        scheduler: Any = None,
    ) -> None:
        """Wire the chunk manager + scheduler into this wrapper.

        Idempotent — re-attaching with the same manager (and updating
        only the scheduler) is allowed. Swapping in a *different*
        chunk manager mid-run is rejected: any ``_ParamHandle``
        previously recorded by ``_pack`` references the prior
        manager's storage map by ``ChunkId``, and resolving those
        handles against a freshly-constructed manager would silently
        decode against unrelated storage. Callers that need to swap
        managers (e.g. a re-search at an epoch boundary) MUST call
        :meth:`detach_runtime` first; that path is only safe between
        forward/backward boundaries when no saved-tensor handles are
        outstanding.
        """
        if self._chunk_manager is not None and self._chunk_manager is not chunk_manager:
            raise RuntimeError(
                "OffloadedBlock.attach_runtime: refusing to swap chunk "
                "managers on an already-attached wrapper. Saved "
                "_ParamHandles from prior forwards reference the old "
                "manager's storage map by ChunkId and would decode "
                "against unrelated storage on the new manager. Call "
                "detach_runtime() first, and only between "
                "forward/backward boundaries when no saved-tensor "
                "handles are outstanding."
            )
        self._chunk_manager = chunk_manager
        self._scheduler = scheduler

    def detach_runtime(self) -> None:
        """Drop the manager reference — wrapper degrades to identity."""
        self._chunk_manager = None
        self._scheduler = None

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run the wrapped block under saved_tensors_hooks that record param handles."""
        mgr = self._chunk_manager

        # Cold path — no runtime attached. Run the block plain. Saved
        # tensors will live on GPU as they normally would; the block
        # isn't really an OFFLOAD block in the runtime sense.
        if mgr is None:
            if not self._warned_no_runtime:
                LOG.warning(
                    "OffloadedBlock forward without attached runtime — "
                    "degrading to identity. Call attach_runtime(chunk_manager) "
                    "after constructing the block."
                )
                self._warned_no_runtime = True
            return self.block(*args, **kwargs)

        # Hot path — install saved_tensors_hooks for the duration of
        # the wrapped block's forward. Every saved tensor created
        # inside this context goes through ``_pack``; backward
        # restores them via ``_unpack``.
        with torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack):
            return self.block(*args, **kwargs)

    # ---- saved-tensors hooks ----------------------------------------------

    def _pack(self, t: torch.Tensor) -> Any:
        """Record metadata for chunk-managed params; pass everything else through.

        The lookup key is ``t.untyped_storage().data_ptr()``. The
        chunk manager populates ``_storage_ptr_to_chunk`` at gather
        time (after every param has been rebound to a view of the pool
        buffer); a hit means ``t`` aliases the chunk's GPU bytes and
        the pack-time strong ref to ``t`` would pin the buffer past
        ``post_block_forward``'s offload — which is exactly what we
        must avoid.

        Returns
        -------
        - ``_ParamHandle`` if ``t`` is a chunk-managed param view.
        - ``t`` (passthrough) if ``t`` is anything else: a pure
          activation, a tensor on a non-CUDA device, or a tensor
          whose storage isn't tracked by the chunk manager. Pure
          activations are SWAP's domain, not ours; passing them
          through cleanly composes the OFFLOAD context with an outer
          SWAP context if a future workstream nests the two.
        """
        if not isinstance(t, torch.Tensor) or not t.is_cuda:
            return t

        mgr = self._chunk_manager
        if mgr is None:
            # Defensive: forward checked above, but a stray callback
            # could plausibly fire after detach_runtime. Pass through.
            return t

        # Storage identity is what autograd actually saved — looking
        # up by `data_ptr()` matches the pool-buffer storage exactly
        # because every chunk param is a `view` of the chunk's flat
        # uint8 buffer (see ChunkManager._rebind_params_to_buffer).
        #
        # The chunk-storage lookup is the SOLE gate — there is no
        # size-threshold check above. A small chunk-managed param view
        # (e.g. a bias or LayerNorm weight below the legacy 1 MiB
        # threshold) still aliases the chunk's GPU storage; if we
        # passed it through on size, autograd's saved-tensor table
        # would retain a strong reference to that view, pinning the
        # chunk buffer past post_block_forward's offload — defeating
        # OFFLOAD on any chunk that contains a small param. Non-chunk
        # tensors (activations, params from non-managed modules) are
        # passed through unconditionally below.
        try:
            ptr = t.untyped_storage().data_ptr()
        except Exception:  # noqa: BLE001 — defensive against aten edge cases
            return t

        chunk_id = mgr.chunk_id_for_storage_ptr(ptr)
        if chunk_id is None:
            # Not a chunk-managed param view (likely a forward
            # activation produced inside this block). Passthrough —
            # pure activations are SWAP's domain, not ours.
            return t

        # Storage offset in BYTES from the start of the chunk's
        # storage. ``t.storage_offset()`` returns ELEMENTS of the
        # tensor's dtype, so multiply by element_size to get bytes —
        # matching how the chunk lays out per-param byte slots.
        storage_offset = int(t.storage_offset()) * int(t.element_size())

        # Drop the strong reference to ``t`` by returning the metadata
        # handle. Autograd's saved-tensor table now holds only the
        # handle — the underlying GPU storage becomes collectible the
        # moment the scheduler issues offload(chunk_id) post-forward.
        return _ParamHandle(
            chunk_id=chunk_id,
            storage_offset=storage_offset,
            shape=t.shape,
            stride=tuple(int(s) for s in t.stride()),
            dtype=t.dtype,
            requires_grad=t.requires_grad,
        )

    def _unpack(self, handle: Any) -> torch.Tensor:
        """Re-gather the chunk and reconstruct the saved view.

        Three cases:

        1. ``handle`` is a ``_ParamHandle`` — the hot path. Call
           ``gather_for_backward`` to materialize the chunk on GPU
           (idempotent; fast-path when already resident), look up the
           pool buffer, slice + dtype-view at the recorded byte
           offset/shape, attach the BackwardHandle to the view's
           lifetime via a private attribute, return the view.
        2. ``handle`` is a ``torch.Tensor`` — the passthrough case
           from ``_pack``. Return as-is.
        3. ``handle`` is anything else (e.g. None for retained_grad
           sentinels, or a future SWAP-style ``_CPUHandle``). Defer
           to whatever the outer hook context (or default save/load)
           does with it — return as-is.

        The unpack hook must NOT touch ``param.data`` directly —
        ``param.data`` may be on CPU mid-CPU-Adam-step (see §6.4 of
        the design doc). It returns a view to autograd; gradient
        kernels read the view, NOT ``param.data``. The chunk's slot
        stays alive across this backward via the ``BackwardHandle``
        refcount, NOT via ``param.data``.
        """
        if not isinstance(handle, _ParamHandle):
            # Pure-activation / unknown handle types pass through.
            # ``handle`` here is whatever the next outer hook (or
            # default save) produced — typically ``handle`` IS the
            # original tensor.
            return handle  # type: ignore[no-any-return]

        mgr = self._chunk_manager
        if mgr is None:
            # Should not happen: we got a _ParamHandle, which means
            # _pack ran with a manager attached. If we somehow lose
            # the manager between forward and backward, raise loudly.
            raise RuntimeError(
                "OffloadedBlock._unpack received a _ParamHandle but the "
                "chunk manager has been detached; backward cannot proceed."
            )

        # Gather the chunk (idempotent if resident) and bump the
        # backward refcount. ``BackwardHandle`` owns the decrement on
        # its __del__ — we attach it to the view below so the autograd
        # engine's reference to the view keeps the handle alive, and
        # the handle's release timing follows the engine's release of
        # the unpacked tensor.
        backward_handle = mgr.gather_for_backward(handle.chunk_id)

        # Explicit runtime check, NOT an ``assert``: ``python -O`` strips
        # asserts, and silently dereferencing a ``None`` buffer_pool
        # below would raise an obscure ``AttributeError`` instead of
        # this descriptive failure. Release the just-bumped backward
        # refcount so we don't leak handle state into the manager.
        if mgr.buffer_pool is None:
            backward_handle.release()
            raise RuntimeError(
                "OffloadedBlock._unpack: chunk manager has no buffer_pool — "
                "cannot reconstruct the saved view. This indicates the "
                "OFFLOAD path was reached on an all-persistent layout, "
                "which the admissibility filter should have rejected."
            )
        buf = mgr.buffer_pool.lookup_resident(handle.chunk_id)
        if buf is None:
            # Defensive: gather_for_backward should have made the chunk
            # resident. If not, an intervening evict-then-deferred-offload
            # raced us; we re-gather synchronously.
            mgr.gather(handle.chunk_id)
            buf = mgr.buffer_pool.lookup_resident(handle.chunk_id)
            if buf is None:
                # Release the refcount we just bumped so we don't leak
                # a handle into the manager state on failure.
                backward_handle.release()
                raise RuntimeError(
                    f"OffloadedBlock._unpack: chunk {int(handle.chunk_id)} "
                    "is not resident after gather_for_backward — pool "
                    "may have been evicted by an unbalanced acquire."
                )

        # Reconstruct the view at the recorded byte offset/shape via
        # ``as_strided`` on a typed view of the chunk's storage. The
        # storage-typed-empty + ``as_strided`` path (rather than
        # ``buf.narrow().view(dtype).view(shape)``) is load-bearing for
        # autograd correctness: with the latter chain, autograd's
        # backward kernels through the unpacked tensor produce wrong
        # gradients on upstream parameters (verified empirically on
        # Linear-block backward — embed.weight grad diverges by ~2x
        # while h.weight grad is correct). The exact failure mode is
        # an autograd metadata mismatch buried in the dtype-changing
        # ``view(dtype)`` step; ``as_strided`` skips that step by
        # walking storage in the param's dtype directly.
        storage = buf.untyped_storage()
        typed = torch.empty(0, dtype=handle.dtype, device=buf.device).set_(  # type: ignore[call-overload]
            storage
        )
        # storage_offset is bytes; as_strided wants ELEMENTS of dtype.
        elem_size = int(handle.dtype.itemsize)
        if handle.storage_offset % elem_size != 0:
            backward_handle.release()
            raise RuntimeError(
                f"OffloadedBlock._unpack: chunk {int(handle.chunk_id)} "
                f"storage_offset {handle.storage_offset} is not aligned "
                f"to dtype {handle.dtype} element size {elem_size}; the "
                "chunk layout's per-param alignment pass should have "
                "prevented this."
            )
        elem_offset = handle.storage_offset // elem_size
        # Use the saved stride directly — pack captured the original
        # tensor's stride at save time, which may not match a row-
        # major contiguous layout (e.g. ``F.linear`` saves ``weight``
        # with ``stride=(1, in_dim)`` because the matmul wants the
        # transposed view). Reconstructing with a guessed contiguous
        # stride would read the storage in the wrong element order —
        # silent gradient corruption on consumers of the saved view.
        shape_t = tuple(int(s) for s in handle.shape)
        view = typed.as_strided(shape_t, handle.stride, elem_offset)

        if handle.requires_grad:
            view.requires_grad_(True)

        # Pin the BackwardHandle to the view's lifetime via a private
        # attribute. The autograd engine holds ``view`` until the
        # consuming Node's apply() returns; once it drops the
        # reference, ``view`` is GC'd, the attribute is dropped, and
        # the BackwardHandle's __del__ decrements the manager's
        # refcount (potentially draining a deferred offload).
        #
        # A weakref-keyed dict would also work, but it would require
        # a finalizer + global state. The private attribute is
        # local to the view, costs one Python attribute set, and is
        # cleaned up by the standard Python ref-counting path.
        view._protrain_backward_handle = backward_handle  # type: ignore[attr-defined]
        return view

    def extra_repr(self) -> str:
        """Return the wrapper's mode tag for ``print(model)``."""
        return f"mode={self._protrain_wrapped_mode.value}"


__all__ = ["OffloadedBlock", "_ParamHandle"]
