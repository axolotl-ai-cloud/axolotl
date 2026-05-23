"""Param-offload-aware block wrapper: saved_tensors_hooks replace param-view saves with metadata handles."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from axolotl.integrations.protrain.block.strategy import BlockMode
from axolotl.utils.logging import get_logger

try:
    from torch.compiler import disable as _compile_disable
except Exception:  # noqa: BLE001 — older torches lack torch.compiler.disable

    def _compile_disable(fn=None, *, recursive=True):  # noqa: ARG001
        return fn if fn is not None else (lambda f: f)


if TYPE_CHECKING:
    from axolotl.integrations.protrain.chunk.manager import ChunkManager
    from axolotl.integrations.protrain.types import ChunkId

LOG = get_logger(__name__)


# Test-override compat shim; OffloadedBlock._pack gates on chunk-storage lookup, not size.
SIZE_THRESHOLD_BYTES: int = 1 << 20  # 1 MiB


@dataclass(slots=True, frozen=True)
class _ParamHandle:
    """Metadata handle (cid, byte-offset, shape, stride) replacing autograd's strong tensor ref."""

    chunk_id: "ChunkId"
    storage_offset: int  # byte offset within the chunk's storage
    shape: torch.Size
    stride: tuple[int, ...]  # in elements of dtype
    dtype: torch.dtype
    requires_grad: bool
    # Monotonic attach-epoch token; id(mgr) recycles after GC and would let stale handles pass the guard.
    runtime_id: int


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

    #: Process-wide monotonic counter handing out attach-epoch tokens.
    #: Each ``attach_runtime()`` call draws a fresh token; the token is
    #: stamped into every ``_ParamHandle`` produced by ``_pack`` and
    #: cross-checked by ``_unpack`` to detect detach + re-attach
    #: between forward and backward. Unlike ``id(chunk_manager)``, the
    #: token is never recycled, so a stale handle cannot collide with
    #: a freshly-allocated manager that happens to land at the prior
    #: manager's address. ``itertools.count()`` advances atomically
    #: under the GIL on a single ``next()`` call, so concurrent
    #: ``attach_runtime`` calls on distinct wrappers always observe
    #: distinct tokens.
    _next_attach_token = itertools.count(1)

    def __init__(self, block: nn.Module) -> None:
        """Wrap ``block`` in identity-mode; runtime wired by :meth:`attach_runtime`."""
        super().__init__()
        self.block = block
        self._protrain_wrapped_mode: BlockMode = BlockMode.OFFLOAD
        self._chunk_manager: "ChunkManager | None" = None
        self._scheduler: Any = None  # M3 owns the scheduler interface contract
        self._warned_no_runtime = False
        #: Monotonic attach-epoch token of the currently-attached chunk
        #: manager, or ``None`` when detached. Stamped into every
        #: ``_ParamHandle`` produced by ``_pack`` and cross-checked by
        #: ``_unpack`` to detect a detach + re-attach-with-a-different-
        #: manager between forward and backward (the in-flight
        #: ``attach_runtime`` swap is rejected outright; this guards
        #: the detach-then-re-attach variant where the in-flight check
        #: no longer fires). See ``_next_attach_token`` for why we
        #: prefer a monotonic counter over ``id(mgr)``.
        self._runtime_id: int | None = None

    def attach_runtime(
        self,
        chunk_manager: "ChunkManager",
        scheduler: Any = None,
    ) -> None:
        """Wire chunk_manager + scheduler. Idempotent; same-mgr re-attach OK, mgr swap rejected."""
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
        # Fresh token only on genuine attach; same-mgr re-attach must not invalidate in-flight handles.
        if self._chunk_manager is None:
            self._runtime_id = next(OffloadedBlock._next_attach_token)
        self._chunk_manager = chunk_manager
        self._scheduler = scheduler

    def detach_runtime(self) -> None:
        """Drop the manager reference — wrapper degrades to identity."""
        self._chunk_manager = None
        self._scheduler = None
        self._runtime_id = None

    @_compile_disable(recursive=True)
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run the wrapped block under saved_tensors_hooks that record param handles."""
        mgr = self._chunk_manager

        # Cold path — no runtime attached; saved tensors live on GPU normally.
        if mgr is None:
            if not self._warned_no_runtime:
                LOG.warning(
                    "OffloadedBlock forward without attached runtime — "
                    "degrading to identity. Call attach_runtime(chunk_manager) "
                    "after constructing the block."
                )
                self._warned_no_runtime = True
            return self.block(*args, **kwargs)

        # Hot path — install saved_tensors_hooks for the duration of the block.
        with torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack):
            return self.block(*args, **kwargs)

    # ---- saved-tensors hooks ----------------------------------------------

    def _pack(self, t: torch.Tensor) -> Any:
        """Record _ParamHandle for chunk-managed param views; passthrough for pure activations."""
        if not isinstance(t, torch.Tensor) or not t.is_cuda:
            return t

        mgr = self._chunk_manager
        if mgr is None:
            return t

        # data_ptr lookup is the sole gate; size-threshold would defeat OFFLOAD for small params.
        try:
            ptr = t.untyped_storage().data_ptr()
        except Exception:  # noqa: BLE001 — defensive against aten edge cases
            return t

        chunk_id = mgr.chunk_id_for_storage_ptr(ptr)
        if chunk_id is None:
            return t

        # Persistent chunks: keep the strong ref; offload round-trip would be wasted.
        if chunk_id in mgr._persistent_ids:  # noqa: SLF001
            return t

        # storage_offset() returns elements; * element_size for bytes.
        storage_offset = int(t.storage_offset()) * int(t.element_size())

        return _ParamHandle(
            chunk_id=chunk_id,
            storage_offset=storage_offset,
            shape=t.shape,
            stride=tuple(int(s) for s in t.stride()),
            dtype=t.dtype,
            requires_grad=t.requires_grad,
            runtime_id=self._runtime_id,  # type: ignore[arg-type]
        )

    def _unpack(self, handle: Any) -> torch.Tensor:
        """Re-gather chunk + reconstruct saved view (passthrough for non-_ParamHandle)."""
        if not isinstance(handle, _ParamHandle):
            return handle  # type: ignore[no-any-return]

        mgr = self._chunk_manager
        if mgr is None:
            raise RuntimeError(
                "OffloadedBlock._unpack received a _ParamHandle but the "
                "chunk manager has been detached; backward cannot proceed."
            )

        # Runtime-identity guard: reject stale handles from a detach+re-attach cycle.
        if handle.runtime_id != self._runtime_id:
            raise RuntimeError(
                "OffloadedBlock._unpack: saved _ParamHandle was produced "
                "against a different chunk manager than the currently-"
                "attached one. The wrapper was detach_runtime()'d and "
                "re-attached with a new manager between forward and "
                "backward; resolving this handle's ChunkId against the "
                "new manager's storage map would decode against unrelated "
                "storage. detach/re-attach cycles are only safe when no "
                "saved-tensor handles are outstanding."
            )

        # Bump backward refcount; BackwardHandle.__del__ decrements via the view's lifetime.
        backward_handle = mgr.gather_for_backward(handle.chunk_id)

        # Any failure path before the final view binding must release backward_handle.
        released = False
        try:
            # Explicit check (asserts would strip under python -O).
            if mgr.buffer_pool is None:
                raise RuntimeError(
                    "OffloadedBlock._unpack: chunk manager has no buffer_pool — "
                    "cannot reconstruct the saved view. This indicates the "
                    "OFFLOAD path was reached on an all-persistent layout, "
                    "which the admissibility filter should have rejected."
                )
            buf = mgr.buffer_pool.lookup_resident(handle.chunk_id)
            if buf is None:
                # Defensive: an evict-then-deferred-offload race; re-gather sync.
                mgr.gather(handle.chunk_id)
                buf = mgr.buffer_pool.lookup_resident(handle.chunk_id)
                if buf is None:
                    raise RuntimeError(
                        f"OffloadedBlock._unpack: chunk {int(handle.chunk_id)} "
                        "is not resident after gather_for_backward — pool "
                        "may have been evicted by an unbalanced acquire."
                    )

            # as_strided on a typed storage view; narrow().view(dtype) breaks autograd correctness.
            storage = buf.untyped_storage()
            typed = torch.empty(0, dtype=handle.dtype, device=buf.device).set_(  # type: ignore[call-overload]
                storage
            )
            # storage_offset is bytes; as_strided wants ELEMENTS of dtype.
            elem_size = int(handle.dtype.itemsize)
            if handle.storage_offset % elem_size != 0:
                raise RuntimeError(
                    f"OffloadedBlock._unpack: chunk {int(handle.chunk_id)} "
                    f"storage_offset {handle.storage_offset} is not aligned "
                    f"to dtype {handle.dtype} element size {elem_size}; the "
                    "chunk layout's per-param alignment pass should have "
                    "prevented this."
                )
            elem_offset = handle.storage_offset // elem_size
            # Use saved stride; F.linear saves weight with non-row-major stride.
            shape_t = tuple(int(s) for s in handle.shape)
            view = typed.as_strided(shape_t, handle.stride, elem_offset)

            if handle.requires_grad:
                view.requires_grad_(True)

            # Pin BackwardHandle to view's lifetime; GC of view drops the handle's __del__.
            view._protrain_backward_handle = backward_handle  # type: ignore[attr-defined]
            released = True  # ownership transferred to the view
            return view
        finally:
            if not released:
                # Any exception path releases the bumped refcount.
                backward_handle.release()

    def extra_repr(self) -> str:
        """Return the wrapper's mode tag for ``print(model)``."""
        return f"mode={self._protrain_wrapped_mode.value}"


__all__ = ["OffloadedBlock", "_ParamHandle"]
