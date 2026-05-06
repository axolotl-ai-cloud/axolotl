"""Pinned-RAM activation pool for the SWAP block path (§3.1.2).

The SWAP wrapper offloads each forward block's output activation to
pinned host memory, then prefetches it back during backward. To make
the D2H copy non-blocking and to give PyTorch a stable pointer to copy
into, we pre-allocate one large pinned host region and hand out fixed-
size slots from it.

This pool is independent of the chunk-buffer pool: the chunk pool
holds parameter slabs (sized to ``S_chunk``), the activation pool
holds activations (sized to ``max_activation_bytes`` per slot). The
two pools never share a slot and are sized independently from the
searcher's decision (``n_swap`` and ``prefetch_depth``).

Lifecycle
---------
Constructed by ``protrain_model_wrapper`` once it knows
``result.cfg.n_swap > 0``. A single :class:`PinnedHostMemory` backs
the entire pool; slots are uint8 narrow views into that region.
Tensors are hashed into slots via :meth:`acquire`; the consumer must
call :meth:`release` (typically inside autograd backward) to return
the slot to the free list. The pool is closed at scheduler tear-down
or ``WrappedModel`` GC, releasing the pinned region.

Sizing
------
``slot_bytes`` is the worst-case activation bytes for a *single* saved
tensor inside any SWAP block (the maximum across the searcher's chosen
swap-band of blocks). ``n_slot`` is ``n_swap * slots_per_block *
prefetch_depth`` where ``slots_per_block`` (K) is the number of saved
tensors a single block forward can produce — typically the residual
stream + Q/K/V/scores + FFN intermediates ≈ 6–8 tensors. K=8 is the
default; the model wrapper may bump it for unusual block shapes. For
the M5+ ``saved_tensors_hooks`` integration each saved tensor inside
a block forward needs its own slot, so K cannot be 1 anymore.
``prefetch_depth = 2`` keeps single-block lookahead during backward.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from axolotl.integrations.protrain.chunk.pinned_alloc import PinnedHostMemory
from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    import torch

LOG = get_logger(__name__)


#: Default number of saved tensors per block. Transformer blocks
#: typically save residual + Q/K/V/scores + 2-3 FFN intermediates ≈ 6-8.
#: Bumped to 8 to cover unusual shapes (gated FFN, MoE) without
#: exhausting the pool. Tunable via ``ActivationSwapPool(slots_per_block=...)``.
DEFAULT_SLOTS_PER_BLOCK: int = 8

#: Default seconds ``close()`` will wait for outstanding ``acquire()`` borrows
#: to be returned via ``release()`` before giving up. Picked large enough that
#: a normal backward sweep on the swap stream completes; tuned-down values are
#: useful for unit tests. Override per-call via ``close(drain_timeout=...)``.
DEFAULT_CLOSE_DRAIN_TIMEOUT_S: float = 30.0


class ActivationSwapPool:
    """Fixed-size pinned-host slot pool for SWAP-block activations.

    Parameters
    ----------
    n_swap:
        Number of SWAP blocks the searcher selected. Must be ``>= 1``;
        callers should not construct a pool when ``n_swap == 0``.
    slot_bytes:
        Worst-case bytes for a single saved tensor inside any SWAP
        block. The pool sizes every slot to exactly this value so any
        saved tensor fits any slot.
    prefetch_depth:
        How many copies-per-block to keep in flight during backward.
        ``2`` is single-block lookahead (one block's saved tensors
        currently resident on CPU, one being H2D-fetched for the next
        backward step). ``1`` collapses to fully-serial SWAP — only
        useful for unit tests.
    slots_per_block:
        How many saved tensors per block-forward call to budget for.
        Default is :data:`DEFAULT_SLOTS_PER_BLOCK` (8). Total slots =
        ``n_swap * slots_per_block * prefetch_depth``.

    Bounds
    ------
    Max in-flight slots = ``n_swap * slots_per_block * prefetch_depth``.
    Total pinned host bytes = ``n_slot * slot_bytes``. Both terms scale
    linearly with K (slots_per_block); setting K too high wastes
    pinned RAM, setting it too low triggers ``RuntimeError("exhausted")``
    inside the swap pack hook (which the wrapper degrades to "keep on
    GPU" — correct but defeats the memory savings).

    Notes
    -----
    The pool is **stream-agnostic** — copies onto/from slots happen on
    the SWAP wrapper's chosen stream (typically the scheduler's
    ``_swap_stream``). Slot ownership is tracked by Python-side ID
    only; CUDA never sees the pool's free-list state. Callers MUST
    synchronize the swap stream with their consumer before
    ``release`` reuses the slot for a fresh acquire — otherwise the
    in-flight D2H/H2D may race against the next acquire's writes.
    """

    def __init__(
        self,
        n_swap: int,
        slot_bytes: int,
        prefetch_depth: int = 2,
        slots_per_block: int = DEFAULT_SLOTS_PER_BLOCK,
    ) -> None:
        """Allocate the backing pinned region and the free-slot LIFO."""
        if n_swap < 1:
            raise ValueError(f"n_swap must be >= 1, got {n_swap}")
        if slot_bytes <= 0:
            raise ValueError(f"slot_bytes must be positive, got {slot_bytes}")
        if prefetch_depth < 1:
            raise ValueError(f"prefetch_depth must be >= 1, got {prefetch_depth}")
        if slots_per_block < 1:
            raise ValueError(f"slots_per_block must be >= 1, got {slots_per_block}")

        self.n_swap = int(n_swap)
        self.slot_bytes = int(slot_bytes)
        self.prefetch_depth = int(prefetch_depth)
        self.slots_per_block = int(slots_per_block)
        self.n_slot = self.n_swap * self.slots_per_block * self.prefetch_depth

        # Backing pinned-host region (split into ``n_slot`` equal slots).
        self._pinned = PinnedHostMemory(n_buffer=self.n_slot, S_chunk=self.slot_bytes)
        self._closed = False
        # Set as soon as ``close()`` begins teardown so concurrent
        # ``acquire``/``release`` callers stop racing the (lock-free)
        # ``_pinned.close()`` window. Without this, a caller could pop
        # a slot, increment ``_inflight``, then fail in ``buffer()``
        # with "PinnedHostMemory is closed" while the pool's free-list
        # accounting is left corrupted.
        self._closing = False
        # Free-list of available slot indices. We use a plain list as a
        # LIFO stack — locality of reuse is irrelevant for pinned host
        # memory (no allocator state to amortize), and a list is
        # cheaper than a deque for the small N_slot we work with
        # (typically <= 16).
        self._free: list[int] = list(range(self.n_slot))
        self._inflight: int = 0
        # Bookkeeping lock. The SWAP wrapper's pack/unpack hooks fire
        # from autograd's worker threads on the swap stream while the
        # main stream calls ``acquire``/``release`` from the forward
        # path; without a lock the ``_free`` list and ``_inflight``
        # counter can race. A plain ``Lock`` (not ``RLock``) suffices
        # because none of the locked sections call back into another
        # locked method on this pool.
        self._lock = threading.Lock()

        LOG.debug(
            "ActivationSwapPool: n_swap=%d slot_bytes=%d prefetch_depth=%d "
            "slots_per_block=%d n_slot=%d total_bytes=%d precise=%s",
            self.n_swap,
            self.slot_bytes,
            self.prefetch_depth,
            self.slots_per_block,
            self.n_slot,
            self.n_slot * self.slot_bytes,
            self._pinned.is_precise_size,
        )

    def acquire(self) -> tuple[int, "torch.Tensor"]:
        """Reserve a slot; return ``(slot_id, pinned_uint8_view)``.

        The returned tensor is a 1-D ``uint8`` view of length
        ``slot_bytes`` over the pinned region. Callers reshape it to
        their target dtype with ``.view(dtype).reshape(shape)`` after
        copying via ``.copy_(src, non_blocking=True)`` on the swap stream.
        """
        with self._lock:
            if self._closed or self._closing:
                raise RuntimeError("ActivationSwapPool is closed")
            if not self._free:
                raise RuntimeError(
                    f"ActivationSwapPool exhausted (n_slot={self.n_slot}, "
                    f"in-flight={self._inflight}); increase prefetch_depth or "
                    "verify the SWAP wrapper releases slots after backward."
                )
            slot_id = self._free.pop()
            self._inflight += 1
            # ``PinnedHostMemory.buffer()`` mutates ``_live_borrows`` and
            # explicitly requires caller synchronization. Hold ``self._lock``
            # across it so concurrent acquire/release/close() callers cannot
            # race on the borrow accounting (which would either drift the
            # count or free the pinned region while a slot view is still live).
            view = self._pinned.buffer(slot_id)
        return slot_id, view

    def release(self, slot_id: int) -> None:
        """Return ``slot_id`` to the free list. Idempotent on bad ids.

        The caller is responsible for ensuring no in-flight CUDA
        operation references this slot before calling — the pool does
        NOT issue stream syncs.
        """
        with self._lock:
            # Allow ``release()`` to proceed during ``_closing`` so late
            # releases can retire outstanding borrows; otherwise
            # ``self._pinned.close()`` blocks on still-live borrows.
            # ``acquire()`` is the one that should reject new work when
            # ``_closing`` is set.
            if self._closed:
                return
            if not 0 <= slot_id < self.n_slot:
                LOG.warning(
                    "ActivationSwapPool.release: slot_id %d out of range [0, %d); ignored",
                    slot_id,
                    self.n_slot,
                )
                return
            if slot_id in self._free:
                # Defensive: double-release. Log loudly because this likely
                # signals a swap-wrapper bug (e.g. backward executed twice
                # because of a retain_graph=True replay).
                LOG.warning(
                    "ActivationSwapPool.release: slot %d already free; double-release",
                    slot_id,
                )
                return
            self._free.append(slot_id)
            self._inflight -= 1
            # Return the borrow to the underlying pinned allocator so its
            # close() guard knows the slot view is no longer live. The view
            # itself is dropped by the caller; ``record_stream`` keeps the
            # bytes alive for the in-flight H2D, but the borrow accounting
            # is mutated by ``release_buffer`` and per ``PinnedHostMemory``'s
            # contract requires caller synchronization — so we hold
            # ``self._lock`` across it to keep ``_live_borrows`` consistent
            # with our slot lifetime under concurrent acquire/release/close().
            self._pinned.release_buffer(slot_id)

    @property
    def total_bytes(self) -> int:
        """Total pinned-host bytes held by the pool."""
        return self.n_slot * self.slot_bytes

    @property
    def free_count(self) -> int:
        with self._lock:
            return len(self._free)

    @property
    def inflight_count(self) -> int:
        with self._lock:
            return self._inflight

    def close(
        self,
        drain_timeout: float = DEFAULT_CLOSE_DRAIN_TIMEOUT_S,
        poll_interval: float = 0.01,
    ) -> None:
        """Free the pinned region. Idempotent.

        Three-phase teardown:

        1. Under ``_lock``, flip ``_closing = True`` and drop the lock.
           From this point, ``acquire()`` raises so no new borrow can
           sneak into the drain window. ``release()`` continues to
           accept returns under ``_closing`` so outstanding borrows can
           retire.
        2. Wait (without holding ``_lock``) for ``_inflight == 0`` so
           the pinned region has no live slot views before we free it.
           If the drain does not complete inside ``drain_timeout``
           seconds we *unset* ``_closing`` and raise ``RuntimeError`` —
           leaving the pool retryable so the caller can resync the swap
           stream / retire stragglers and try ``close()`` again.
        3. Call ``_pinned.close()`` WITHOUT holding ``self._lock`` — it
           is on a separate lock-domain (its own bookkeeping, not part
           of this pool's free-list/inflight invariants), it may be
           slow, and dropping the lock keeps concurrent ``free_count`` /
           ``inflight_count`` reads responsive during teardown. If
           ``_pinned.close()`` itself raises (e.g. a late borrow snuck
           in despite the drain), we likewise unset ``_closing`` so the
           caller can retry rather than be stuck with a permanently
           half-closed pool that silently no-ops future ``close()``s.
        4. Re-acquire ``_lock`` and flip ``_closed = True``, clearing
           the free-list / inflight counter.

        Parameters
        ----------
        drain_timeout:
            Seconds to wait for ``_inflight`` to drain before raising.
            Defaults to :data:`DEFAULT_CLOSE_DRAIN_TIMEOUT_S`. Tests
            may pass a small value (e.g. ``0.05``) to exercise the
            timeout path quickly.
        poll_interval:
            Sleep interval (seconds) between drain-counter checks.
        """
        with self._lock:
            if self._closed:
                return
            if self._closing:
                # A concurrent ``close()`` is mid-drain. Surface the race
                # rather than silently no-op'ing — the caller can decide
                # whether to retry, await the in-flight close via its own
                # synchronization, or treat it as an error.
                raise RuntimeError(
                    "ActivationSwapPool.close: close in progress on another thread"
                )
            # Block new acquires BEFORE we drop the lock for the drain
            # wait + (potentially slow) ``_pinned.close()`` call.
            self._closing = True
        try:
            # Wait for outstanding ``acquire()`` borrows to be retired
            # via ``release()``. ``release()`` deliberately keeps
            # decrementing ``_inflight`` while ``_closing`` is True so
            # the counter can converge to zero here.
            deadline = time.monotonic() + max(0.0, float(drain_timeout))
            while True:
                with self._lock:
                    inflight = self._inflight
                if inflight == 0:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"ActivationSwapPool.close: timed out after "
                        f"{drain_timeout:.3f}s waiting for {inflight} "
                        "in-flight slot(s) to drain. Caller is missing "
                        "release() pairs or the swap stream has not "
                        "synchronized — retry close() after stragglers "
                        "retire."
                    )
                time.sleep(max(0.0, float(poll_interval)))
            # Drain complete — now free the pinned region. If this still
            # raises (e.g. a late borrow leaked through some other path),
            # we treat ``close()`` as failed-and-retryable rather than
            # silently no-op'ing on subsequent attempts.
            self._pinned.close()
        except BaseException:
            # Failed close: roll back ``_closing`` so callers can retry
            # after fixing the underlying cause. ``acquire()`` re-opens
            # only because ``_closed`` is still False; the pinned region
            # has not been freed.
            with self._lock:
                self._closing = False
            raise
        with self._lock:
            self._closed = True
            self._free.clear()
            self._inflight = 0

    def __del__(self) -> None:  # noqa: D401
        try:
            # Best-effort non-blocking cleanup. ``__del__`` runs during GC
            # / interpreter teardown where blocking on a 30s drain would
            # stall shutdown; explicit ``close()`` callers still get the
            # full ``DEFAULT_CLOSE_DRAIN_TIMEOUT_S`` default.
            self.close(drain_timeout=0)
        except Exception:  # noqa: BLE001 — destructor must not throw
            pass


__all__ = ["ActivationSwapPool", "DEFAULT_SLOTS_PER_BLOCK"]
