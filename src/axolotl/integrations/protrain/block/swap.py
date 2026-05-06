"""Activation-swap wrapper (§3.1.2 — paper-real implementation, M5+).

SWAP mode in the ProTrain three-way block strategy: forward activations
are offloaded to pinned CPU memory, then prefetched back during
backward. The wrapper installs a
:func:`torch.autograd.graph.saved_tensors_hooks` context around the
block's forward so **every** saved tensor (residuals, attention QKV/
scores, FFN intermediates) is D2H'd to a pinned CPU pool and H2D'd
back on backward — not just the block's output tensor.

This is the M5+ upgrade over option-2A. Option-2A only swapped the
block's output tensor via a custom autograd Function; the GPU
activation stayed pinned by autograd because ``ctx.save_for_backward``
keeps a CUDA reference. With ``saved_tensors_hooks`` the saved-tensor
references handed to autograd are CPU-only handles, so the GPU storage
is reclaimed when the local Python frame drops its last GPU reference
to the activation. The result: actual GPU memory is freed between
forward and backward, not just shuffled.

Stream policy
-------------
Both D2H and H2D copies run on the scheduler's ``_swap_stream`` (one
shared stream per scheduler). The compute stream waits on the swap
stream's H2D event before the upstream backward kernel reads the
re-materialised activation. In forward the swap stream waits on the
compute stream before reading the GPU tensor we are offloading.

Hot path / cold path
--------------------
The pool + stream are injected post-construction by the model wrapper
via :meth:`SwappedBlock.attach_runtime`. If a block is constructed
WITHOUT runtime attached (e.g. unit tests, or a model wrapper that
forgot to call attach_runtime when ``n_swap > 0``), the wrapper
degrades to a no-op identity hook in autograd: the activations live on
GPU as they normally would, and no D2H/H2D happens. This keeps
correctness intact while preserving the historical "constructible
without runtime" surface that test fixtures rely on. A WARNING is
logged once per instance so the configuration drift is visible.

Tunable: ``SIZE_THRESHOLD_BYTES``
---------------------------------
Saved tensors smaller than this byte threshold pass through as-is
(kept on GPU). Small tensors don't recover much memory and the
pinned-slot bookkeeping + PCIe round trip cost dominates. The default
1 MiB is chosen to cover scalar-ish saved tensors (LayerNorm gamma/
beta, softmax masks, attention biases) while still capturing the big
ones (residual stream ``(batch, seq, hidden)`` and attention scores
``(batch, heads, seq, seq)``). Override per-test via the constant.

Per-Node fanout floor (single-block backward peak)
--------------------------------------------------
The headline 43-66% memory reduction comes from compounding across
stacked SWAP blocks: while block ``i`` runs backward, blocks
``i+1, …, n-1`` are still done with their saved tensors on CPU.
A *single* block's backward peak only drops ~10-15% — investigated
2026-05-01 with a register_hook-based early-free prototype that
showed no measurable improvement over the natural ``__del__`` path.

The bound is an autograd-engine internal:

    For each backward Node, the C++ engine calls
    ``SavedVariable::unpack()`` for ALL the Node's saved tensors
    BEFORE invoking the Node's ``apply()``. The unpacked tensors
    are held as locals in the C++ derivative function and released
    only when ``apply()`` returns. Multiple saved tensors per Node
    therefore yield concurrent live unpacked GPU buffers during
    that single Node's backward call.

For a transformer block, the dominant fanout is the attention
score-times-V matmul (saves both ``attn`` and ``v``) and the
QKV-projection linear (saves activation and weight). With B=16
S=256 D=512 fp32 the maximum concurrent unpacked bytes is ~42 MB —
that's the bound on how much we can shrink the per-block backward
peak without intervening mid-apply. No Python hook
(``saved_tensors_hooks``, ``Node.register_hook``,
``Node.register_prehook``) fires inside an ``apply()``.

Two paths could push past the floor — both deemed out of scope:

* Replace each matmul/softmax/etc. with an autograd Function that
  stages saved-tensor lifetimes manually. Breaks model-agnosticism;
  would have to wrap every op in every block.
* Modify PyTorch C++ engine to release individual saved tensors
  after each derivative step. Upstream change.

The single-block floor is recorded by
``test_swap_single_block_backward_peak_at_autograd_floor`` so
future maintainers don't re-run the investigation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from axolotl.integrations.protrain.block.strategy import BlockMode
from axolotl.integrations.protrain.runtime.streams import SingleStreamAllocator
from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    from axolotl.integrations.protrain.block.swap_pool import ActivationSwapPool

LOG = get_logger(__name__)


#: Saved tensors smaller than this many bytes are kept on GPU (not
#: swapped). 1 MiB is the default; tests may override by reassigning
#: this module attribute. See the module docstring for derivation.
SIZE_THRESHOLD_BYTES: int = 1 << 20  # 1 MiB


#: Safety margin reserved on top of the swap-in's own allocation when
#: gating in :func:`unpack_from_pool`. Backward kernels enqueued *after*
#: the headroom check may also allocate transients (workspace buffers,
#: intermediate gradients) before the swap-in's ``empty_strided`` runs;
#: 64 MiB is generous enough to absorb typical transformer-block
#: backward transients without cutting into the headroom that motivated
#: the gate. Tunable: lower if profiling shows the gate is over-firing,
#: raise if backward still OOMs after the gate's host-side drain.
_SWAP_HEADROOM_SAFETY_BYTES: int = 64 * 1024 * 1024


#: Number of synchronize-and-recheck retries before the SWAP gate
#: raises. Each retry calls ``torch.cuda.synchronize()`` to drain
#: in-flight backward kernels (which release their saved-tensor
#: storage). Three retries is empirically sufficient for typical
#: transformer backward shapes; raise if your workload chains many
#: small SWAP blocks tightly.
_SWAP_MAX_DRAIN_RETRIES: int = 3


def _swap_stream_wait_compute(
    device: "torch.device", swap_stream: "torch.cuda.Stream"
) -> None:
    """Make ``swap_stream`` wait on the compute stream of ``device``.

    The device argument is load-bearing: ``torch.cuda.current_stream()``
    without a device follows the *ambient* current device, which under
    multi-GPU / model-parallel runs can synchronize against a different
    GPU's compute stream and race the D2H/H2D copy on this tensor's
    real device.
    """
    if swap_stream is None or not torch.cuda.is_available():
        return
    swap_stream.wait_stream(torch.cuda.current_stream(device=device))


def _compute_stream_wait_swap(
    device: "torch.device", swap_stream: "torch.cuda.Stream"
) -> None:
    """Make the compute stream of ``device`` wait on ``swap_stream``.

    See ``_swap_stream_wait_compute`` for why the device argument is
    passed explicitly rather than relying on the ambient current device.
    """
    if swap_stream is None or not torch.cuda.is_available():
        return
    torch.cuda.current_stream(device=device).wait_stream(swap_stream)


@dataclass
class _CPUHandle:
    """CPU-resident handle returned by ``pack_to_pool``.

    Holds the pool slot id + the metadata needed to reconstruct the
    GPU tensor in ``unpack_from_pool``. Because the handle does NOT
    reference the GPU tensor, autograd's saved-tensor table no longer
    pins GPU storage — that is the whole point of the M5+ rewrite.
    """

    pool: "ActivationSwapPool"
    swap_stream: "torch.cuda.Stream"
    slot_id: int
    shape: tuple[int, ...]
    #: Stride (in ELEMENTS of dtype, matching ``torch.Tensor.stride()``)
    #: of the original GPU tensor at pack time. Capturing it is
    #: load-bearing: PyTorch's ``F.linear`` saves ``weight`` with stride
    #: ``(1, in_dim)`` because the matmul wants the transposed view, and
    #: other ops likewise save tensors with non-row-major strides. If
    #: ``unpack_from_pool`` rebuilt the GPU view with a guessed
    #: contiguous stride (the default of ``torch.empty(shape)``),
    #: backward kernels would read storage in the wrong element order
    #: and produce silently-wrong upstream gradients. Same lesson as
    #: ``OffloadedBlock``'s ``_ParamHandle.stride`` — see ``offload.py``
    #: for the empirical Linear-block divergence that motivated it.
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    nbytes: int
    requires_grad: bool


class _PassThrough:
    """Sentinel for tensors that bypass swapping (too small / not on GPU).

    We wrap the original tensor so the pack/unpack pair is symmetrical
    and ``unpack_from_pool`` can dispatch on type rather than checking
    ``isinstance(handle, torch.Tensor)`` which would conflict with the
    "saved tensor IS a tensor" idiom on the cold path.
    """

    __slots__ = ("tensor",)

    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor


def _make_pack_unpack(
    pool: "ActivationSwapPool",
    swap_stream: "torch.cuda.Stream",
    size_threshold: int,
):
    """Build the (pack, unpack) hook pair bound to ``pool``/``swap_stream``.

    A factory rather than a class so the hooks are plain closures —
    ``saved_tensors_hooks`` accepts any pair of callables and the
    closure form keeps the per-block state minimal.
    """

    def pack_to_pool(t: torch.Tensor):
        # Cold path — non-CUDA tensor or below the swap threshold.
        # Returning a ``_PassThrough`` keeps the saved-tensor reference
        # cheap (no slot acquisition) without changing the autograd
        # contract.
        if not isinstance(t, torch.Tensor) or not t.is_cuda:
            return _PassThrough(t)
        nbytes = t.numel() * t.element_size()
        if nbytes < size_threshold:
            return _PassThrough(t)
        if nbytes > pool.slot_bytes:
            # Defensive: tensor exceeds slot size. Keep on GPU rather
            # than corrupt memory. The wrap-time sizing in the model
            # wrapper should have prevented this; log and pass through.
            LOG.error(
                "_swap pack: tensor of %d bytes exceeds pool slot "
                "%d bytes — keeping on GPU",
                nbytes,
                pool.slot_bytes,
            )
            return _PassThrough(t)
        # Pool may be exhausted under pathological scheduling. Fall
        # back to identity rather than raising — autograd will simply
        # keep this tensor on GPU.
        try:
            slot_id, slot_view = pool.acquire()
        except RuntimeError:
            LOG.warning(
                "_swap pack: pool exhausted (n_slot=%d, in-flight=%d); "
                "keeping tensor on GPU",
                pool.n_slot,
                pool.inflight_count,
            )
            return _PassThrough(t)

        # Make the swap stream wait on the compute stream before
        # reading ``t``. Device-scoped so the wait targets ``t``'s real
        # device rather than the ambient current device (multi-GPU safe).
        # Any failure between ``pool.acquire()`` succeeding and the
        # ``_CPUHandle`` being returned would otherwise leak the slot
        # for the rest of the run; release on every exceptional path
        # so the pool can't get artificially exhausted.
        try:
            _swap_stream_wait_compute(t.device, swap_stream)
            with torch.cuda.stream(swap_stream):
                slot_target = slot_view[:nbytes].view(t.dtype).reshape(t.shape)
                slot_target.copy_(t.detach(), non_blocking=True)
                # Tell the allocator: this storage is in use by swap_stream
                # too, so don't reuse it until swap_stream catches up.
                t.record_stream(swap_stream)

            return _CPUHandle(
                pool=pool,
                swap_stream=swap_stream,
                slot_id=slot_id,
                shape=tuple(t.shape),
                stride=tuple(int(s) for s in t.stride()),
                dtype=t.dtype,
                device=t.device,
                nbytes=nbytes,
                requires_grad=t.requires_grad,
            )
        except BaseException:
            pool.release(slot_id)
            raise

    def unpack_from_pool(handle):
        # Cold-path passthrough — return the original tensor unchanged.
        if isinstance(handle, _PassThrough):
            return handle.tensor

        if not isinstance(handle, _CPUHandle):
            # Defensive: PyTorch internals may pass other types through
            # the unpack hook (e.g. None for retained_grad sentinels).
            return handle

        # H2D from pinned slot to a fresh GPU buffer.
        # ``record_stream`` keeps the GPU-side ``gpu_buf`` storage alive
        # across the swap stream, but pinned **host** memory is NOT
        # managed by the CUDA caching allocator, so ``record_stream``
        # gives us nothing on the source side — the only thing that
        # protects the pinned slot from a concurrent ``pool.close()``
        # (which frees the pinned region as soon as ``_live_borrows``
        # hits zero, see ``PinnedHostMemory.close``) is keeping the
        # borrow alive until the DMA has actually completed. Stream
        # ordering on swap_stream itself guards reuse-via-acquire
        # within the same stream, but ``close()`` consults the borrow
        # counter on the host with no awareness of swap_stream events.
        # Allocate the destination GPU buffer with the ORIGINAL tensor's
        # stride, not a contiguous default. ``torch.empty(shape)`` would
        # give us ``stride=row-major(shape)``, which mismatches the
        # ``.stride()`` of the tensor we packed for any non-contiguous
        # save (e.g. ``F.linear``'s ``(1, in_dim)`` weight stride).
        # Backward kernels that consume the saved tensor read its
        # storage via the recorded stride; rebuilding with a guessed
        # stride silently corrupts upstream gradients. ``empty_strided``
        # allocates storage sized to cover the full strided extent and
        # exposes the requested stride directly. The downstream
        # ``copy_`` from the contiguous CPU slot resolves logically
        # (PyTorch's ``copy_`` performs an elementwise copy regardless
        # of source/destination stride mismatch), so the saved-tensor
        # values match the original at every logical index while the
        # underlying storage is laid out the way the original tensor's
        # storage was.
        #
        # Headroom gate — uphold the cost model's "swap-in only when
        # memory is available" invariant. ``cost/memory.py`` documents
        # that SWAP blocks are modelled as zero contribution to the
        # op-walk peak under the paper's assumption that swap-in only
        # fires when memory is available. The runtime never actually
        # checked this: ``empty_strided`` would just OOM in backward,
        # and an analytically-feasible config could crash mysteriously.
        # The gate below bridges that gap. ``mem_get_info`` queries the
        # CUDA driver's free-memory counter (which already accounts for
        # memory cached by PyTorch's allocator), so it answers "would a
        # fresh allocation succeed" — exactly what we need.
        #
        # Two-stage enforcement:
        #   1. Bounded retry with ``synchronize()`` — backward kernels
        #      free their saved tensors after they finish, and a sync
        #      drains any in-flight ones, typically opening enough
        #      headroom for the swap-in. We only synchronize on the
        #      deficit branch — the drain has a real cost (full
        #      compute-stream wait), so the conditional is load-bearing
        #      for steady-state throughput.
        #   2. Hard raise — if even after draining we still cannot
        #      satisfy the headroom requirement, raise ``RuntimeError``
        #      with an actionable message. Falling through to
        #      ``empty_strided`` would also abort, but as a CUDA
        #      kernel-allocator OOM that blames the wrong layer; the
        #      raise here turns the cost model's zero-peak assumption
        #      into a checkable invariant.
        # The ``handle.slot_id`` was acquired by ``pack_to_pool``; this
        # function owns its release. Wrap the entire body so the
        # explicit headroom ``RuntimeError``, allocator failures, and
        # copy failures all flow through ``pool.release`` (and
        # ``release_buffer`` for the second borrow taken below) — without
        # this, a single SWAP gate trip leaks the slot for the rest of
        # the run and the pool can stay artificially exhausted.
        second_borrow_acquired = False
        try:
            required_bytes = handle.nbytes
            total_safety = required_bytes + _SWAP_HEADROOM_SAFETY_BYTES

            free_bytes, _total = torch.cuda.mem_get_info(handle.device)
            retries_remaining = _SWAP_MAX_DRAIN_RETRIES
            drained = False
            while free_bytes < total_safety and retries_remaining > 0:
                torch.cuda.synchronize(handle.device)
                free_bytes, _total = torch.cuda.mem_get_info(handle.device)
                retries_remaining -= 1
                drained = True

            if free_bytes < total_safety:
                raise RuntimeError(
                    f"ProTrain SWAP gate: insufficient GPU headroom for "
                    f"activation swap-in on device {handle.device} after "
                    f"{_SWAP_MAX_DRAIN_RETRIES} sync-and-retry attempts. "
                    f"Need {required_bytes} bytes + "
                    f"{_SWAP_HEADROOM_SAFETY_BYTES} safety margin; have "
                    f"{free_bytes} free. The cost model assumed SWAP would "
                    f"not contribute to peak (paper §3.3 'swap-in only when "
                    f"memory available'); this configuration violates that "
                    f"invariant. Reduce n_swap or set n_swap=0 and re-run "
                    f"the searcher, or check whether parallel SWAP traffic "
                    f"from sibling unpacks is competing for the same "
                    f"headroom."
                )

            if drained:
                # Near-miss: the gate had to drain in-flight backward
                # kernels to recover headroom. Surface to operators so
                # repeated near-misses are visible before they tip into
                # an actual raise.
                LOG.warning(
                    "SWAP unpack: drained in-flight backward kernels to "
                    "recover headroom for swap-in on device %s "
                    "(need %d bytes + %d safety, have %d free after drain). "
                    "Repeated near-misses suggest n_swap is too high for "
                    "this configuration.",
                    handle.device,
                    required_bytes,
                    _SWAP_HEADROOM_SAFETY_BYTES,
                    free_bytes,
                )

            # App B.2: route the GPU activation buffer through the
            # default-stream heap. The subsequent H2D copy runs on
            # ``swap_stream`` for compute/copy overlap, but the allocation
            # itself must come from the default heap so the caching
            # allocator can reuse this region across iterations rather than
            # pinning it to ``swap_stream``'s per-stream free-list. The
            # ``gpu_buf.record_stream(handle.swap_stream)`` call inside the
            # swap_stream context below ties the buffer's lifetime to the
            # swap_stream's work so the allocator's free path waits for the
            # H2D to retire — preventing the allocator-frees-mid-DMA
            # silent-corruption window.
            with SingleStreamAllocator():
                gpu_buf = torch.empty_strided(
                    handle.shape,
                    handle.stride,
                    dtype=handle.dtype,
                    device=handle.device,
                )
            _swap_stream_wait_compute(handle.device, handle.swap_stream)
            h2d_done: "torch.cuda.Event | None" = None
            with torch.cuda.stream(handle.swap_stream):
                slot_view = handle.pool._pinned.buffer(handle.slot_id)  # noqa: SLF001
                second_borrow_acquired = True
                slot_src = (
                    slot_view[: handle.nbytes].view(handle.dtype).reshape(handle.shape)
                )
                gpu_buf.copy_(slot_src, non_blocking=True)
                gpu_buf.record_stream(handle.swap_stream)
                # Record an event on swap_stream that fires when the H2D
                # copy above has completed. We use this below to gate the
                # borrow release so the pinned slot stays "live" (from the
                # allocator's perspective) until the DMA is actually done.
                h2d_done = torch.cuda.Event()
                h2d_done.record(handle.swap_stream)
                # Drop our local references to the slot view BEFORE
                # releasing the borrow that backs them. ``release_buffer``
                # only decrements the borrow counter; the underlying
                # storage stays alive while the DMA is in flight thanks to
                # the event-gated release sequencing below.
                del slot_view, slot_src
            _compute_stream_wait_swap(handle.device, handle.swap_stream)

            # Block the host until the H2D copy has actually retired on
            # the device. Only after the event has fired is it safe to
            # decrement the pinned-allocator borrow counter, because that
            # counter is the sole signal ``PinnedHostMemory.close()`` uses
            # to decide whether ``cudaFreeHost`` is safe — releasing
            # before the DMA finishes opens a window where a concurrent
            # ``close()`` would free the pinned region mid-transfer and
            # the H2D DMA would read freed memory (silent data corruption
            # in the activation that backward then consumes).
            #
            # The host-side wait is acceptable here: backward is the
            # consumer of the unpacked tensor and will already wait on
            # swap_stream before the kernel that reads ``gpu_buf`` runs;
            # this synchronize() simply pulls that wait to the host so
            # the borrow accounting is honest. Pipelined throughput is
            # unaffected as long as backward kernels keep the compute
            # stream busy while the next unpack's H2D enqueues.
            if h2d_done is not None:
                h2d_done.synchronize()
        finally:
            # Release the second ``buffer()`` borrow if it was
            # acquired (success path *and* failure path after the
            # ``with torch.cuda.stream`` block opened), then return
            # the acquire-time slot to the pool. Same-stream ordering
            # guards reuse on the success path; on the failure path
            # the host-side ``synchronize`` above is skipped, but the
            # exception propagates so no later kernel will read this
            # slot anyway.
            if second_borrow_acquired:
                handle.pool._pinned.release_buffer(handle.slot_id)  # noqa: SLF001
            handle.pool.release(handle.slot_id)

        # Restore requires_grad flag if the original tensor had one.
        # Saved tensors that participated in autograd should preserve
        # their grad-fn linkage; ``empty()`` returns a leaf, but the
        # consumer of an unpacked saved-tensor reads it as data only
        # (no grad flows backward through the saved tensor itself —
        # that's a property of save_for_backward semantics).
        if handle.requires_grad:
            gpu_buf.requires_grad_(True)
        return gpu_buf

    return pack_to_pool, unpack_from_pool


class SwappedBlock(nn.Module):
    """Wrap an ``nn.Module`` so its saved tensors are swapped to pinned CPU.

    Construction is unconditional. Gating happens via the searcher's
    ``n_swap`` decision (the cost model + memory feasibility filters).

    The pool + swap stream are injected post-construction via
    :meth:`attach_runtime`. Until that call, the wrapper passes the
    block forward through unchanged — no saved_tensors_hooks context
    is installed, so saved tensors live on GPU as they normally would.
    """

    def __init__(self, block: nn.Module) -> None:
        """Wrap ``block`` in identity-mode; runtime wiring deferred to :meth:`attach_runtime`."""
        super().__init__()
        self.block = block
        self._protrain_wrapped_mode: BlockMode = BlockMode.SWAP
        self._swap_pool: "ActivationSwapPool | None" = None
        self._swap_stream: "torch.cuda.Stream | None" = None
        self._warned_no_runtime = False

    def attach_runtime(
        self,
        pool: "ActivationSwapPool",
        swap_stream: "torch.cuda.Stream | None",
    ) -> None:
        """Wire the pinned-pool + swap stream into this wrapper.

        Idempotent — re-attaching with the same pool/stream is a no-op;
        re-attaching with a new pool/stream is legal (e.g. after a
        re-search at epoch boundaries).
        """
        self._swap_pool = pool
        self._swap_stream = swap_stream

    def detach_runtime(self) -> None:
        """Drop the pool + stream refs — wrapper degrades to identity."""
        self._swap_pool = None
        self._swap_stream = None

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run the wrapped block under saved_tensors_hooks that swap to pinned CPU."""
        pool = self._swap_pool
        stream = self._swap_stream

        # Cold path — no runtime attached. Run the block plain.
        if pool is None or stream is None or not torch.cuda.is_available():
            if (pool is None or stream is None) and not self._warned_no_runtime:
                missing = (
                    "pool+stream"
                    if pool is None and stream is None
                    else ("pool" if pool is None else "stream")
                )
                LOG.warning(
                    "SwappedBlock forward without attached runtime "
                    "(missing %s) — degrading to identity. Call "
                    "attach_runtime(pool, stream) after constructing "
                    "the block.",
                    missing,
                )
                self._warned_no_runtime = True
            return self.block(*args, **kwargs)

        # Hot path — install saved_tensors_hooks for the duration of
        # the wrapped block's forward. Every saved tensor created
        # inside this context goes through ``pack_to_pool``; backward
        # restores them via ``unpack_from_pool``.
        pack, unpack = _make_pack_unpack(pool, stream, SIZE_THRESHOLD_BYTES)
        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            out = self.block(*args, **kwargs)
        return out

    def extra_repr(self) -> str:
        """Return the wrapper's mode tag for ``print(model)``."""
        return f"mode={self._protrain_wrapped_mode.value}"


__all__ = ["SIZE_THRESHOLD_BYTES", "SwappedBlock"]
