"""Activation-swap wrapper: saved_tensors_hooks D2H every saved tensor, H2D on backward."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from axolotl.integrations.protrain.block.strategy import BlockMode
from axolotl.integrations.protrain.runtime.streams import SingleStreamAllocator
from axolotl.utils.logging import get_logger

try:
    from torch.compiler import disable as _compile_disable
except Exception:  # noqa: BLE001 — older torches lack torch.compiler.disable

    def _compile_disable(fn=None, *, recursive=True):  # noqa: ARG001
        return fn if fn is not None else (lambda f: f)


if TYPE_CHECKING:
    from axolotl.integrations.protrain.block.swap_pool import ActivationSwapPool

LOG = get_logger(__name__)


#: Saved tensors smaller than this many bytes are kept on GPU (not
#: swapped). 1 MiB is the default; tests may override by reassigning
#: this module attribute. See the module docstring for derivation.
SIZE_THRESHOLD_BYTES: int = 1 << 20  # 1 MiB


def _is_non_overlapping_and_dense(t: "torch.Tensor") -> bool:
    """Return True iff ``t``'s strided layout has no internal aliasing."""
    if t.numel() == 0:
        return True
    pairs = sorted(
        ((s, st) for s, st in zip(t.shape, t.stride(), strict=True) if s > 1),
        key=lambda p: p[1],
    )
    if not pairs:
        return True
    expected = 1
    for size, stride in pairs:
        if stride != expected:
            return False
        expected = stride * size
    return True


# 64 MiB safety margin absorbs in-flight backward transients beyond the swap-in alloc.
_SWAP_HEADROOM_SAFETY_BYTES: int = 64 * 1024 * 1024


# 3 sync-recheck retries drain backward kernels' saved-tensor storage.
_SWAP_MAX_DRAIN_RETRIES: int = 3


def _swap_stream_wait_compute(
    device: "torch.device", swap_stream: "torch.cuda.Stream"
) -> None:
    """Make swap_stream wait on the compute stream of device (explicit device avoids cross-GPU race)."""
    if swap_stream is None or not torch.cuda.is_available():
        return
    swap_stream.wait_stream(torch.cuda.current_stream(device=device))


def _compute_stream_wait_swap(
    device: "torch.device", swap_stream: "torch.cuda.Stream"
) -> None:
    """Make compute stream wait on swap_stream (explicit device avoids cross-GPU race)."""
    if swap_stream is None or not torch.cuda.is_available():
        return
    torch.cuda.current_stream(device=device).wait_stream(swap_stream)


@dataclass
class _CPUHandle:
    """CPU-resident handle returned by ``pack_to_pool``."""

    pool: "ActivationSwapPool"
    swap_stream: "torch.cuda.Stream"
    slot_id: int
    shape: tuple[int, ...]
    # Stride captured at pack time; F.linear saves weight with non-row-major stride.
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    nbytes: int
    requires_grad: bool


class _PassThrough:
    """Sentinel for tensors that bypass swapping (too small / not on GPU)."""

    __slots__ = ("tensor",)

    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor


def _make_pack_unpack(
    pool: "ActivationSwapPool",
    swap_stream: "torch.cuda.Stream",
    size_threshold: int,
):
    """Build the (pack, unpack) hook pair bound to ``pool``/``swap_stream``."""

    def pack_to_pool(t: torch.Tensor):
        # Cold paths: non-CUDA, below threshold, or overlapping/non-dense → PassThrough.
        if not isinstance(t, torch.Tensor) or not t.is_cuda:
            return _PassThrough(t)
        nbytes = t.numel() * t.element_size()
        if nbytes < size_threshold:
            return _PassThrough(t)
        # empty_strided + copy_ requires non-overlapping; reimplement std PyTorch check.
        if not _is_non_overlapping_and_dense(t):
            return _PassThrough(t)
        if nbytes > pool.slot_bytes:
            # Tensor exceeds slot size; keep on GPU to avoid corruption.
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
        did_dma = False
        try:
            _swap_stream_wait_compute(t.device, swap_stream)
            with torch.cuda.stream(swap_stream):
                slot_target = slot_view[:nbytes].view(t.dtype).reshape(t.shape)
                slot_target.copy_(t.detach(), non_blocking=True)
                # The async D2H is now enqueued on swap_stream. Any failure
                # past this point must drain swap_stream BEFORE releasing
                # the pinned slot; otherwise the allocator can hand the
                # slot's pinned memory back out (or ``PinnedHostMemory.close``
                # can ``cudaFreeHost`` it) while the in-flight copy is still
                # writing to it, silently corrupting the next acquirer's
                # data.
                did_dma = True
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
            if did_dma:
                # Block the host until the in-flight D2H retires so that
                # ``pool.release`` (and any subsequent reuse / close) can't
                # free the pinned region mid-transfer.
                swap_stream.synchronize()
            pool.release(slot_id)
            raise

    def unpack_from_pool(handle):
        # Cold paths: passthrough wraps + defensive non-handle types.
        if isinstance(handle, _PassThrough):
            return handle.tensor

        if not isinstance(handle, _CPUHandle):
            return handle

        # Headroom-gated H2D into empty_strided buffer matching original stride.
        second_borrow_acquired = False
        # h2d_done is the precise fence; did_h2d is the coarse fallback for finally.
        h2d_done: "torch.cuda.Event | None" = None
        did_h2d = False
        try:
            # Use strided storage extent (max_offset+1), not numel*esize, for headroom.
            element_size = handle.dtype.itemsize
            if handle.shape:
                max_offset = sum(
                    (s - 1) * st
                    for s, st in zip(handle.shape, handle.stride, strict=True)
                )
                storage_numel = max_offset + 1
            else:
                storage_numel = 1
            required_bytes = storage_numel * element_size
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
                # Near-miss: drain succeeded. Surface so repeated drains are visible pre-failure.
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

            # Route alloc through default-stream heap; record_stream ties lifetime to swap_stream.
            with SingleStreamAllocator():
                gpu_buf = torch.empty_strided(
                    handle.shape,
                    handle.stride,
                    dtype=handle.dtype,
                    device=handle.device,
                )
            _swap_stream_wait_compute(handle.device, handle.swap_stream)
            with torch.cuda.stream(handle.swap_stream):
                slot_view = handle.pool._pinned.buffer(handle.slot_id)  # noqa: SLF001
                second_borrow_acquired = True
                slot_src = (
                    slot_view[: handle.nbytes].view(handle.dtype).reshape(handle.shape)
                )
                gpu_buf.copy_(slot_src, non_blocking=True)
                # Mark DMA enqueued so finally can fence even if a later statement raises pre-event.
                did_h2d = True
                gpu_buf.record_stream(handle.swap_stream)
                # Event-gated borrow release: only safe to drop counter after DMA retires.
                h2d_done = torch.cuda.Event()
                h2d_done.record(handle.swap_stream)
                del slot_view, slot_src
            _compute_stream_wait_swap(handle.device, handle.swap_stream)

            # Host sync before borrow release; otherwise close() could free pinned mid-DMA.
            if h2d_done is not None:
                h2d_done.synchronize()
        finally:
            # Three-tier fence: event sync, full stream sync, or no-op.
            if h2d_done is not None:
                h2d_done.synchronize()
            elif did_h2d:
                handle.swap_stream.synchronize()
            if second_borrow_acquired:
                handle.pool._pinned.release_buffer(handle.slot_id)  # noqa: SLF001
            handle.pool.release(handle.slot_id)

        # Restore requires_grad; saved tensors are read as data only by consumers.
        if handle.requires_grad:
            gpu_buf.requires_grad_(True)
        return gpu_buf

    return pack_to_pool, unpack_from_pool


class SwappedBlock(nn.Module):
    """Wrap an nn.Module so its saved tensors are swapped to pinned CPU."""

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

    @_compile_disable(recursive=True)
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

        # Install saved_tensors_hooks for the duration of the block's forward.
        pack, unpack = _make_pack_unpack(pool, stream, SIZE_THRESHOLD_BYTES)
        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            out = self.block(*args, **kwargs)
        return out

    def extra_repr(self) -> str:
        """Return the wrapper's mode tag for ``print(model)``."""
        return f"mode={self._protrain_wrapped_mode.value}"


__all__ = ["SIZE_THRESHOLD_BYTES", "SwappedBlock"]
