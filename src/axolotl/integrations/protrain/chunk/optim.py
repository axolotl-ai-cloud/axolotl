"""Fused-Adam adapters for persistent (GPU) and non-persistent (CPU) chunks.

Two classes with a similar shape:

* :class:`CpuFusedAdamAdapter` wraps ``deepspeed.ops.adam.DeepSpeedCPUAdam``
  and adds a ``step_async(chunk_id)`` path so the CPU optimizer step for
  chunk ``c`` can launch the instant that chunk's grads have been
  reduce-offloaded — overlapping with GPU backward for later chunks (§5).
* :class:`GpuFusedAdamAdapter` wraps Apex ``FusedAdam`` (or falls back to
  ``torch.optim.AdamW`` with a warning) for the persistent-resident subset.

Async semantics: we use a single-worker ``ThreadPoolExecutor``. DeepSpeed's
CPU Adam kernel releases the GIL inside its compiled op, so "async" here
means "run overlapped with the GPU kernels the main Python thread is
launching", not parallel across chunks. Serializing through one worker also
sidesteps the CPU Adam op's internal state sharing between chunks of the
same optimizer instance.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Iterable

from axolotl.integrations.protrain.types import ChunkId
from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    from torch import nn

LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# CPU FusedAdam — non-persistent chunks
# ---------------------------------------------------------------------------


class CpuFusedAdamAdapter:
    """Per-chunk CPU FusedAdam driver for the non-persistent chunk set.

    We construct one underlying ``DeepSpeedCPUAdam`` instance per chunk.
    That matches the design where each non-persistent chunk's params live
    on CPU (sharded), their gradients are reduced and D2H-copied back to
    the same shard, and the CPU step consumes them in place. Keeping the
    instances separate per chunk means :meth:`step_async` can target
    exactly one chunk's param group without touching the others.
    """

    def __init__(
        self,
        params_per_chunk: dict[ChunkId, list["nn.Parameter"]],
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        """Build one ``DeepSpeedCPUAdam`` instance per chunk and a single worker thread."""
        try:
            from deepspeed.ops.adam import (
                DeepSpeedCPUAdam,  # type: ignore[import-not-found]
            )
        except ImportError as err:
            raise ImportError(
                "CpuFusedAdamAdapter requires DeepSpeed's CPU Adam kernel — "
                "install via `pip install axolotl[deepspeed]`."
            ) from err

        self._DeepSpeedCPUAdam = DeepSpeedCPUAdam
        self._params_per_chunk = dict(params_per_chunk)
        self.lr = float(lr)
        self.betas = (float(betas[0]), float(betas[1]))
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)

        # One DeepSpeedCPUAdam per chunk — cheap; shares no state.
        # DeepSpeedCPUAdam silently constructs a half-initialized object
        # when the C++ adam_bindings extension fails to compile (e.g.
        # under a system CUDA / torch CUDA version mismatch — the
        # warning surfaces from `deepspeed.ops.op_builder` but the
        # constructor doesn't raise). The half-init object lacks
        # ``ds_opt_adam`` and crashes later in both ``.step()`` and
        # ``__del__``. We probe for the attribute right after each
        # construction; missing means the extension isn't loaded and we
        # raise so callers' try/except can fall back to the inline GPU
        # optimizer path. Without this guard the bad objects survive,
        # their ``__del__`` AttributeErrors propagate as
        # PytestUnraisableExceptionWarning and accumulate into test
        # failures whenever multiple adapter constructions happen
        # (phase-2 profiler bootstrap → rebuild → user optim wrapper).
        self._optims: dict[ChunkId, Any] = {}
        for cid, params in self._params_per_chunk.items():
            if not params:
                continue
            opt = DeepSpeedCPUAdam(
                params,
                lr=self.lr,
                betas=self.betas,
                eps=self.eps,
                weight_decay=self.weight_decay,
            )
            if not hasattr(opt, "ds_opt_adam"):
                # Suppress this object's __del__ AttributeError so the
                # raise below propagates cleanly. DeepSpeed's destructor
                # calls ``self.ds_opt_adam.destroy_adam(self.opt_id)``;
                # planting a no-op stub keeps the destructor harmless
                # without monkey-patching the special __del__ slot.
                class _NoopDsAdam:  # noqa: N801 — internal stub
                    def destroy_adam(self, _opt_id):
                        return None

                try:
                    opt.ds_opt_adam = _NoopDsAdam()  # type: ignore[attr-defined]
                except Exception:  # noqa: BLE001 — best-effort cleanup
                    pass
                raise RuntimeError(
                    "DeepSpeedCPUAdam C++ extension (adam_bindings) is not "
                    "loaded — the constructed object is missing "
                    "`ds_opt_adam` and will crash on .step(). Common "
                    "cause: system nvcc CUDA version differs from the "
                    "version PyTorch was compiled with. Either install a "
                    "matching CUDA toolkit or set DS_SKIP_CUDA_CHECK=1 "
                    "and rebuild DeepSpeed."
                )
            self._optims[cid] = opt

        # Single-worker executor — see module docstring for rationale.
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="protrain-cpu-adam"
        )
        self._pending: dict[ChunkId, Future[None]] = {}

    # ---- step interface -------------------------------------------------

    def step_async(
        self,
        chunk_id: ChunkId,
        d2h_event: Any = None,
        post_step: Any = None,
    ) -> "Future[None]":
        """Submit the CPU Adam step for ``chunk_id`` to the worker thread.

        Idempotent with :meth:`wait`: if a prior step is still pending for
        the same chunk, we wait for it first so we never run two steps
        concurrently against the same param shard.

        Parameters
        ----------
        chunk_id:
            The chunk whose CPU Adam step to run.
        d2h_event:
            Optional :class:`torch.cuda.Event` recorded by the caller on
            the CUDA stream immediately after the grad D2H copy was
            issued. When provided, the worker thread calls
            ``event.synchronize()`` before invoking ``optim.step()`` —
            this closes the CPU-Adam ↔ D2H race (BUG 1 fix): without
            this wait, the worker can read uninitialized/partial bytes
            from the pinned grad shard before the async D2H finishes.
        post_step:
            Optional zero-arg callable invoked on the worker thread
            after ``optim.step()`` returns (before the future resolves).
            The chunk manager uses this to repoint ``param.data`` back
            to the GPU empty-placeholder so intermediate code between
            iters doesn't see CPU-resident ``.data`` (BUG 4 fix).
        """
        prev = self._pending.get(chunk_id)
        if prev is not None and not prev.done():
            prev.result()  # propagate any exception
        optim = self._optims.get(chunk_id)
        if optim is None:
            # No params belonging to this chunk live on CPU (e.g. a fully
            # persistent layout). Run the post_step (if any) inline and
            # return an already-completed future.
            fut: Future[None] = Future()
            if post_step is not None:
                try:
                    post_step()
                except Exception as exc:  # noqa: BLE001
                    fut.set_exception(exc)
                    self._pending[chunk_id] = fut
                    return fut
            fut.set_result(None)
            self._pending[chunk_id] = fut
            return fut

        def _run() -> None:
            # Wait on the CUDA event (if any) so the D2H copy into the
            # pinned grad shard is guaranteed complete before Adam reads
            # it. ``Event.synchronize`` blocks the calling thread (here,
            # the Adam worker) until the event has been recorded on the
            # GPU — the main Python thread is free to continue launching
            # subsequent backward kernels, which is the overlap we want.
            if d2h_event is not None:
                d2h_event.synchronize()
            optim.step()
            if post_step is not None:
                post_step()

        fut = self._executor.submit(_run)
        self._pending[chunk_id] = fut
        return fut

    def wait(self, chunk_id: ChunkId) -> None:
        """Block until ``step_async(chunk_id)``'s worker has finished."""
        fut = self._pending.get(chunk_id)
        if fut is None:
            return
        fut.result()  # re-raises worker exceptions on the caller's thread

    def wait_all(self) -> None:
        """Block until every in-flight chunk step has finished.

        Every pending future is awaited even if one raises, so gradient
        computation is not left in an incomplete state. The first captured
        exception is re-raised after all futures have been awaited; any
        additional exceptions are logged.
        """
        errors: list[BaseException] = []
        for fut in list(self._pending.values()):
            try:
                fut.result()
            except BaseException as exc:  # noqa: BLE001 — re-raised below
                errors.append(exc)
        if errors:
            if len(errors) > 1:
                LOG.error(
                    "wait_all: %d additional errors suppressed",
                    len(errors) - 1,
                )
            raise errors[0]

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero gradients across every chunk's params."""
        for optim in self._optims.values():
            optim.zero_grad(set_to_none=set_to_none)

    # ---- lifecycle ------------------------------------------------------

    def shutdown(self) -> None:
        """Tear down the worker pool. Call explicitly before process exit.

        ``wait_all()`` may re-raise a worker exception. We still need to
        release the executor in that case — otherwise the thread pool
        leaks on the explicit-cleanup path and ``__del__`` would swallow
        the failure silently. Run the executor shutdown in ``finally``
        and re-raise the original error after the pool is released.
        """
        error: BaseException | None = None
        try:
            self.wait_all()
        except BaseException as exc:  # noqa: BLE001 — re-raised below
            error = exc
        finally:
            self._executor.shutdown(wait=True)
        if error is not None:
            raise error

    def __del__(self) -> None:  # noqa: D401
        try:
            self.shutdown()
        except Exception:  # noqa: BLE001 — destructors must not throw
            pass


# ---------------------------------------------------------------------------
# GPU FusedAdam — persistent chunks
# ---------------------------------------------------------------------------


class GpuFusedAdamAdapter:
    """Synchronous fused GPU Adam for the persistent chunk set.

    Prefers ``apex.optimizers.FusedAdam`` (paper-cited backend). Falls back
    to stock ``torch.optim.AdamW`` with a warning when Apex is unavailable
    — the cost model will be off in that case (AdamW is a distinct update
    rule, not just a different kernel) but training stays correct.
    """

    def __init__(
        self,
        params: Iterable["nn.Parameter"],
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        """Build the underlying fused GPU optimizer over ``params``."""
        param_list = [p for p in params if p is not None]

        self.lr = float(lr)
        self.betas = (float(betas[0]), float(betas[1]))
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)

        # Empty persistent set is a valid Mode-C state (e.g. a config where
        # all chunks are non-persistent / live on CPU). Both Apex FusedAdam
        # and torch.optim.AdamW raise ValueError on an empty params list,
        # so short-circuit to a no-op adapter: step()/zero_grad() do
        # nothing and state_dict() returns the empty dict shape that
        # torch optimizers use.
        if len(param_list) == 0:
            self._optim = None
            return

        self._optim = self._build_optim(param_list)

    def _build_optim(self, params: list["nn.Parameter"]) -> Any:
        """Return Apex ``FusedAdam`` if importable, else ``torch.optim.AdamW``."""
        try:
            from apex.optimizers import FusedAdam  # type: ignore[import-not-found]
        except ImportError as exc:
            exc_repr = f"{type(exc).__name__}: {exc}"
            LOG.warning(
                "apex.optimizers.FusedAdam import failure (%s); falling back to "
                "torch.optim.AdamW for the persistent-chunk optimizer. "
                "Install Apex for the paper-configured fused kernel.",
                exc_repr,
            )
            del exc

            import torch

            return torch.optim.AdamW(
                params,
                lr=self.lr,
                betas=self.betas,
                eps=self.eps,
                weight_decay=self.weight_decay,
            )

        return FusedAdam(
            params,
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )

    # ---- step interface -------------------------------------------------

    def step(self) -> None:
        """Synchronous fused GPU Adam step over persistent-chunk params."""
        optim = self._optim
        if optim is None:
            return
        optim.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero gradients on every persistent-chunk parameter."""
        optim = self._optim
        if optim is None:
            return
        optim.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        """Return the wrapped optimizer's state dict (empty when no-op)."""
        optim = self._optim
        if optim is None:
            return {"state": {}, "param_groups": []}
        return optim.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state into the wrapped optimizer (no-op when adapter is empty)."""
        optim = self._optim
        if optim is None:
            if state_dict.get("state") or state_dict.get("param_groups"):
                raise ValueError(
                    "Cannot load non-empty optimizer state into an empty "
                    "GpuFusedAdamAdapter: this layout has no persistent-chunk "
                    "params but the checkpoint contains optimizer state "
                    "(likely a Mode-A/Mode-C config mismatch on resume)."
                )
            return
        optim.load_state_dict(state_dict)

    @property
    def underlying(self) -> Any:
        """The wrapped optimizer instance (useful for LR schedulers).

        ``None`` when the adapter wraps an empty persistent param set.
        """
        return self._optim


__all__ = ["CpuFusedAdamAdapter", "GpuFusedAdamAdapter"]
