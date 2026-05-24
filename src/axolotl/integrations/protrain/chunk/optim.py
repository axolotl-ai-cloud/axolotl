"""Fused-Adam adapters for persistent (GPU) and non-persistent (CPU) chunks."""

from __future__ import annotations

import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Iterable

from axolotl.integrations.protrain.types import ChunkId
from axolotl.utils.logging import get_logger


def _slow_adam_step_threshold_s() -> float:
    """PROTRAIN_DEBUG_SLOW_ADAM_STEP_S (default 5.0) — adam steps above this WARN-log per-chunk."""
    raw = os.environ.get("PROTRAIN_DEBUG_SLOW_ADAM_STEP_S", "5.0")
    try:
        v = float(raw)
    except ValueError:
        return 5.0
    return max(0.0, v)


_SLOW_ADAM_STEP_THRESHOLD_S: float = _slow_adam_step_threshold_s()

if TYPE_CHECKING:
    from torch import nn

LOG = get_logger(__name__)


class _DestroyedDsAdam:
    """Replaces a destroyed DeepSpeedCPUAdam C-state binding to neutralise __del__."""

    def destroy_adam(self, _opt_id):  # noqa: D401, ANN001
        return None


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

        # One DeepSpeedCPUAdam per chunk; probe for ds_opt_adam to catch half-init from extension load failure.
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
                # Plant no-op stub so __del__ doesn't AttributeError before the raise below.
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
        """Submit the CPU Adam step for ``chunk_id`` to the worker thread."""
        prev = self._pending.get(chunk_id)
        if prev is not None:
            # result() waits pending; raises for failed futures (don't short-circuit on done).
            prev.result()
        optim = self._optims.get(chunk_id)
        if optim is None:
            # No CPU params for this chunk; run post_step inline.
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
            t0 = time.perf_counter() if _SLOW_ADAM_STEP_THRESHOLD_S > 0.0 else 0.0
            if d2h_event is not None:
                d2h_event.synchronize()
            t_sync = time.perf_counter() if _SLOW_ADAM_STEP_THRESHOLD_S > 0.0 else 0.0
            optim.step()
            if post_step is not None:
                post_step()
            if _SLOW_ADAM_STEP_THRESHOLD_S > 0.0:
                total = time.perf_counter() - t0
                if total >= _SLOW_ADAM_STEP_THRESHOLD_S:
                    LOG.warning(
                        "CpuFusedAdamAdapter.step_async: chunk_id=%d total=%.3fs "
                        "(d2h_event_wait=%.3fs, optim.step=%.3fs, "
                        "threshold=%.1fs). First-call DS-CPU-Adam state alloc "
                        "+ kernel init may dominate; persistent slowness "
                        "indicates the executor is the bottleneck.",
                        int(chunk_id),
                        total,
                        t_sync - t0,
                        total - (t_sync - t0),
                        _SLOW_ADAM_STEP_THRESHOLD_S,
                    )

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
        """Await every in-flight chunk step; re-raise first Exception, log the rest."""
        errors: list[Exception] = []
        for fut in list(self._pending.values()):
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001 — re-raised below
                errors.append(exc)
        if errors:
            if len(errors) > 1:
                LOG.error(
                    "wait_all: %d additional errors suppressed",
                    len(errors) - 1,
                )
            raise errors[0]

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero grads after draining in-flight step_async futures; nulling mid-step would corrupt Adam."""
        self.wait_all()
        for optim in self._optims.values():
            optim.zero_grad(set_to_none=set_to_none)

    # ---- lifecycle ------------------------------------------------------

    def shutdown(self) -> None:
        """Tear down the worker pool + explicitly destroy DeepSpeedCPUAdam C state. Idempotent."""
        error: Exception | None = None
        try:
            self.wait_all()
        except Exception as exc:  # noqa: BLE001 — re-raised below
            error = exc
        finally:
            self._executor.shutdown(wait=True)
            self._destroy_ds_adam_state()
        if error is not None:
            raise error

    def _destroy_ds_adam_state(self) -> None:
        """Free each per-chunk DeepSpeedCPUAdam C++ state; stub ds_opt_adam to neutralise __del__."""
        for cid, opt in self._optims.items():
            ds_opt_adam = getattr(opt, "ds_opt_adam", None)
            if ds_opt_adam is None or isinstance(ds_opt_adam, _DestroyedDsAdam):
                continue
            opt_id = getattr(opt, "opt_id", None)
            try:
                ds_opt_adam.destroy_adam(opt_id)
            except Exception as exc:  # noqa: BLE001 — best-effort cleanup
                LOG.warning(
                    "DeepSpeedCPUAdam destroy_adam failed for chunk %s: %s",
                    cid,
                    exc,
                )
            try:
                opt.ds_opt_adam = _DestroyedDsAdam()  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001 — best-effort cleanup
                pass

    def __del__(self) -> None:  # noqa: D401
        try:
            self.shutdown()
        except Exception:  # noqa: BLE001 — destructors must not throw
            # GC ordering at shutdown causes spurious failures; debug-log only.
            LOG.debug(
                "CpuFusedAdamAdapter.__del__: shutdown failed",
                exc_info=True,
            )


# ---------------------------------------------------------------------------
# GPU FusedAdam — persistent chunks
# ---------------------------------------------------------------------------


class GpuFusedAdamAdapter:
    """Synchronous fused GPU Adam for the persistent chunk set; falls back to torch.optim.AdamW."""

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

        # Empty persistent set is valid (all-CPU Mode-C); short-circuit to a no-op adapter.
        if len(param_list) == 0:
            self._optim = None
            return

        self._optim = self._build_optim(param_list)

    def _build_optim(self, params: list["nn.Parameter"]) -> Any:
        """Return Apex FusedAdam if importable+instantiable, else torch.optim.AdamW."""
        import torch

        def _fallback_adamw() -> Any:
            return torch.optim.AdamW(
                params,
                lr=self.lr,
                betas=self.betas,
                eps=self.eps,
                weight_decay=self.weight_decay,
            )

        try:
            from apex.optimizers import FusedAdam  # type: ignore[import-not-found]
        except (ImportError, RuntimeError) as exc:
            # ImportError + RuntimeError both indicate apex CUDA extension load failure.
            exc_repr = f"{type(exc).__name__}: {exc}"
            LOG.warning(
                "apex.optimizers.FusedAdam import failure (%s); falling back to "
                "torch.optim.AdamW for the persistent-chunk optimizer. "
                "Install Apex for the paper-configured fused kernel.",
                exc_repr,
            )
            del exc
            return _fallback_adamw()

        try:
            return FusedAdam(
                params,
                lr=self.lr,
                betas=self.betas,
                eps=self.eps,
                weight_decay=self.weight_decay,
            )
        except Exception as exc:  # noqa: BLE001 — apex can raise non-ImportError on broken installs
            LOG.warning(
                "apex.optimizers.FusedAdam instantiation failure (%r); "
                "falling back to torch.optim.AdamW for the persistent-chunk "
                "optimizer. Install Apex with working CUDA extensions for the "
                "paper-configured fused kernel.",
                exc,
            )
            del exc
            return _fallback_adamw()

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
        """Return the wrapped optimizer (None when adapter has no persistent params)."""
        return self._optim


# bnb 8-bit Adam is CUDA-only: restricted to persistent chunks.


class GpuAdamW8bitAdapter:
    """Synchronous bitsandbytes 8-bit AdamW for persistent (GPU-resident) chunks."""

    def __init__(
        self,
        params: Iterable["nn.Parameter"],
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        *,
        paged: bool = False,
    ) -> None:
        """Build the underlying ``bnb.optim.AdamW8bit`` (or paged variant) over ``params``."""
        param_list = [p for p in params if p is not None]

        self.lr = float(lr)
        self.betas = (float(betas[0]), float(betas[1]))
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.paged = bool(paged)

        if len(param_list) == 0:
            self._optim = None
            return

        # Defer bnb import; JIT-loads CUDA libs, opt-in only.
        try:
            from bitsandbytes.optim import (  # type: ignore[import-not-found]
                AdamW8bit,
                PagedAdamW8bit,
            )
        except (ImportError, RuntimeError) as err:
            # Catch both: bnb's JIT-CUDA load can raise RuntimeError, not ImportError.
            raise ImportError(
                "GpuAdamW8bitAdapter requires `bitsandbytes` (>=0.41) for "
                "the 8-bit AdamW kernels. Install via "
                "`pip install bitsandbytes`."
            ) from err

        # bnb 8-bit Adam crashes on CPU params; fail fast at construction.
        for p in param_list:
            if not p.is_cuda:
                raise RuntimeError(
                    "GpuAdamW8bitAdapter received a parameter on device "
                    f"{p.device}; bitsandbytes' 8-bit AdamW kernels run "
                    "on CUDA only. Non-persistent (CPU-resident) chunks "
                    "must continue to use CpuFusedAdamAdapter "
                    "(DeepSpeedCPUAdam) - only persistent (GPU) chunks "
                    "may use the 8-bit adapter."
                )

        cls = PagedAdamW8bit if self.paged else AdamW8bit
        self._optim = cls(
            param_list,
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )

    # ---- step interface -------------------------------------------------

    def step(self) -> None:
        """Synchronous bnb 8-bit AdamW step over persistent-chunk params."""
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
        """Return the wrapped 8-bit optimizer's state dict (empty when no-op)."""
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
                    "GpuAdamW8bitAdapter: this layout has no persistent-chunk "
                    "params but the checkpoint contains optimizer state "
                    "(likely a Mode-A/Mode-C config mismatch on resume)."
                )
            return
        optim.load_state_dict(state_dict)

    @property
    def underlying(self) -> Any:
        """Return the wrapped optimizer (None when adapter has no persistent params)."""
        return self._optim


__all__ = [
    "CpuFusedAdamAdapter",
    "GpuAdamW8bitAdapter",
    "GpuFusedAdamAdapter",
]
