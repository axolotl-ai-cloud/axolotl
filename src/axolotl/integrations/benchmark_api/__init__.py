"""
Plugin that forwards checkpoint events to an external benchmark runner and
logs the returned scalar metrics back into the trainer.

The plugin is intentionally generic: it knows nothing about the benchmark
tasks, datasets, model format, or metric semantics. It POSTs a minimal
checkpoint payload to a user-configured HTTP endpoint and logs whatever scalar
metrics come back, optionally applying single-metric early stopping.

Two modes:
  - sync (default): the POST blocks until the runner returns metrics.
  - async: the runner replies ``queued`` immediately; the plugin tracks the job
    and polls it (on ``on_step_end`` every ``poll_interval_steps`` and, finally,
    a blocking drain at ``on_train_end``). This keeps training moving while an
    expensive benchmark runs in the background.
"""

import ipaddress
import math
import os
import time
from dataclasses import dataclass
from urllib.parse import urlparse

import requests
import torch
import torch.distributed as dist
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from axolotl.utils.logging import get_logger

from ..base import BasePlugin
from .args import BenchmarkAPIArgs as BenchmarkAPIArgs
from .early_stopping import EarlyStopper

LOG = get_logger(__name__)

# status codes broadcast from the main process to all ranks
_CONTINUE = 0
_STOP = 1
_ERROR = 2

# short read timeout for async submit/poll calls (timeout_sec is the job deadline)
_ASYNC_HTTP_TIMEOUT = 30.0
# wait between poll rounds while draining outstanding jobs at train end
_DRAIN_POLL_SLEEP_SEC = 2.0

# runner statuses that mean "not done yet"
_PENDING_STATUSES = {"queued", "running", "accepted", "pending"}


def _is_offline_mode() -> bool:
    """True when HF offline mode is requested (same convention as cli/checks.py)."""
    return any(
        os.getenv(var, "").upper() in ("1", "ON", "YES", "TRUE")
        for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )


def _is_local_endpoint(endpoint: str) -> bool:
    """True for loopback/0.0.0.0 endpoints, which stay on the local machine."""
    host = urlparse(endpoint).hostname or ""
    if host in ("localhost", "0.0.0.0"):  # nosec B104 - matching, not binding
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _parse_json_object(response) -> dict:
    """Validate a runner response: no redirects, 2xx status, JSON object body."""
    # with allow_redirects=False a 3xx lands here; following it would defeat the origin pin
    if 300 <= response.status_code < 400:
        raise ValueError(
            f"refusing to follow redirect to {response.headers.get('Location')!r}"
        )
    response.raise_for_status()
    result = response.json()
    if not isinstance(result, dict):
        raise ValueError(f"expected a JSON object, got {type(result).__name__}")
    return result


def _extract_scalar_metrics(raw_metrics) -> dict:
    """Keep only finite scalar int/float values (bools and nan/inf excluded)."""
    if not isinstance(raw_metrics, dict):
        return {}
    return {
        key: value
        for key, value in raw_metrics.items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    }


@dataclass
class _PendingJob:
    """An async benchmark job awaiting completion (tracked on rank 0 only)."""

    job_id: str
    step: int
    poll_url: str
    deadline: float  # time.monotonic() past which the job is considered timed out


class BenchmarkAPICallback(TrainerCallback):
    """
    Sends checkpoint events to an external benchmark API and logs the results.
    """

    def __init__(self, cfg, trainer):
        bench = cfg.benchmark_api
        self.trainer = trainer
        self.endpoint = bench.endpoint
        self.mode = bench.execution_mode or "sync"
        self.poll_interval_steps = bench.poll_interval_steps or 10
        # distinguish an explicit empty list (disable all triggers) from a missing value
        self.run_on = set(bench.run_on if bench.run_on is not None else ["save"])
        self.timeout_sec = bench.timeout_sec
        self.fail_training_on_error = bench.fail_training_on_error
        self._headers = None
        if bench.auth_env:
            token = os.environ.get(bench.auth_env, "")
            if not token:
                raise ValueError(
                    f"benchmark_api.auth_env names {bench.auth_env!r} but that "
                    "environment variable is unset or empty"
                )
            self._headers = {"Authorization": f"Bearer {token}"}
        # timeout_sec == 0 disables the timeout: sync calls block indefinitely
        # (requests uses timeout=None) and async jobs never expire. Async
        # submit/poll calls still use the short _ASYNC_HTTP_TIMEOUT so a dead
        # runner can't wedge the training step.
        no_timeout = self.timeout_sec == 0
        if self.mode == "sync" and no_timeout and int(os.getenv("WORLD_SIZE", "1")) > 1:
            LOG.warning(
                "benchmark_api: sync mode with timeout_sec=0 blocks rank 0 "
                "indefinitely and can exceed the collective watchdog under "
                "multi-GPU; prefer async or a finite timeout_sec"
            )
        if self.mode == "sync":
            self._http_timeout = None if no_timeout else self.timeout_sec
        else:
            self._http_timeout = (
                _ASYNC_HTTP_TIMEOUT
                if no_timeout
                else min(_ASYNC_HTTP_TIMEOUT, self.timeout_sec)
            )
        # async jobs awaiting completion; only ever populated on the main process
        self._pending: list[_PendingJob] = []

        self.early_stopper = None
        es = bench.early_stopping
        if es and es.enabled:
            self.early_stopper = EarlyStopper(
                metric=es.metric,
                mode=es.mode,
                patience=es.patience,
                min_delta=es.min_delta,
                threshold=es.threshold,
            )

    # ------------------------------------------------------------------ #
    # trainer callback hooks
    # ------------------------------------------------------------------ #

    def on_save(self, args, state, control, **kwargs):
        return self._trigger("save", args, state, control)

    def on_evaluate(self, args, state, control, **kwargs):
        return self._trigger("eval", args, state, control)

    def on_step_end(self, args, state, control, **kwargs):
        # poll outstanding async jobs on a step-based cadence so every rank
        # reaches the broadcast in lockstep (global_step is identical everywhere)
        if self.mode != "async":
            return control
        if state.global_step % self.poll_interval_steps != 0:
            return control
        status = _CONTINUE
        if state.is_world_process_zero:
            status = self._safe_main(lambda: self._poll_on_main(args, state), "poll")
        return self._finish(status, args, control, "poll")

    def on_train_end(self, args, state, control, **kwargs):
        # Training is over: the stop decision is moot and no other rank is waiting
        # on a step/gradient collective, so the (possibly long) drain runs on the
        # main process only, without a broadcast that would otherwise force the
        # workers to wait on rank 0 past the collective watchdog. Errors raise
        # locally on rank 0, which is sufficient to signal a failed run at the end.
        if not state.is_world_process_zero:
            return control
        status = _CONTINUE
        if "train_end" in self.run_on:
            status = _merge_status(
                status,
                self._safe_main(
                    lambda: self._submit(args, state, "train_end"), "train_end"
                ),
            )
        if self.mode == "async":
            status = _merge_status(
                status,
                self._safe_main(lambda: self._drain_on_main(args, state), "drain"),
            )
        if status == _ERROR:
            raise RuntimeError(
                "Benchmark API call failed (train_end) and fail_training_on_error is set"
            )
        return control

    # ------------------------------------------------------------------ #
    # dispatch / distributed sync
    # ------------------------------------------------------------------ #

    def _trigger(self, event, args, state, control):
        # `event in run_on` and the sync/error flags are config-derived, so every
        # rank agrees on control flow and reaches the same collective (no deadlock).
        if event not in self.run_on:
            return control
        status = _CONTINUE
        if state.is_world_process_zero:
            status = self._safe_main(lambda: self._submit(args, state, event), event)
        return self._finish(status, args, control, event)

    def _submit(self, args, state, event) -> int:
        """Run (sync) or enqueue (async) a benchmark for a trigger event."""
        if self.mode == "async":
            return self._submit_on_main(event, args, state)
        return self._benchmark_on_main(event, args, state)

    def _finish(self, status, args, control, event):
        # the stop/error decision is made on rank 0 but must apply on every rank,
        # or the workers desync at the next collective
        if self.early_stopper is not None or self.fail_training_on_error:
            status = self._sync_status(status, args)
        if status == _ERROR:
            raise RuntimeError(
                f"Benchmark API call failed ({event}) and fail_training_on_error is set"
            )
        if status == _STOP:
            control.should_training_stop = True
        return control

    @staticmethod
    def _sync_status(status: int, args) -> int:
        """Broadcast rank 0's status code to all ranks (identity if not distributed)."""
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            flag = torch.tensor([status], device=args.device)
            dist.broadcast(flag, src=0)
            return int(flag.item())
        return status

    def _safe_main(self, fn, label) -> int:
        """Run main-process work, converting any unexpected error into a status.

        Guarantees the caller still reaches the broadcast in ``_finish`` even if
        e.g. ``trainer.log`` raises, so the other ranks never hang on the
        collective waiting for a rank 0 that died early.
        """
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            return self._on_error("callback", label, exc)

    # ------------------------------------------------------------------ #
    # main-process work (never raises; returns a status code)
    # ------------------------------------------------------------------ #

    def _payload(self, event, args, state) -> dict:
        return {
            "event": event,
            "step": state.global_step,
            "checkpoint_dir": self._resolve_checkpoint_dir(args, state),
            "output_dir": args.output_dir,
        }

    def _resolve_checkpoint_dir(self, args, state) -> str:
        candidate = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )
        return candidate if os.path.isdir(candidate) else args.output_dir

    def _resolve_poll_url(self, poll_url, job_id) -> str:
        """Trust a runner-supplied poll_url only if it shares the endpoint origin.

        The plugin GETs this URL, so a runner that returns a poll_url pointing at
        a different scheme/host/port (SSRF) is rejected in favor of the URL built
        from the configured endpoint.
        """
        fallback = f"{self.endpoint.rstrip('/')}/{job_id}"
        if not poll_url:
            return fallback
        endpoint = urlparse(self.endpoint)
        candidate = urlparse(poll_url)
        if (candidate.scheme, candidate.hostname, candidate.port) != (
            endpoint.scheme,
            endpoint.hostname,
            endpoint.port,
        ):
            LOG.warning(
                f"Benchmark API: poll_url {poll_url!r} origin differs from the "
                f"configured endpoint; using {fallback} instead"
            )
            return fallback
        return poll_url

    def _benchmark_on_main(self, event, args, state) -> int:
        """Synchronous benchmark: POST and block for the result."""
        try:
            response = requests.post(
                self.endpoint,
                json=self._payload(event, args, state),
                headers=self._headers,
                timeout=self._http_timeout,
                allow_redirects=False,
            )
            result = _parse_json_object(response)
        except Exception as exc:  # noqa: BLE001
            return self._on_error("call", event, exc)

        if result.get("status") != "completed":
            LOG.warning(
                f"Benchmark API returned status {result.get('status')!r} "
                f"for {event}; skipping metric logging"
            )
            return _CONTINUE
        return self._process_result(result, event, state.global_step)

    def _submit_on_main(self, event, args, state) -> int:
        """Async submit: POST and either process an immediate result or enqueue."""
        try:
            response = requests.post(
                self.endpoint,
                json=self._payload(event, args, state),
                headers=self._headers,
                timeout=self._http_timeout,
                allow_redirects=False,
            )
            result = _parse_json_object(response)
        except Exception as exc:  # noqa: BLE001
            return self._on_error("submit", event, exc)

        status = result.get("status")
        if status == "completed":
            return self._process_result(result, event, state.global_step)
        if status in _PENDING_STATUSES:
            job_id = result.get("job_id") or f"{event}-{state.global_step}"
            poll_url = self._resolve_poll_url(result.get("poll_url"), job_id)
            self._pending.append(
                _PendingJob(
                    job_id=job_id,
                    step=state.global_step,
                    poll_url=poll_url,
                    deadline=(
                        math.inf
                        if self.timeout_sec == 0
                        else time.monotonic() + self.timeout_sec
                    ),
                )
            )
            LOG.info(
                f"Benchmark API: queued job {job_id} for {event} "
                f"(step {state.global_step}); {len(self._pending)} pending"
            )
            return _CONTINUE

        return self._on_error(
            "submit", event, ValueError(f"unexpected submit status {status!r}")
        )

    def _poll_on_main(self, args, state) -> int:
        """Poll every pending job once; log completed ones, drop timed-out ones."""
        status = _CONTINUE
        still_pending: list[_PendingJob] = []
        for job in self._pending:
            timed_out = time.monotonic() > job.deadline
            try:
                response = requests.get(
                    job.poll_url,
                    headers=self._headers,
                    timeout=self._http_timeout,
                    allow_redirects=False,
                )
                result = _parse_json_object(response)
            except Exception as exc:  # noqa: BLE001
                # transient poll error: retry next round unless past the deadline
                if timed_out:
                    status = _merge_status(
                        status, self._on_error("poll", job.job_id, exc)
                    )
                else:
                    LOG.warning(
                        f"Benchmark API: poll error for {job.job_id}, will retry: {exc}"
                    )
                    still_pending.append(job)
                continue

            job_status = result.get("status")
            if job_status == "completed":
                status = _merge_status(
                    status, self._process_result(result, job.job_id, job.step)
                )
            elif job_status in _PENDING_STATUSES:
                if timed_out:
                    status = _merge_status(
                        status,
                        self._on_error(
                            "timeout",
                            job.job_id,
                            TimeoutError(f"job {job.job_id} exceeded timeout_sec"),
                        ),
                    )
                else:
                    still_pending.append(job)
            else:
                # unexpected/failed status: stop waiting on this job
                status = _merge_status(
                    status,
                    self._on_error(
                        "poll",
                        job.job_id,
                        ValueError(f"unexpected status {job_status!r}"),
                    ),
                )
        self._pending = still_pending
        return status

    def _drain_on_main(self, args, state) -> int:
        """Block until all pending jobs complete or time out."""
        status = _CONTINUE
        if self._pending:
            LOG.info(f"Benchmark API: draining {len(self._pending)} pending job(s)")
        while self._pending:
            status = _merge_status(status, self._poll_on_main(args, state))
            if self._pending:
                time.sleep(_DRAIN_POLL_SLEEP_SEC)
        return status

    def _process_result(self, result, label, step) -> int:
        """Log scalar metrics and apply early stopping for a completed result."""
        metrics = _extract_scalar_metrics(result.get("metrics"))
        if metrics:
            # Log the values through the axolotl logger first: at train-end drain
            # the trainer's printer/reporting callbacks have already closed, so
            # trainer.log alone would silently drop them. Hand trainer.log a copy
            # since it augments the dict in place (epoch, memory stats, ...).
            LOG.info(f"Benchmark API metrics for {label} (step {step}): {metrics}")
            self.trainer.log(dict(metrics))
        if self.early_stopper is not None:
            should_stop, reason = self.early_stopper.update(metrics)
            if should_stop:
                LOG.info(f"Benchmark API: early stopping — {reason}")
                return _STOP
        return _CONTINUE

    def _on_error(self, phase, label, exc) -> int:
        if self.fail_training_on_error:
            LOG.error(f"Benchmark API {phase} failed ({label}): {exc}")
            return _ERROR
        LOG.warning(f"Benchmark API {phase} failed ({label}): {exc}")
        return _CONTINUE


def _merge_status(a: int, b: int) -> int:
    # _ERROR (2) > _STOP (1) > _CONTINUE (0)
    return max(a, b)


class BenchmarkAPIPlugin(BasePlugin):
    """
    Registers the benchmark API callback with the trainer.
    """

    def get_input_args(self):
        return "axolotl.integrations.benchmark_api.args.BenchmarkAPIArgs"

    def add_callbacks_post_trainer(self, cfg, trainer):
        if not cfg.benchmark_api or not cfg.benchmark_api.endpoint:
            return []
        if _is_offline_mode() and not _is_local_endpoint(cfg.benchmark_api.endpoint):
            LOG.warning(
                "Offline mode is set (HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE) and "
                f"benchmark_api.endpoint {cfg.benchmark_api.endpoint!r} is not "
                "local; disabling the benchmark API callback"
            )
            return []
        LOG.info("Adding Benchmark API callback to the trainer")
        return [BenchmarkAPICallback(cfg, trainer)]
