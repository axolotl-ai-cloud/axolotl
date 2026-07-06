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

import os
import time
from dataclasses import dataclass

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


def _extract_scalar_metrics(raw_metrics) -> dict:
    """Keep only scalar int/float values (bools excluded)."""
    if not isinstance(raw_metrics, dict):
        return {}
    return {
        key: value
        for key, value in raw_metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
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
        self.mode = bench.mode or "sync"
        self.poll_interval_steps = bench.poll_interval_steps or 10
        # distinguish an explicit empty list (disable all triggers) from a missing value
        self.run_on = set(bench.run_on if bench.run_on is not None else ["save"])
        self.timeout_sec = bench.timeout_sec
        self.fail_training_on_error = bench.fail_training_on_error
        self._http_timeout = (
            self.timeout_sec
            if self.mode == "sync"
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
            status = self._poll_on_main(args, state)
        return self._finish(status, args, control, "poll")

    def on_train_end(self, args, state, control, **kwargs):
        status = _CONTINUE
        if state.is_world_process_zero:
            if "train_end" in self.run_on:
                status = _merge_status(status, self._submit(args, state, "train_end"))
            if self.mode == "async":
                status = _merge_status(status, self._drain_on_main(args, state))
        return self._finish(status, args, control, "train_end")

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
            status = self._submit(args, state, event)
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

    def _benchmark_on_main(self, event, args, state) -> int:
        """Synchronous benchmark: POST and block for the result."""
        try:
            response = requests.post(
                self.endpoint,
                json=self._payload(event, args, state),
                timeout=self._http_timeout,
            )
            response.raise_for_status()
            result = response.json()
            if not isinstance(result, dict):
                raise ValueError(f"expected a JSON object, got {type(result).__name__}")
        except Exception as exc:  # noqa: BLE001
            return self._on_error("call", event, exc)

        if result.get("status") != "completed":
            LOG.warning(
                f"Benchmark API returned status {result.get('status')!r} "
                f"for {event}; skipping metric logging"
            )
            return _CONTINUE
        return self._process_result(result, event)

    def _submit_on_main(self, event, args, state) -> int:
        """Async submit: POST and either process an immediate result or enqueue."""
        try:
            response = requests.post(
                self.endpoint,
                json=self._payload(event, args, state),
                timeout=self._http_timeout,
            )
            response.raise_for_status()
            result = response.json()
            if not isinstance(result, dict):
                raise ValueError(f"expected a JSON object, got {type(result).__name__}")
        except Exception as exc:  # noqa: BLE001
            return self._on_error("submit", event, exc)

        status = result.get("status")
        if status == "completed":
            return self._process_result(result, event)
        if status in _PENDING_STATUSES:
            job_id = result.get("job_id") or f"{event}-{state.global_step}"
            poll_url = result.get("poll_url") or f"{self.endpoint.rstrip('/')}/{job_id}"
            self._pending.append(
                _PendingJob(
                    job_id=job_id,
                    step=state.global_step,
                    poll_url=poll_url,
                    deadline=time.monotonic() + self.timeout_sec,
                )
            )
            LOG.info(
                f"Benchmark API: queued job {job_id} for {event} "
                f"(step {state.global_step}); {len(self._pending)} pending"
            )
            return _CONTINUE

        LOG.warning(
            f"Benchmark API: unexpected submit status {status!r} for {event}; ignoring"
        )
        return _CONTINUE

    def _poll_on_main(self, args, state) -> int:
        """Poll every pending job once; log completed ones, drop timed-out ones."""
        status = _CONTINUE
        still_pending: list[_PendingJob] = []
        for job in self._pending:
            timed_out = time.monotonic() > job.deadline
            try:
                response = requests.get(job.poll_url, timeout=self._http_timeout)
                response.raise_for_status()
                result = response.json()
                if not isinstance(result, dict):
                    raise ValueError(
                        f"expected a JSON object, got {type(result).__name__}"
                    )
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
                status = _merge_status(status, self._process_result(result, job.job_id))
            elif timed_out:
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

    def _process_result(self, result, label) -> int:
        """Log scalar metrics and apply early stopping for a completed result."""
        metrics = _extract_scalar_metrics(result.get("metrics"))
        if metrics:
            self.trainer.log(metrics)
            LOG.info(f"Benchmark API: logged {len(metrics)} metric(s) for {label}")
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
        LOG.info("Adding Benchmark API callback to the trainer")
        return [BenchmarkAPICallback(cfg, trainer)]
