"""
Plugin that forwards checkpoint events to an external benchmark runner and
logs the returned scalar metrics back into the trainer.

The plugin is intentionally generic: it knows nothing about the benchmark
tasks, datasets, model format, or metric semantics. It POSTs a minimal
checkpoint payload to a user-configured HTTP endpoint and logs whatever scalar
metrics come back, optionally applying single-metric early stopping.
"""

import os

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


def _extract_scalar_metrics(raw_metrics) -> dict:
    """Keep only scalar int/float values (bools excluded)."""
    if not isinstance(raw_metrics, dict):
        return {}
    return {
        key: value
        for key, value in raw_metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


class BenchmarkAPICallback(TrainerCallback):
    """
    Sends checkpoint events to an external benchmark API and logs the results.
    """

    def __init__(self, cfg, trainer):
        bench = cfg.benchmark_api
        self.trainer = trainer
        self.endpoint = bench.endpoint
        # distinguish an explicit empty list (disable all triggers) from a missing value
        self.run_on = set(bench.run_on if bench.run_on is not None else ["save"])
        self.timeout_sec = bench.timeout_sec
        self.fail_training_on_error = bench.fail_training_on_error

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

    def _resolve_checkpoint_dir(self, args, state) -> str:
        candidate = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )
        return candidate if os.path.isdir(candidate) else args.output_dir

    def _run_benchmark(self, event, args, state, control):
        # `event in run_on` and the sync/error flags are config-derived, so every
        # rank agrees on control flow and reaches the same collective (no deadlock).
        if event not in self.run_on:
            return control

        # only the main process talks to the benchmark runner and logs metrics
        status = _CONTINUE
        if state.is_world_process_zero:
            status = self._benchmark_on_main(event, args, state)

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

    def _benchmark_on_main(self, event, args, state) -> int:
        """Run the benchmark on rank 0 and return a status code; never raises."""
        payload = {
            "event": event,
            "step": state.global_step,
            "checkpoint_dir": self._resolve_checkpoint_dir(args, state),
            "output_dir": args.output_dir,
        }

        try:
            LOG.info(
                f"Benchmark API: posting {event} (step {state.global_step}) "
                f"to {self.endpoint}"
            )
            response = requests.post(
                self.endpoint, json=payload, timeout=self.timeout_sec
            )
            response.raise_for_status()
            result = response.json()
            if not isinstance(result, dict):
                raise ValueError(f"expected a JSON object, got {type(result).__name__}")
        except Exception as exc:  # noqa: BLE001
            if self.fail_training_on_error:
                LOG.error(f"Benchmark API call failed ({event}): {exc}")
                return _ERROR
            LOG.warning(f"Benchmark API call failed ({event}): {exc}")
            return _CONTINUE

        if result.get("status") != "completed":
            LOG.warning(
                f"Benchmark API returned status {result.get('status')!r} "
                f"for {event}; skipping metric logging"
            )
            return _CONTINUE

        metrics = _extract_scalar_metrics(result.get("metrics"))
        if metrics:
            self.trainer.log(metrics)
            LOG.info(f"Benchmark API: logged {len(metrics)} metric(s) for {event}")

        if self.early_stopper is not None:
            should_stop, reason = self.early_stopper.update(metrics)
            if should_stop:
                LOG.info(f"Benchmark API: early stopping — {reason}")
                return _STOP

        return _CONTINUE

    @staticmethod
    def _sync_status(status: int, args) -> int:
        """Broadcast rank 0's status code to all ranks (identity if not distributed)."""
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            flag = torch.tensor([status], device=args.device)
            dist.broadcast(flag, src=0)
            return int(flag.item())
        return status

    def on_save(self, args, state, control, **kwargs):
        return self._run_benchmark("save", args, state, control)

    def on_evaluate(self, args, state, control, **kwargs):
        return self._run_benchmark("eval", args, state, control)

    def on_train_end(self, args, state, control, **kwargs):
        return self._run_benchmark("train_end", args, state, control)


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
