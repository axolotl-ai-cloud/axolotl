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
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from axolotl.utils.logging import get_logger

from ..base import BasePlugin
from .args import BenchmarkAPIArgs as BenchmarkAPIArgs
from .early_stopping import EarlyStopper

LOG = get_logger(__name__)


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
        self.run_on = set(bench.run_on or ["save"])
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
        # only the main process talks to the benchmark runner and logs metrics
        if not state.is_world_process_zero:
            return control
        if event not in self.run_on:
            return control

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
                raise
            LOG.warning(f"Benchmark API call failed ({event}): {exc}")
            return control

        if result.get("status") != "completed":
            LOG.warning(
                f"Benchmark API returned status {result.get('status')!r} "
                f"for {event}; skipping metric logging"
            )
            return control

        metrics = _extract_scalar_metrics(result.get("metrics"))
        if metrics:
            self.trainer.log(metrics)
            LOG.info(f"Benchmark API: logged {len(metrics)} metric(s) for {event}")

        if self.early_stopper is not None:
            should_stop, reason = self.early_stopper.update(metrics)
            if should_stop:
                LOG.info(f"Benchmark API: early stopping — {reason}")
                control.should_training_stop = True

        return control

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
