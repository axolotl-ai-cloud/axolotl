"""Trainer callbacks for reporting runtime metrics at regular intervals."""

import time
from typing import Any

from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from axolotl.telemetry.manager import TelemetryManager
from axolotl.telemetry.runtime_metrics import RuntimeMetrics, RuntimeMetricsTracker
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

TIME_SINCE_LAST = 60
TRAINING_METRICS = {
    "loss",
    "ppl",
    "learning_rate",
    "grad_norm",
    "tokens/total",
    "tokens/trainable",
    "tokens/train_per_sec_per_gpu",
}


class TelemetryCallback(TrainerCallback):
    """
    Trainer callback for tracking and reporting runtime metrics.

    This callback tracks training progress, runtime, and memory usage,
    sending telemetry at configurable intervals.
    """

    report_interval_steps: int = 100

    def __init__(self):
        """Initialize the metrics callback."""
        self.tracker = RuntimeMetricsTracker()
        self.telemetry_manager = TelemetryManager.get_instance()
        self.current_epoch = -1
        self.start_time = time.time()
        self.last_report_time = None
        self.last_report_step = 0
        self.start_step = 0
        self.latest_metrics: dict[str, Any] = {}
        self.metric_steps: dict[str, int] = {}
        self.aggregate_metrics: dict[str, Any] = {}

    # pylint: disable=unused-argument
    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Handle training start."""
        self.start_time = time.time()
        self.start_step = state.global_step
        self.last_report_step = self.start_step
        self.last_report_time = self.start_time
        self.current_epoch = int(state.epoch or 0)
        self.tracker.metrics = RuntimeMetrics(start_time=self.start_time)
        self.latest_metrics = {}
        self.metric_steps = {}
        self.aggregate_metrics = {}
        self.telemetry_manager.send_event(event_type="train-start")

    # pylint: disable=unused-argument
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Handle training end."""
        self.tracker.update_memory_metrics()
        self.telemetry_manager.send_event(
            event_type="train-end",
            properties=self.tracker.metrics.to_dict()
            | {
                "step": state.global_step,
                "start_step": self.start_step,
                "aggregate": self.aggregate_metrics.copy(),
                "latest_step": self.latest_metrics
                | {"metric_steps": self.metric_steps.copy()},
            },
        )

    # pylint: disable=unused-argument
    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Handle epoch start."""
        self.current_epoch = int(state.epoch or 0)
        self.tracker.start_epoch(self.current_epoch)

    # pylint: disable=unused-argument
    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Handle epoch end."""
        self.tracker.end_epoch(self.current_epoch)

    # pylint: disable=unused-argument
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Handle step end."""
        step = state.global_step
        self.tracker.update_step(step)

        # Check if we should report metrics
        should_report = (
            step % self.report_interval_steps == 0
            or step == self.start_step + 1
            or step - self.last_report_step >= self.report_interval_steps
        )

        if should_report:
            current_time = time.time()
            if self.last_report_time is not None:
                time_since_last_report = current_time - self.last_report_time
            else:
                time_since_last_report = current_time - self.start_time
            steps_since_last_report = step - self.last_report_step

            # Only report if enough time has passed
            if (
                step == self.start_step + 1
                or time_since_last_report >= TIME_SINCE_LAST
                or steps_since_last_report >= self.report_interval_steps
            ):
                # Calculate steps per second for this interval
                if time_since_last_report > 0 and steps_since_last_report > 0:
                    steps_per_second = steps_since_last_report / time_since_last_report
                else:
                    steps_per_second = 0

                # Update memory metrics
                self.tracker.update_memory_metrics()

                # Prepare metrics to report
                metrics = self.latest_metrics | {
                    "metric_steps": self.metric_steps.copy(),
                    "start_step": self.start_step,
                    "step": step,
                    "epoch": self.current_epoch,
                    "progress": state.epoch,  # Fractional epoch progress
                    "steps_per_second": steps_per_second,
                    "elapsed_time": current_time - self.start_time,
                    "time_since_last_report": time_since_last_report,
                }

                # Add memory metrics
                memory_metrics = self.tracker.get_memory_metrics()
                metrics.update({"memory": memory_metrics})

                # Send telemetry
                self.telemetry_manager.send_event(
                    event_type="train-progress", properties=metrics
                )

                # Update last report time and step
                self.last_report_time = current_time
                self.last_report_step = step

    def on_prediction_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Capture evaluation peaks before the memory logger resets them."""
        self.tracker.update_gpu_memory_metrics()

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict | None = None,
        **kwargs,
    ):
        """Record training metrics with the step at which they were logged."""
        if not logs:
            return
        if logs.get("train_loss") is not None:
            self.aggregate_metrics["train_loss"] = logs["train_loss"]
        for key in TRAINING_METRICS.intersection(logs):
            if logs[key] is not None:
                self.latest_metrics[key] = logs[key]
                self.metric_steps[key] = state.global_step
