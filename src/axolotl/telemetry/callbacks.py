"""Trainer callbacks for reporting runtime metrics at regular intervals."""

import logging
import time

from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from axolotl.telemetry.manager import TelemetryManager
from axolotl.telemetry.runtime_metrics import RuntimeMetricsTracker

LOG = logging.getLogger(__name__)

TIME_SINCE_LAST = 30


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

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,  # pylint: disable=unused-argument
        control: TrainerControl,  # pylint: disable=unused-argument
        **kwargs,  # pylint: disable=unused-argument
    ):
        """Handle training start."""
        self.telemetry_manager.send_event(event_type="train-started")

    def on_train_end(
        self,
        args: TrainingArguments,  # pylint: disable=unused-argument
        state: TrainerState,
        control: TrainerControl,  # pylint: disable=unused-argument
        **kwargs,  # pylint: disable=unused-argument
    ):
        """Handle training end."""
        # Send training completion event
        self.telemetry_manager.send_event(
            event_type="train-ended",
            properties={
                "loss": state.log_history[-1].get("loss", 0)
                if state.log_history
                else None,
                "learning_rate": state.log_history[-1].get("learning_rate", 0)
                if state.log_history
                else None,
            }
            | self.tracker.metrics.to_dict(),
        )

    def on_epoch_begin(
        self,
        args: TrainingArguments,  # pylint: disable=unused-argument
        state: TrainerState,  # pylint: disable=unused-argument
        control: TrainerControl,  # pylint: disable=unused-argument
        **kwargs,  # pylint: disable=unused-argument
    ):
        """Handle epoch start."""
        self.current_epoch += 1
        self.tracker.start_epoch(self.current_epoch)

    def on_epoch_end(
        self,
        args: TrainingArguments,  # pylint: disable=unused-argument
        state: TrainerState,  # pylint: disable=unused-argument
        control: TrainerControl,  # pylint: disable=unused-argument
        **kwargs,  # pylint: disable=unused-argument
    ):
        """Handle epoch end."""
        self.tracker.end_epoch(self.current_epoch)

    def on_step_end(
        self,
        args: TrainingArguments,  # pylint: disable=unused-argument
        state: TrainerState,
        control: TrainerControl,  # pylint: disable=unused-argument
        **kwargs,  # pylint: disable=unused-argument
    ):
        """Handle step end."""
        step = state.global_step
        self.tracker.update_step(step)

        # Check if we should report metrics
        should_report = (
            step % self.report_interval_steps == 0
            or step == 1  # Always report first step
            or step - self.last_report_step >= self.report_interval_steps
        )

        if should_report:
            current_time = time.time()
            if self.last_report_time is not None:
                time_since_last_report = current_time - self.last_report_time
            else:
                time_since_last_report = current_time - self.start_time
            steps_since_last_report = step - self.last_report_step

            # Only report if enough time has passed to avoid flooding
            if (
                step == 1
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
                metrics = {
                    "step": step,
                    "epoch": self.current_epoch,
                    "progress": state.epoch,  # Fractional epoch progress
                    "loss": state.log_history[-1].get("loss", 0)
                    if state.log_history
                    else 0,
                    "learning_rate": state.log_history[-1].get("learning_rate", 0)
                    if state.log_history
                    else 0,
                    "steps_per_second": steps_per_second,
                    "elapsed_time": current_time - self.start_time,
                    "time_since_last_report": time_since_last_report,
                }

                # Add memory metrics
                memory_metrics = self.tracker.get_memory_metrics()
                metrics.update(memory_metrics)

                # Send telemetry
                self.telemetry_manager.send_event(
                    event_type="train-progress", properties=metrics
                )

                # Update last report time and step
                self.last_report_time = current_time
                self.last_report_step = step
