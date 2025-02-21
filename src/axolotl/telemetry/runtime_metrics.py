"""Telemetry utilities for runtime and memory metrics."""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import psutil
import torch

from axolotl.telemetry.manager import TelemetryManager

LOG = logging.getLogger(__name__)


@dataclass
class RuntimeMetrics:
    """Container for runtime metrics to be tracked throughout training."""

    # Timing metrics
    start_time: float
    epoch_start_times: dict[int, float] = field(init=False)
    epoch_end_times: dict[int, float] = field(init=False)

    # Memory metrics
    peak_cpu_memory: int = 0
    peak_gpu_memory: dict[int, int] = field(init=False)

    # Progress metrics
    total_steps: int = 0
    current_epoch: int = 0
    current_step: int = 0

    def __post_init__(self):
        """Initialize empty metric mappings."""
        self.epoch_start_times = {}
        self.epoch_end_times = {}
        self.peak_gpu_memory = {}

    @property
    def elapsed_time(self) -> float:
        """Calculate total elapsed time in seconds."""
        return time.time() - self.start_time

    def epoch_time(self, epoch: int) -> float | None:
        """Calculate time taken for a specific epoch in seconds."""
        if epoch in self.epoch_start_times and epoch in self.epoch_end_times:
            return self.epoch_end_times[epoch] - self.epoch_start_times[epoch]

        return None

    def average_epoch_time(self) -> float | None:
        """Calculate average time per epoch in seconds."""
        completed_epochs = [
            epoch for epoch in self.epoch_start_times if epoch in self.epoch_end_times
        ]
        if not completed_epochs:
            return None

        total_time = 0.0
        for epoch in completed_epochs:
            epoch_time = self.epoch_time(epoch)
            if epoch_time is not None:  # Check to avoid mypy warning
                total_time += epoch_time

        return total_time / len(completed_epochs)

    def steps_per_second(self) -> float | None:
        """Calculate average steps per second across all training."""
        if self.total_steps == 0 or self.elapsed_time == 0:
            return None

        return self.total_steps / self.elapsed_time

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to a dictionary for telemetry reporting."""
        metrics = {
            "total_time_seconds": self.elapsed_time,
            "total_steps": self.total_steps,
            "steps_per_second": self.steps_per_second(),
            "epochs_completed": len(
                [
                    epoch
                    for epoch in self.epoch_start_times
                    if epoch in self.epoch_end_times
                ]
            ),
            "peak_cpu_memory_bytes": self.peak_cpu_memory,
        }

        # Add per-epoch timing if available
        epoch_times: dict[str, float] = {}
        for epoch in sorted(self.epoch_end_times.keys()):
            time_taken = self.epoch_time(epoch)
            if time_taken is not None:
                epoch_times[f"epoch_{epoch}_seconds"] = time_taken

        if epoch_times:
            metrics["epoch_times"] = epoch_times  # type: ignore
            metrics["average_epoch_time_seconds"] = self.average_epoch_time()

        # Add GPU memory metrics if available
        if self.peak_gpu_memory:
            gpu_metrics: dict[str, int] = {}
            for gpu_id, memory in self.peak_gpu_memory.items():
                gpu_metrics[f"gpu_{gpu_id}_peak_memory_bytes"] = memory
            metrics["gpu_memory"] = gpu_metrics  # type: ignore

        return metrics


class RuntimeMetricsTracker:
    """Tracker for runtime metrics during training."""

    def __init__(self):
        """Initialize the runtime metrics tracker."""
        self.metrics = RuntimeMetrics(start_time=time.time())
        self.telemetry_manager = TelemetryManager.get_instance()

    def start_epoch(self, epoch: int):
        """Record the start of a new epoch."""
        self.metrics.current_epoch = epoch
        self.metrics.epoch_start_times[epoch] = time.time()
        self.update_memory_metrics()

    def end_epoch(self, epoch: int):
        """Record the end of an epoch."""
        self.metrics.epoch_end_times[epoch] = time.time()

    def update_step(self, step: int):
        """Update the current step count."""
        self.metrics.current_step = step
        self.metrics.total_steps += 1

        # Periodically update memory metrics (e.g., every 100 steps)
        if step % 100 == 0:
            self.update_memory_metrics()

    def update_memory_metrics(self):
        """Update peak memory usage metrics."""
        # CPU memory
        cpu_memory = psutil.Process().memory_info().rss
        self.metrics.peak_cpu_memory = max(self.metrics.peak_cpu_memory, cpu_memory)

        # GPU memory if available
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_used = torch.cuda.memory_allocated(i)
                self.metrics.peak_gpu_memory[i] = max(
                    self.metrics.peak_gpu_memory.get(i, 0), memory_used
                )

    def get_memory_metrics(self) -> dict[str, Any]:
        """Get the current memory metrics as a dictionary."""
        memory_metrics = {
            "cpu_memory_bytes": psutil.Process().memory_info().rss,
            "peak_cpu_memory_bytes": self.metrics.peak_cpu_memory,
        }

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_metrics[f"gpu_{i}_memory_bytes"] = torch.cuda.memory_allocated(i)
                memory_metrics[
                    f"gpu_{i}_peak_memory_bytes"
                ] = self.metrics.peak_gpu_memory.get(i, 0)

        return {"memory": memory_metrics}
