"""Telemetry utilities for runtime and memory metrics."""

import time
from dataclasses import dataclass, field
from typing import Any

import psutil
import torch

from axolotl.telemetry.manager import TelemetryManager
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


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
    gpu_peak_memory_source: dict[int, str] = field(default_factory=dict)

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
            gpu_metrics: dict[str, Any] = {}
            for gpu_id, memory in self.peak_gpu_memory.items():
                gpu_metrics[f"gpu_{gpu_id}_peak_memory_bytes"] = memory
                gpu_metrics[f"gpu_{gpu_id}_peak_memory_source"] = (
                    self.gpu_peak_memory_source.get(gpu_id, "sampled")
                )
            metrics["gpu_memory"] = gpu_metrics  # type: ignore

        return metrics


class RuntimeMetricsTracker:
    """Tracker for runtime metrics during training."""

    update_interval = 100

    def __init__(self):
        """Initialize the runtime metrics tracker."""
        self.metrics = RuntimeMetrics(start_time=time.time())
        self.telemetry_manager = TelemetryManager.get_instance()
        self._process = psutil.Process()

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

        # Periodically update memory metrics
        if step % self.update_interval == 0:
            self.update_memory_metrics()
        else:
            # Trainer logging resets allocator peaks after on_step_end.
            self.update_gpu_memory_metrics()

    def _get_memory_backend(self):
        """Select the same accelerator for current and peak allocation queries."""
        for name in ("cuda", "hip"):
            backend = getattr(torch, name, None)
            if backend is not None and backend.is_available():
                return backend
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.mps
        for name in ("xpu", "npu"):
            backend = getattr(torch, name, None)
            if backend is not None and backend.is_available():
                return backend
        return None

    def _get_allocated_memory(self) -> dict[int, int]:
        """Get current allocations by accelerator device ID."""
        backend = self._get_memory_backend()
        if backend is None:
            return {}
        if backend is getattr(torch, "mps", None):
            if hasattr(backend, "current_allocated_memory"):
                return {0: backend.current_allocated_memory()}
            return {}
        if not hasattr(backend, "memory_allocated"):
            return {}
        return {i: backend.memory_allocated(i) for i in range(backend.device_count())}

    def update_memory_metrics(self):
        """Update peak memory usage metrics."""
        # CPU memory
        cpu_memory = self._process.memory_info().rss
        self.metrics.peak_cpu_memory = max(self.metrics.peak_cpu_memory, cpu_memory)

        self.update_gpu_memory_metrics()

    def update_gpu_memory_metrics(self):
        """Accumulate allocator high-water marks, falling back to samples."""
        memory_used = self._get_allocated_memory()
        backend = self._get_memory_backend()
        peak_memory = getattr(backend, "max_memory_allocated", None)
        for i, memory in memory_used.items():
            if peak_memory is not None:
                memory = max(memory, peak_memory(i))
            self.metrics.gpu_peak_memory_source[i] = (
                "allocator_high_water_mark" if peak_memory is not None else "sampled"
            )
            self.metrics.peak_gpu_memory[i] = max(
                self.metrics.peak_gpu_memory.get(i, 0), memory
            )

    def get_memory_metrics(self) -> dict[str, Any]:
        """Get the current memory metrics as a dictionary."""
        memory_metrics: dict[str, Any] = {
            "cpu_memory_bytes": self._process.memory_info().rss,
            "peak_cpu_memory_bytes": self.metrics.peak_cpu_memory,
        }

        # GPU memory (if available)
        memory_used = self._get_allocated_memory()
        for i, memory in memory_used.items():
            memory_metrics[f"gpu_{i}_memory_bytes"] = memory
            memory_metrics[f"gpu_{i}_peak_memory_bytes"] = (
                self.metrics.peak_gpu_memory.get(i, 0)
            )
            memory_metrics[f"gpu_{i}_peak_memory_source"] = (
                self.metrics.gpu_peak_memory_source.get(i, "sampled")
            )

        return memory_metrics
