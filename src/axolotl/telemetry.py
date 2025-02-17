"""Telemetry manager and associated utilities."""

import logging
import os
import platform
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any

import posthog
import psutil
import torch

logger = logging.getLogger(__name__)


@dataclass
class TelemetryConfig:
    """Configuration for telemetry system"""

    enabled: bool
    project_api_key: str
    host: str = "https://app.posthog.com"
    queue_size: int = 100
    batch_size: int = 10
    whitelist_path: str = "telemetry_whitelist.yaml"


class TelemetryManager:
    """Manages telemetry collection and transmission"""

    def __init__(self, config: TelemetryConfig):
        """
        Telemetry manager constructor.

        Args:
            config: Telemetry configuration object.
        """
        self.config = config
        self.enabled = self._check_telemetry_enabled()
        self.run_id = str(uuid.uuid4())
        self.event_queue: Queue = Queue(maxsize=config.queue_size)

        if self.enabled:
            self._init_posthog()
            self._start_worker()

    def _check_telemetry_enabled(self) -> bool:
        """Check if telemetry is enabled based on environment variables"""
        if not self.config.enabled:
            return False

        do_not_track = os.getenv("DO_NOT_TRACK", "0").lower() in ("1", "true")
        axolotl_do_not_track = os.getenv("AXOLOTL_DO_NOT_TRACK", "0").lower() in (
            "1",
            "true",
        )

        return not (do_not_track or axolotl_do_not_track)

    def _init_posthog(self):
        """Initialize PostHog client"""
        posthog.project_api_key = self.config.project_api_key
        posthog.host = self.config.host

    def _start_worker(self):
        """Start background worker thread for processing events"""
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()

    def _process_queue(self):
        """Process events from queue and send to PostHog"""
        while True:
            events = []
            # Always get at least one event (blocking)
            events.append(self.event_queue.get())

            # Try to get more events up to batch size (non-blocking)
            remaining_batch = self.config.batch_size - 1
            for _ in range(remaining_batch):
                try:
                    event = self.event_queue.get_nowait()
                    events.append(event)
                except Empty:
                    # No more events available right now
                    break

            if events:
                try:
                    posthog.capture_batch(events)
                except (posthog.RequestError, posthog.RateLimitError) as e:
                    logger.warning(f"Failed to send telemetry batch: {e}")
                except ConnectionError as e:
                    logger.warning(f"Network error while sending telemetry: {e}")
                finally:
                    # Mark tasks as done even if sending failed
                    for _ in range(len(events)):
                        self.event_queue.task_done()

    def _sanitize_path(self, path: str) -> str:
        """Remove personal information from file paths"""
        return Path(path).name

    def _sanitize_error(self, error: str) -> str:
        """Remove personal information from error messages"""
        # Replace file paths with just filename
        sanitized = error
        try:
            for path in Path(error).parents:
                sanitized = sanitized.replace(str(path), "")
        except (ValueError, RuntimeError) as e:
            # ValueError: Invalid path format
            # RuntimeError: Other path parsing errors
            logger.debug(f"Could not parse path in error message: {e}")

        return sanitized

    def _get_system_info(self) -> dict[str, Any]:
        """Collect system information"""
        gpu_info = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_info.append(
                    {
                        "name": torch.cuda.get_device_name(i),
                        "memory": torch.cuda.get_device_properties(i).total_memory,
                    }
                )

        return {
            "os": platform.system(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "gpu_count": len(gpu_info),
            "gpu_info": gpu_info,
        }

    def track_event(self, event_type: str, properties: dict[str, Any]):
        """Track a telemetry event"""
        if not self.enabled:
            return

        try:
            # Get system info first - most likely source of errors
            system_info = self._get_system_info()

            # Construct event dict - could raise TypeError if properties aren't serializable
            event = {
                "event": event_type,
                "properties": {
                    "run_id": self.run_id,
                    "system_info": system_info,
                    **properties,
                },
            }

            try:
                self.event_queue.put_nowait(event)
            except Full:
                logger.warning("Telemetry queue full, dropping event")
        except (RuntimeError, OSError) as e:
            # Hardware info collection errors
            logger.warning(f"Failed to collect system info for telemetry: {e}")
        except TypeError as e:
            # Dict construction/serialization errors
            logger.warning(f"Invalid property type in telemetry event: {e}")
        except AttributeError as e:
            # Missing attributes when collecting system info
            logger.warning(f"Failed to access system attribute for telemetry: {e}")

    @contextmanager
    def track_training(self, config: dict[str, Any]):
        """Context manager to track training run"""
        if not self.enabled:
            yield
            return

        # Track training start
        sanitized_config = {
            k: v
            for k, v in config.items()
            if not any(p in k.lower() for p in ["path", "dir", "file"])
        }

        self.track_event("training_start", {"config": sanitized_config})

        try:
            yield
            # Track successful completion
            self.track_event("training_complete", {})

        except Exception as e:
            # Track error
            self.track_event("training_error", {"error": self._sanitize_error(str(e))})
            raise


def init_telemetry(project_api_key: str, enabled: bool = True) -> TelemetryManager:
    """Initialize telemetry system"""
    config = TelemetryConfig(enabled=enabled, project_api_key=project_api_key)
    return TelemetryManager(config)
