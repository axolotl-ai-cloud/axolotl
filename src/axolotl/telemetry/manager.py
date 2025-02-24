"""Telemetry manager and associated utilities."""

import atexit
import logging
import os
import platform
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import posthog
import psutil
import torch
import transformers
import yaml

import axolotl
from axolotl.utils.distributed import is_main_process

LOG = logging.getLogger(__name__)

POSTHOG_WRITE_KEY = "phc_1kUR0o04oJKKTTeSsIz2Mfm5mpiVsQEf2WOlzljMD7y"
ENABLED_WARNING_SLEEP_SECONDS = 15
ENABLED_WARNING = (
    "\nTelemetry is enabled. This helps Axolotl's maintainers by providing insights into:\n"
    "- Which models and configurations are most commonly used\n"
    "- What hardware setups need to be supported\n"
    "- Where users encounter errors\n\n"
    "This data helps us prioritize features, optimize performance, and fix bugs.\n\n"
    "To disable telemetry, set either:\n"
    "- AXOLOTL_DO_NOT_TRACK=1 (Axolotl-specific)\n"
    "- DO_NOT_TRACK=1 (Global standard)\n\n"
    "To remove this warning and continue with telemetry enabled,"
    "explicitly set AXOLOTL_DO_NOT_TRACK=0 (and leave DO_NOT_TRACK unset / set to 0)\n\n"
    "No personally identifiable information is collected."
    "For details, see: https://axolotl-ai-cloud.github.io/axolotl/docs/telemetry.html\n\n"
    f"Sleeping for {ENABLED_WARNING_SLEEP_SECONDS}s..."
)


@dataclass
class TelemetryConfig:
    """Configuration for telemetry manager"""

    host: str = "https://app.posthog.com"
    queue_size: int = 100
    batch_size: int = 10
    whitelist_path: str = str(Path(__file__).parent / "whitelist.yaml")
    retention_days: int = 365


class TelemetryManager:
    """Manages telemetry collection and transmission"""

    _instance = None
    _initialized = False

    def __new__(cls):
        """
        Telemetry manager constructor. Creates the singleton instance of this class if
        it doesn't already exist.
        """
        if cls._instance is None:
            cls._instance = super(TelemetryManager, cls).__new__(cls)
            cls._instance._initialized = False

        return cls._instance

    def __init__(self):
        """Telemetry manager initializer"""
        if self._initialized:
            return

        self.enabled, self.explicit_enable = self._check_telemetry_enabled()

        if self.enabled:
            # Warn about telemetry collection
            if not self.explicit_enable:
                LOG.warning(ENABLED_WARNING)
                time.sleep(ENABLED_WARNING_SLEEP_SECONDS)

            self.config = TelemetryConfig()
            self.run_id = str(uuid.uuid4())
            self.whitelist = self._load_whitelist()
            self.system_info = self._get_system_info()
            self._init_posthog()

            # Register shutdown method to flush posthog telemetry
            atexit.register(self.shutdown)

        self._initialized = True

    @classmethod
    def get_instance(cls) -> "TelemetryManager":
        if cls._instance is None:
            cls._instance = TelemetryManager()

        return cls._instance

    def _check_telemetry_enabled(self) -> tuple[bool, bool]:
        """
        Check if telemetry is enabled based on environment variables. We also check
        whether this is the main process (for the distributed setting and to avoid
        sending duplicate PostHog events per GPU).

        Note: This is enabled by default on an opt-out basis. Set either
        `AXOLOTL_DO_NOT_TRACK=1` or `DO_NOT_TRACK=1` to disable telemetry. For more
        details, see https://axolotl-ai-cloud.github.io/axolotl/docs/telemetry.html.

        Returns:
            Tuple containing:
                - Boolean denoting whether telemetry is enabled or disabled.
                - Boolean denoting whether telemetry is explicitly enabled or not.
        """
        # In the distributed setting, check whether we're running on rank 0
        if not is_main_process():
            return False, False

        # Parse relevant env vars and fill opt-out default values
        axolotl_do_not_track = os.getenv("AXOLOTL_DO_NOT_TRACK")
        do_not_track = os.getenv("DO_NOT_TRACK")

        # If explicitly enabled, we'll disable the telemetry warning message
        explicit_enabled = axolotl_do_not_track in ["0", "false"]

        if axolotl_do_not_track is None:
            axolotl_do_not_track = "0"

        if do_not_track is None:
            do_not_track = "0"

        # Respect AXOLOTL_DO_NOT_TRACK, DO_NOT_TRACK if enabled
        enabled = axolotl_do_not_track.lower() not in (
            "1",
            "true",
        ) and do_not_track.lower() not in ("1", "true")

        return enabled, explicit_enabled

    def _load_whitelist(self) -> dict:
        """Load organization/model whitelist"""
        with open(self.config.whitelist_path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _is_whitelisted(self, base_model: str) -> bool:
        """Check if model org is in whitelist"""
        if not base_model:
            return False

        base_model = base_model.lower()
        return any(
            org.lower() in base_model for org in self.whitelist.get("organizations", [])
        )

    def _init_posthog(self):
        """Initialize PostHog client"""
        posthog.project_api_key = POSTHOG_WRITE_KEY
        posthog.host = self.config.host

    def _sanitize_properties(self, properties: dict[str, Any]) -> dict[str, Any]:
        """
        Sanitize properties to remove any personally identifiable information such as:
        - File paths
        - URLs / Links
        - Cloud storage locations

        Args:
            properties: Dictionary of properties to sanitize.

        Returns:
            Sanitized properties dictionary.
        """
        if not properties:
            return {}

        # Define regex patterns for different types of personal information
        patterns = {
            # File paths (Unix and Windows)
            "file_path": re.compile(r"(?:/|\\)(?:[^/\\]+(?:/|\\))+[^/\\]+"),
            # URLs/Links
            "url": re.compile(r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?:/[^/\s]*)*"),
            # Cloud storage paths (S3, GCS, Azure)
            "cloud_path": re.compile(r"s3://|gs://|azure://|blob.core.windows.net"),
        }

        # Deep copy isn't needed; we'll create a new dict with sanitized values
        sanitized = {}

        def sanitize_value(value):
            """Recursively sanitize values within nested structures"""
            if isinstance(value, str):
                # For file paths, extract just the filename
                path_match = patterns["file_path"].search(value)
                if path_match:
                    try:
                        # Try to extract just the filename
                        path_str = path_match.group(0)
                        value = value.replace(path_str, Path(path_str).name)
                    except (ValueError, RuntimeError):
                        # If path extraction fails, just redact the path
                        value = patterns["file_path"].sub("[REDACTED_PATH]", value)

                # Redact other sensitive information
                value = patterns["url"].sub("[REDACTED_URL]", value)
                value = patterns["cloud_path"].sub("[REDACTED_CLOUD]", value)

                return value
            if isinstance(value, dict):
                return {k: sanitize_value(v) for k, v in value.items()}
            if isinstance(value, list):
                return [sanitize_value(item) for item in value]

            return value

        # Apply the sanitization to all properties
        for key, value in properties.items():
            sanitized[key] = sanitize_value(value)

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
            "pytorch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "axolotl_version": axolotl.__version__,
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "gpu_count": len(gpu_info),
            "gpu_info": gpu_info,
        }

    def send_event(self, event_type: str, properties: dict[str, Any] | None = None):
        """Send a telemetry event"""
        if not self.enabled:
            return

        if properties is None:
            properties = {}

        # Sanitize properties to remove PII
        properties = self._sanitize_properties(properties)

        # Wrap PostHog errors in try / except to not raise errors during Axolotl usage
        try:
            LOG.warning(f"*** Sending telemetry for {event_type} ***")

            # Send event via PostHog
            posthog.capture(
                distinct_id=self.run_id,
                event=event_type,
                properties=properties,
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            LOG.warning(f"Failed to send telemetry event: {e}")

    def send_system_info(self):
        """Helper method for sending system info"""
        self.send_event(event_type="system-info", properties=self.system_info)

    def shutdown(self):
        """Ensure all queued events are processed before shutdown"""
        if self.enabled:
            posthog.flush()
