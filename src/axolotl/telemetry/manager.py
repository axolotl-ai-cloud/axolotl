"""Telemetry manager and associated utilities."""

import atexit
import importlib
import logging
import os
import platform
import time
import uuid
from pathlib import Path
from typing import Any

import posthog
import psutil
import torch
import yaml

LOG = logging.getLogger(__name__)

POSTHOG_HOST = "https://app.posthog.com"
POSTHOG_WRITE_KEY = "phc_1kUR0o04oJKKTTeSsIz2Mfm5mpiVsQEf2WOlzljMD7y"

OPT_OUT_WARNING_SLEEP_SECONDS = 10
OPT_OUT_WARNING = (
    "\nTelemetry is now enabled by default to help improve Axolotl. "
    "If you'd like to disable it, set AXOLOTL_DO_NOT_TRACK=1 in your environment.\n\n"
    "Telemetry data helps us understand:\n"
    "- Which features are most used\n"
    "- What hardware configurations to prioritize\n"
    "- Where users encounter errors\n\n"
    "Personally identifiable information (PII) is not collected.\n\n"
    "To remove this warning, explicitly set AXOLOTL_DO_NOT_TRACK=0 (enable telemetry) "
    "or AXOLOTL_DO_NOT_TRACK=1 (disable telemetry).\n\n"
    "For details, see: https://docs.axolotl.ai/docs/telemetry.html\n\n"
    f"Sleeping for {OPT_OUT_WARNING_SLEEP_SECONDS}s..."
)

WHITELIST_PATH = str(Path(__file__).parent / "whitelist.yaml")

# NOTE: Need to keep these up to date with any config schema changes
FIELDS_TO_REDACT = {
    "base_model",
    "tokenizer_config",
    "base_model_config",
    "pretraining_dataset",  # NOTE: this field may be a string or a dictionary
    "resume_from_checkpoint",
    "hub_model_id",
}
PREFIXES_TO_REDACT = {"wandb_", "comet_", "mlflow_", "gradio_"}
PATH_INDICATORS = {"path", "dir"}

# pylint: disable=duplicate-code
RELEVANT_PACKAGES = {
    "torch",
    "transformers",
    "trl",
    "datasets",
    "peft",
    "bitsandbytes",
    "accelerate",
    "optimum",
    "deepspeed",
    "ray",
    "axolotl",
    "triton",
    "mamba-ssm",
    "flash-attn",
    "xformers",
    "autoawq",
    "tokenizers",
    "sentencepiece",
    "torchao",
    "lm_eval",
}


def is_main_process() -> bool:
    """
    Check whether we're running in the main process.

    Note:
        We're using this function instead of `torch.utils.distributed.is_main_process`
        causes issues with DeepSpeed world_size since. This function avoids that issue
        by checking env vars that are set by various launchers.

    Returns:
        Whether we're running in the main process.
    """
    # If PyTorch distributed is already initialized, use it
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0

    # Otherwise check environment variables for global rank
    # NOTE: need to verify this in SLURM / OpenMPI environments
    global_rank = int(
        os.environ.get(
            "RANK",
            os.environ.get(
                "GLOBAL_RANK",
                os.environ.get(
                    "SLURM_PROCID",
                    os.environ.get(
                        "OMPI_COMM_WORLD_RANK",
                        "0",
                    ),
                ),
            ),
        )
    )

    return global_rank == 0


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

        self.enabled = self._check_telemetry_enabled()

        if self.enabled:
            self.run_id = str(uuid.uuid4())
            self.whitelist = self._load_whitelist()

            try:
                self.system_info = self._get_system_info()
            except Exception as e:  # pylint: disable=broad-exception-caught
                LOG.warning(f"Error during system info collection: {e}")
                self.system_info = None

            self._init_posthog()

            # Register shutdown method to flush posthog telemetry
            atexit.register(self.shutdown)

        self._initialized = True

    @classmethod
    def get_instance(cls) -> "TelemetryManager":
        if cls._instance is None:
            cls._instance = TelemetryManager()

        return cls._instance

    def _check_telemetry_enabled(self) -> bool:
        """
        Check if telemetry is enabled based on environment variables. We also check
        whether this is the main process (for the distributed setting and to avoid
        sending duplicate PostHog events per GPU).

        Note: This is enabled by default on an opt-out basis. Set
        `AXOLOTL_DO_NOT_TRACK=1` to disable telemetry. For more details, see
        https://axolotl-ai-cloud.github.io/axolotl/docs/telemetry.html.

        Returns:
            Boolean denoting whether telemetry is enabled or not.
        """
        # Parse relevant env vars
        axolotl_do_not_track = os.getenv("AXOLOTL_DO_NOT_TRACK")
        do_not_track = os.getenv("DO_NOT_TRACK")

        # Default to enabled (opt-out model)
        if axolotl_do_not_track is None or axolotl_do_not_track.lower() not in (
            "0",
            "1",
            "false",
            "true",
        ):
            # Print opt-out info message for main process only
            if is_main_process():
                LOG.warning(OPT_OUT_WARNING)
            time.sleep(OPT_OUT_WARNING_SLEEP_SECONDS)

            return True

        # Only rank 0 will send telemetry
        if not is_main_process():
            return False

        if do_not_track is None:
            do_not_track = "0"

        # Respect AXOLOTL_DO_NOT_TRACK, DO_NOT_TRACK if enabled
        enabled = axolotl_do_not_track.lower() not in (
            "1",
            "true",
        ) and do_not_track.lower() not in ("1", "true")

        return enabled

    def _load_whitelist(self) -> dict:
        """Load HuggingFace Hub organization whitelist"""
        with open(WHITELIST_PATH, encoding="utf-8") as f:
            whitelist = yaml.safe_load(f)

            # Send org strings to lowercase since model names are case insensitive
            whitelist["organizations"] = {
                org.lower() for org in whitelist["organizations"]
            }

            return whitelist

    def _is_whitelisted(self, value: str) -> bool:
        """
        Check if model / dataset / etc. org is in whitelist.

        Args:
            value: Value for one of `axolotl.telemetry.manager.FIELDS_WITH_ORGS`
                ("base_model", etc.).

        Returns:
            Boolean indicating whitelist membership.
        """
        # NOTE: This membership-checking logic can be improved.
        # What happens when a local model path matches a whitelisted org?
        parts = value.split("/")
        if len(parts) < 2:
            return False
        org = parts[0]
        whitelisted = org.lower() in self.whitelist["organizations"]

        return whitelisted

    def _init_posthog(self):
        """Initialize PostHog client"""
        posthog.api_key = POSTHOG_WRITE_KEY
        posthog.project_api_key = POSTHOG_WRITE_KEY
        posthog.host = POSTHOG_HOST

    def _redact_paths(self, properties: dict[str, Any]) -> dict[str, Any]:
        """
        Redact properties to remove any paths, so as to avoid inadvertently collecting
        private or personally identifiable information (PII). We also remove
        information related to Wandb, MLflow, etc. configuration.

        Args:
            properties: Dictionary of properties to redact.

        Returns:
            Properties dictionary with redaction applied.
        """
        if not properties:
            return {}

        def redact_value(value: Any, key: str = "") -> Any:
            """Recursively sanitize values, redacting those with path-like keys"""
            if isinstance(key, str) and isinstance(value, str):
                # Other redaction special cases
                if (
                    key in FIELDS_TO_REDACT
                    or any(prefix in key for prefix in PREFIXES_TO_REDACT)
                    or any(indicator in key.lower() for indicator in PATH_INDICATORS)
                ):
                    # Fields with whitelisted orgs don't need to be redacted
                    if not self._is_whitelisted(value):
                        return "[REDACTED]"

            # Handle nested values
            if isinstance(value, dict):
                return {k: redact_value(v, k) for k, v in value.items()}
            if isinstance(value, list):
                return [redact_value(item) for item in value]

            return value

        # Create new dict with redacted values
        redacted = {k: redact_value(v, k) for k, v in properties.items()}

        return redacted

    def _get_system_info(self) -> dict[str, Any]:
        """Collect system information for various hardware accelerators"""
        gpu_info = []
        accelerator_type = "none"

        # NVIDIA GPUs
        if torch.cuda.is_available():
            accelerator_type = "cuda"
            for i in range(torch.cuda.device_count()):
                gpu_info.append(
                    {
                        "name": torch.cuda.get_device_name(i),
                        "memory": torch.cuda.get_device_properties(i).total_memory,
                    }
                )

        # AMD GPUs
        elif hasattr(torch, "hip") and torch.hip.is_available():
            accelerator_type = "hip"
            for i in range(torch.hip.device_count()):
                gpu_info.append(
                    {
                        "name": torch.hip.get_device_name(i),
                        "memory": (
                            torch.hip.get_device_properties(i).total_memory
                            if hasattr(torch.hip, "get_device_properties")
                            else None
                        ),
                    }
                )

        # Apple Silicon
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            accelerator_type = "mps"
            gpu_info.append(
                {
                    "name": "Apple Silicon",
                    # NOTE: this is memory allocated to this process, not total memory
                    "memory": torch.mps.driver_allocated_memory(),
                }
            )

        # Intel GPUs
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            accelerator_type = "xpu"
            for i in range(torch.xpu.device_count()):
                memory = None
                if hasattr(torch.xpu, "get_device_properties"):
                    memory = torch.xpu.get_device_properties(i).total_memory

                gpu_info.append(
                    {
                        "name": torch.xpu.get_device_name(i),
                        "memory": memory,
                    }
                )

        # NPUs
        elif hasattr(torch, "npu") and torch.npu.is_available():
            accelerator_type = "npu"
            for i in range(torch.npu.device_count()):
                memory = None
                if hasattr(torch.npu, "get_device_properties"):
                    memory = torch.npu.get_device_properties(i).total_memory

                gpu_info.append(
                    {
                        "name": torch.npu.get_device_name(i),
                        "memory": memory,
                    }
                )

        # Get relevant package versions
        installed_packages = {}
        for package in RELEVANT_PACKAGES:
            try:
                version = importlib.metadata.version(package)
                installed_packages[f"{package}_version"] = version
            except importlib.metadata.PackageNotFoundError:
                pass

        return {
            "os": platform.system(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "accelerator_type": accelerator_type,
            "accelerator_count": len(gpu_info),
            "accelerator_info": gpu_info,
            **installed_packages,
        }

    def send_event(self, event_type: str, properties: dict[str, Any] | None = None):
        """Send a telemetry event"""
        if not self.enabled:
            return

        if properties is None:
            properties = {}

        # Sanitize properties to remove PII
        properties = self._redact_paths(properties)

        # Wrap PostHog errors in try / except to not raise errors during Axolotl usage
        try:
            # Send event via PostHog
            posthog.capture(
                distinct_id=self.run_id,
                event=event_type,
                properties=properties,
                disable_geoip=True,
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            LOG.warning(f"Failed to send telemetry event: {e}")

        # Additionally, send system info telemetry when loading config.
        # NOTE: Is this the best place for this?
        if event_type == "config-loaded":
            self.send_system_info()

    def send_system_info(self):
        """Helper method for sending system info"""
        if self.system_info is not None:
            self.send_event(event_type="system-info", properties=self.system_info)

    def shutdown(self):
        """Ensure all queued events are processed before shutdown"""
        if self.enabled:
            posthog.shutdown()
