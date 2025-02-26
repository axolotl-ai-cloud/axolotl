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

OPT_IN_WARNING_SLEEP_SECONDS = 15
OPT_IN_INFO = (
    "\nTelemetry is currently disabled by default. If you'd like to help improve "
    "Axolotl, consider enabling it by setting:\n"
    "AXOLOTL_DO_NOT_TRACK=0\n\n"
    "Telemetry data helps us understand:\n"
    "- Which features are most used\n"
    "- What hardware configurations to prioritize\n"
    "- Where users encounter errors\n\n"
    "No personally identifiable information is collected.\n"
    "To remove this warning, explicitly set AXOLOTL_DO_NOT_TRACK=0 (enable telemetry) "
    "or AXOLOTL_DO_NOT_TRACK=1 (explicitly disable telemetry).\n\n"
    "NOTE: Telemetry will move to an opt-out in a later release.\n"
    "For details, see: https://axolotl-ai-cloud.github.io/axolotl/docs/telemetry.html\n"
    f"Sleeping for {OPT_IN_WARNING_SLEEP_SECONDS}s..."
)

WHITELIST_PATH = str(Path(__file__).parent / "whitelist.yaml")

# NOTE: Keep these up to date with any config schema changes
FIELDS_WITH_ORGS = {
    "base_model",
    "tokenizer_config",
    "base_model_config",
    "pretraining_dataset",  # NOTE: this field may be a string or a dictionary
}
FIELDS_TO_REDACT = {"resume_from_checkpoint", "hub_model_id"}
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

        Note: This is disabled by default on an opt-in basis. Set
        `AXOLOTL_DO_NOT_TRACK=0` to enable telemetry. We plan to move to an opt-out
        model in a later release. For more details, see
        https://axolotl-ai-cloud.github.io/axolotl/docs/telemetry.html.

        Returns:
            Tuple containing:
                - Boolean denoting whether telemetry is enabled or not.
        """
        # Parse relevant env vars and fill opt-out default values
        axolotl_do_not_track = os.getenv("AXOLOTL_DO_NOT_TRACK")
        do_not_track = os.getenv("DO_NOT_TRACK")

        # Default to disabled (opt-in model for initial release)
        if axolotl_do_not_track is None:
            # Print opt-in info message for main process only
            if is_main_process():
                LOG.info(OPT_IN_INFO)
            time.sleep(OPT_IN_WARNING_SLEEP_SECONDS)

            return False

        if do_not_track is None:
            do_not_track = "0"

        # Respect AXOLOTL_DO_NOT_TRACK, DO_NOT_TRACK if enabled
        enabled = axolotl_do_not_track.lower() not in (
            "1",
            "true",
        ) and do_not_track.lower() not in ("1", "true")

        # Only rank 0 will send telemetry
        if not is_main_process():
            return False

        return enabled

    def _load_whitelist(self) -> dict:
        """Load HuggingFace Hub organization whitelist"""
        with open(WHITELIST_PATH, encoding="utf-8") as f:
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
        posthog.host = POSTHOG_HOST
        posthog.project_api_key = POSTHOG_WRITE_KEY

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
                # Fields that should be redacted if org is not whitelisted
                if key in FIELDS_WITH_ORGS:
                    org = value.split("/")[0]
                    if org not in self.whitelist["organizations"]:
                        return "[REDACTED]"

                # Other redaction special cases
                if (
                    key in FIELDS_TO_REDACT
                    or any(prefix in key for prefix in PREFIXES_TO_REDACT)
                    or any(indicator in key.lower() for indicator in PATH_INDICATORS)
                ):
                    return "[REDACTED]"

            # Handle nested structures
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
                        "memory": torch.hip.get_device_properties(i).total_memory
                        if hasattr(torch.hip, "get_device_properties")
                        else None,
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
        self.send_event(event_type="system-info", properties=self.system_info)

    def shutdown(self):
        """Ensure all queued events are processed before shutdown"""
        if self.enabled:
            posthog.flush()
