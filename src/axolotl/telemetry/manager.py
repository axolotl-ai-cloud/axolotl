"""Telemetry manager and associated utilities."""

import logging
import os
import platform
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import posthog
import psutil
import torch
import yaml

logger = logging.getLogger(__name__)

POSTHOG_WRITE_KEY = "phc_RbAa7Bxu6TLIN9xd8gbg1PLemrStaymi8pxQbRbIwfC"


@dataclass
class ModelConfig:
    """Tracked model configuration details"""

    base_model: str
    model_type: str
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    tokenizer_config: dict
    flash_attention: bool
    quantization_config: dict | None
    training_approach: str  # 'lora', 'qlora', 'full_finetune'

    @classmethod
    def from_config(cls, config: dict) -> "ModelConfig":
        """Create from Axolotl config dict"""
        return cls(
            base_model=config.get("base_model", ""),
            model_type=config.get("model_type", ""),
            hidden_size=config.get("hidden_size", 0),
            num_layers=config.get("num_layers", 0),
            num_attention_heads=config.get("num_attention_heads", 0),
            tokenizer_config=config.get("tokenizer", {}),
            flash_attention=config.get("flash_attention", False),
            quantization_config=config.get("quantization", None),
            training_approach=config.get("training_approach", ""),
        )

    def to_dict(self) -> dict:
        """Convert to PostHog-compatible dict"""
        return {
            "base_model": self.base_model,
            "model_type": self.model_type,
            "architecture": {
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "num_attention_heads": self.num_attention_heads,
            },
            "optimizations": {
                "flash_attention": self.flash_attention,
                "quantization": self.quantization_config is not None,
                "quantization_config": self.quantization_config,
            },
            "training_approach": self.training_approach,
        }


@dataclass
class TelemetryConfig:
    """Configuration for telemetry manager"""

    host: str = "https://app.posthog.com"
    queue_size: int = 100
    batch_size: int = 10
    whitelist_path: str = str(Path(__file__).parent / "whitelist.yaml")
    retention_days: int = 365
    distinct_id: str = str(uuid.uuid4())
    schema_version: str = "0.1.0"


class TelemetryManager:
    """Manages telemetry collection and transmission"""

    def __init__(self, config: TelemetryConfig):
        """
        Telemetry manager constructor.

        Args:
            config: Telemetry configuration object.
        """
        self.config = config
        self.run_id = str(uuid.uuid4())
        self.enabled = self._check_telemetry_enabled()

        if self.enabled:
            self.whitelist = self._load_whitelist()
            self._init_posthog()

    def _check_telemetry_enabled(self) -> bool:
        """
        Check if telemetry is enabled based on environment variables.

        Note: This is enabled on an opt-in basis. Please consider setting
        `AXOLOTL_TELEMETRY=1` to send us valuable data on which models and algos you're
        using so we can focus our engineering efforts!
        """
        # Only enable if explicitly opted in
        axolotl_telemetry = os.getenv("AXOLOTL_TELEMETRY", "0").lower() in ("1", "true")

        # Respect DO_NOT_TRACK as an override even if telemetry is enabled
        do_not_track = os.getenv("DO_NOT_TRACK", "0").lower() in ("1", "true")

        return axolotl_telemetry and not do_not_track

    def _load_whitelist(self) -> dict:
        """Load organization/model whitelist"""
        with open(self.config.whitelist_path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _is_whitelisted(self, base_model: str) -> bool:
        """Check if model/org is in whitelist"""
        if not base_model:
            return False

        base_model = base_model.lower()
        return any(
            org.lower() in base_model 
            for org in self.whitelist.get("organizations", [])
        )

    def _init_posthog(self):
        """Initialize PostHog client"""
        posthog.project_api_key = POSTHOG_WRITE_KEY
        posthog.host = self.config.host

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

            # Send event via PostHog
            try:
                posthog.capture(
                    distinct_id=self.config.distinct_id,
                    event=event_type,
                    properties={
                        "run_id": self.run_id,
                        "system_info": system_info,
                        **properties,
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to send telemetry event: {e}")
        except (RuntimeError, OSError) as e:
            logger.warning(f"Failed to collect system info for telemetry: {e}")
        except TypeError as e:
            logger.warning(f"Invalid property type in telemetry event: {e}")
        except AttributeError as e:
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

    def track_model_load(self, model_config: ModelConfig):
        """Track model loading and configuration"""
        if not self.enabled or not self._is_whitelisted(model_config.base_model):
            return

        self.track_event(
            "model_load",
            {
                "model_config": model_config.to_dict(),
                "system_info": self._get_system_info(),
            },
        )

    def track_training_metrics(self, metrics: dict):
        """Track training progress metrics"""
        if not self.enabled:
            return

        self.track_event(
            "training_metrics",
            {
                "duration": metrics.get("duration"),
                "peak_memory": metrics.get("peak_memory"),
                "steps_completed": metrics.get("steps_completed"),
                "current_loss": metrics.get("loss"),
                "learning_rate": metrics.get("learning_rate"),
            },
        )

    def shutdown(self):
        """Ensure all queued events are processed before shutdown"""
        if self.enabled:
            posthog.flush()


def init_telemetry_manager() -> TelemetryManager:
    """Initialize telemetry system"""
    return TelemetryManager(TelemetryConfig())