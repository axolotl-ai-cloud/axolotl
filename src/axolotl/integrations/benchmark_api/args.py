"""
Config schema for the external benchmark API plugin.
"""

import re
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# alias -> normalized mode
_MODE_ALIASES = {
    "lower": "min",
    "min": "min",
    "smaller": "min",
    "decrease": "min",
    "higher": "max",
    "max": "max",
    "larger": "max",
    "increase": "max",
}


class EarlyStoppingConfig(BaseModel):
    """
    Generic early stopping on a single benchmark metric.
    """

    model_config = ConfigDict(extra="forbid")  # surface typo'd config keys

    enabled: bool = False
    metric: Optional[str] = None
    mode: Literal["min", "max"] = "min"
    patience: int = Field(default=3, ge=1)
    min_delta: float = Field(default=0.0, ge=0.0)
    threshold: Optional[float] = None

    @field_validator("mode", mode="before")
    @classmethod
    def _normalize_mode(cls, value):
        if isinstance(value, str):
            normalized = _MODE_ALIASES.get(value.strip().lower())
            if normalized is None:
                raise ValueError(
                    f"invalid early_stopping mode {value!r}; "
                    f"expected one of {sorted(_MODE_ALIASES)}"
                )
            return normalized
        return value

    @model_validator(mode="after")
    def _require_metric(self):
        if self.enabled and not self.metric:
            raise ValueError(
                "benchmark_api.early_stopping.metric is required when enabled is true"
            )
        return self


class BenchmarkAPIConfig(BaseModel):
    """
    Settings for the external benchmark runner webhook.
    """

    model_config = ConfigDict(extra="forbid")  # surface typo'd config keys

    endpoint: str
    # name of an env var holding a bearer token; sent as an Authorization header
    auth_env: Optional[str] = None
    execution_mode: Literal["sync", "async"] = "sync"
    poll_interval_steps: int = Field(default=10, gt=0)
    run_on: List[Literal["save", "eval", "train_end"]] = Field(default=["save"])
    timeout_sec: int = Field(default=3600, ge=0)  # 0 = no timeout (wait indefinitely)
    fail_training_on_error: bool = False
    early_stopping: Optional[EarlyStoppingConfig] = None

    @field_validator("endpoint")
    @classmethod
    def _require_http_scheme(cls, value: str) -> str:
        if not re.match(r"^https?://", value):
            raise ValueError(
                "benchmark_api.endpoint must start with http:// or https://"
            )
        return value


class BenchmarkAPIArgs(BaseModel):
    """
    Top-level `benchmark_api:` config block.
    """

    benchmark_api: Optional[BenchmarkAPIConfig] = None
