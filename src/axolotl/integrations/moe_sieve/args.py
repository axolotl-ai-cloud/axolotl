"""Configuration for routing-guided expert LoRA."""

from pydantic import BaseModel, Field


class MoeSieveConfig(BaseModel):
    """Calibration and selection settings."""

    selection_file: str | None = None
    fraction: float = Field(default=0.25, gt=0, le=1)
    calibration_samples: int = Field(default=256, gt=0)


class MoeSieveArgs(BaseModel):
    """Plugin configuration schema."""

    moe_sieve: MoeSieveConfig = Field(default_factory=MoeSieveConfig)
