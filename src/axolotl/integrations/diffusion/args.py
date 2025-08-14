"""Configuration arguments for diffusion LM training."""

from typing import Literal

from pydantic import BaseModel, Field


class DiffusionArgs(BaseModel):
    """Arguments for diffusion LM training plugin."""

    # Noise schedule configuration
    noise_schedule: Literal["linear", "cosine"] = Field(
        default="linear", description="Type of noise schedule for diffusion training"
    )
    min_mask_ratio: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Minimum masking ratio for diffusion noise schedule",
    )
    max_mask_ratio: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Maximum masking ratio for diffusion noise schedule",
    )
    num_diffusion_steps: int = Field(
        default=1000, ge=1, description="Number of diffusion timesteps"
    )

    # Forward process parameters
    eps: float = Field(
        default=1e-3,
        ge=0.0,
        le=1.0,
        description="Epsilon value for minimum masking probability in forward process",
    )

    # Training configuration
    importance_weighting: bool = Field(
        default=True,
        description="Apply importance weighting to loss based on masking probability",
    )
