"""Config args for diffusion LM training."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class DiffusionArgs(BaseModel):
    """Arguments for diffusion LM training plugin."""

    # Noise schedule config
    diffusion_noise_schedule: Literal["linear", "cosine"] = Field(
        default="linear", description="Type of noise schedule for diffusion training"
    )
    diffusion_min_mask_ratio: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Minimum masking ratio for diffusion noise schedule",
    )
    diffusion_max_mask_ratio: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Maximum masking ratio for diffusion noise schedule",
    )
    diffusion_num_diffusion_steps: int = Field(
        default=128, ge=1, description="Number of diffusion timesteps"
    )
    diffusion_eps: float = Field(
        default=1e-3,
        ge=0.0,
        le=1.0,
        description="Epsilon value for minimum masking probability in forward process",
    )

    # Training config
    diffusion_importance_weighting: bool = Field(
        default=True,
        description="Apply importance weighting to loss based on masking probability",
    )
    diffusion_mask_token_id: int | None = Field(
        default=None,
        description=(
            "Token ID to use for masking. Unset by default; can use one of the "
            "tokenizer's special tokens here."
        ),
    )
    diffusion_mask_token_str: str | None = Field(
        default=None,
        description=(
            "Token string to use as a mask. If `diffusion_mask_token_id` is invalid "
            "or unset, this token will be ensured to exist as an additional special "
            "token and used. If absent, a default '<|diffusion_mask|>' will be added."
        ),
    )

    # Sample generation config
    diffusion_generate_samples: bool = Field(
        default=True, description="Enable sample generation during training"
    )
    diffusion_generation_interval: int = Field(
        default=100, ge=1, description="Generate samples every N steps"
    )
    diffusion_num_generation_samples: int = Field(
        default=3, ge=1, description="Number of samples to generate each time"
    )
    diffusion_generation_steps: int = Field(
        default=128, ge=1, description="Number of diffusion steps for generation"
    )
    diffusion_generation_temperature: float = Field(
        default=0.0,
        ge=0.0,
        description="Temperature for generation sampling (0.0 = deterministic)",
    )
    diffusion_generation_max_length: int = Field(
        default=100, ge=1, description="Maximum sequence length for generation"
    )

    @model_validator(mode="after")
    def _validate_mask_ratios(self) -> DiffusionArgs:
        if self.diffusion_min_mask_ratio > self.diffusion_max_mask_ratio:
            raise ValueError(
                "diffusion_min_mask_ratio must be ≤ diffusion_max_mask_ratio"
            )
        return self
