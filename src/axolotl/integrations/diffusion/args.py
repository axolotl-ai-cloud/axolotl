"""Config args for diffusion LM training (nested under `diffusion:`)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class DiffusionConfig(BaseModel):
    """Nested diffusion configuration available under the `diffusion` key."""

    # Noise schedule config
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
        default=128, ge=1, description="Number of diffusion timesteps"
    )
    eps: float = Field(
        default=1e-3,
        ge=0.0,
        le=1.0,
        description="Epsilon value for minimum masking probability in forward process",
    )

    # Training config
    importance_weighting: bool = Field(
        default=True,
        description="Apply importance weighting to loss based on masking probability",
    )
    mask_token_id: int | None = Field(
        default=None,
        description=(
            "Token ID to use for masking. Unset by default; can use one of the "
            "tokenizer's special tokens here."
        ),
    )
    mask_token_str: str | None = Field(
        default=None,
        description=(
            "Token string to use as a mask. If `mask_token_id` is invalid or unset, "
            "this token will be ensured to exist as an additional special token and "
            "used. If absent, a default '<|diffusion_mask|>' will be added."
        ),
    )

    # Sample generation config
    generate_samples: bool = Field(
        default=True, description="Enable sample generation during training"
    )
    generation_interval: int = Field(
        default=100, ge=1, description="Generate samples every N steps"
    )
    num_generation_samples: int = Field(
        default=3, ge=1, description="Number of samples to generate each time"
    )
    generation_steps: int = Field(
        default=128, ge=1, description="Number of diffusion steps for generation"
    )
    generation_temperature: float = Field(
        default=0.0,
        ge=0.0,
        description="Temperature for generation sampling (0.0 = deterministic)",
    )
    generation_max_length: int = Field(
        default=100, ge=1, description="Maximum sequence length for generation"
    )

    @model_validator(mode="after")
    def _validate_mask_ratios(self) -> "DiffusionConfig":
        if self.min_mask_ratio > self.max_mask_ratio:
            raise ValueError("min_mask_ratio must be ≤ max_mask_ratio")
        return self


class DiffusionArgs(BaseModel):
    """Plugin entry that exposes the nested `diffusion` block to the core config."""

    diffusion: DiffusionConfig = Field(
        default_factory=DiffusionConfig,
        description="Diffusion training configuration. Only nested block is supported.",
    )
